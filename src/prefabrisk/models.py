"""Optimisation models of the case study (python-mip, CBC solver).

Each function follows the corresponding notebook cell:
  solve_deterministic       Task 1   expected completion fractions
  solve_chance_constrained  Task 2b  joint chance constraint, big-M
  solve_two_stage           Task 3a  emergency crew as recourse
  solve_expected_value      Task 3b  EV problem (expected alphas, one scenario)
  solve_insurance           Task 4   two-stage model plus insurance choice
Passing fixed_x to the two-stage and insurance models evaluates a given
first-stage decision instead of optimising it (used for EEV and VSS).
"""
import mip
import numpy as np
from mip import minimize


def _crew_params(config):
    crews = config["crews"]
    return (crews["c_fixed"], crews["c_var"],
            crews["minimum_contracts"], crews["maximum_capacities"])


def _first_stage_vars(model, config, fixed_x=None):
    c_fixed, c_var, minimum_contracts, maximum_capacities = _crew_params(config)
    assembly_crews = range(len(c_fixed))
    if fixed_x is None:
        return [model.add_var(name=f"installation units {i}", lb=minimum_contracts[i],
                              ub=maximum_capacities[i]) for i in assembly_crews]
    return [model.add_var(name=f"installation units {i}", lb=fixed_x[i], ub=fixed_x[i])
            for i in assembly_crews]


def solve_deterministic(config, expected_alpha):
    """Task 1: allocate exactly target_units using E[alpha]."""
    c_fixed, c_var, _, _ = _crew_params(config)
    assembly_crews = range(len(c_fixed))
    target_units = config["project"]["target_units"]
    penalty_per_unit = config["project"]["penalty_per_unit"]

    m_det = mip.Model("deterministic")
    m_det.verbose = 0
    x_det = _first_stage_vars(m_det, config)
    m_det.objective = minimize(sum((c_fixed[i] * x_det[i]) + (c_var[i] * x_det[i] * expected_alpha[i])
                                   + (penalty_per_unit * x_det[i] * (1 - expected_alpha[i]))
                                   for i in assembly_crews))
    m_det += sum(x_det[i] for i in assembly_crews) == target_units
    m_det.optimize()
    return {"x": np.array([v.x for v in x_det]), "objective": m_det.objective_value}


def solve_chance_constrained(config, alpha_200):
    """Task 2b: cover target_units in at least confidence_level of the scenarios."""
    c_fixed, c_var, _, _ = _crew_params(config)
    assembly_crews = range(len(c_fixed))
    target_units = config["project"]["target_units"]
    confidence_level = config["chance_constraint"]["confidence_level"]
    scenarios_200 = range(alpha_200.shape[0])
    prob = [1.0 / alpha_200.shape[0] for s in scenarios_200]

    m_joint = mip.Model("joint chance-constrained")
    m_joint.verbose = 0
    x_joint = _first_stage_vars(m_joint, config)
    y = [m_joint.add_var(var_type=mip.BINARY, name=f"y_{s}") for s in scenarios_200]
    m_joint.objective = minimize(sum(c_fixed[i] * x_joint[i] for i in assembly_crews)
                                 + sum(c_var[i] * x_joint[i] * alpha_200[s, i] * prob[s]
                                       for i in assembly_crews for s in scenarios_200))
    bigM = target_units
    for s in scenarios_200:
        m_joint += sum(x_joint[i] * alpha_200[s, i] for i in assembly_crews) + bigM * y[s] >= target_units
    m_joint += sum(prob[s] * y[s] for s in scenarios_200) <= 1 - confidence_level
    status = m_joint.optimize()
    if status not in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
        return {"x": None, "objective": None, "status": status.name}
    return {"x": np.array([v.x for v in x_joint]), "objective": m_joint.objective_value,
            "status": status.name}


def solve_two_stage(config, alpha_200, fixed_x=None):
    """Task 3a: minimise expected cost with the emergency crew as recourse."""
    c_fixed, c_var, _, _ = _crew_params(config)
    assembly_crews = range(len(c_fixed))
    target_units = config["project"]["target_units"]
    penalty_per_unit = config["project"]["penalty_per_unit"]
    premium_rate = config["emergency_crew"]["premium_rate"]
    maximum_emergency_capacity = config["emergency_crew"]["maximum_emergency_capacity"]
    scenarios_200 = range(alpha_200.shape[0])
    prob = [1.0 / alpha_200.shape[0] for s in scenarios_200]

    m_two = mip.Model("two-stage")
    m_two.verbose = 0
    x_two = _first_stage_vars(m_two, config, fixed_x)
    emergency_two = [m_two.add_var(name=f"emergency crew scenario {s}", lb=0,
                                   ub=maximum_emergency_capacity) for s in scenarios_200]
    penalty_two = [m_two.add_var(name=f"penalty cost {s}", lb=0) for s in scenarios_200]
    scenario_cost_two = [m_two.add_var(name=f"expected cost {s}", lb=-np.inf) for s in scenarios_200]

    m_two.objective = minimize(sum(prob[s] * scenario_cost_two[s] for s in scenarios_200))
    for s in scenarios_200:
        installed_units = sum(alpha_200[s, i] * x_two[i] for i in assembly_crews)
        m_two += penalty_two[s] >= target_units - emergency_two[s] - installed_units
        m_two += scenario_cost_two[s] == (sum(c_fixed[i] * x_two[i] + c_var[i] * x_two[i] * alpha_200[s, i]
                                              for i in assembly_crews)
                                          + penalty_per_unit * penalty_two[s] + premium_rate * emergency_two[s])
    m_two.optimize()
    return {"x": np.array([v.x for v in x_two]), "objective": m_two.objective_value,
            "expected_emergency": sum(prob[s] * emergency_two[s].x for s in scenarios_200),
            "scenario_costs": np.array([scenario_cost_two[s].x for s in scenarios_200])}


def solve_expected_value(config, expected_alpha):
    """Task 3b: the two-stage model with every alpha replaced by its mean."""
    result = solve_two_stage(config, np.asarray(expected_alpha).reshape(1, -1))
    return {"x": result["x"], "objective": result["objective"]}


def value_of_stochastic_solution(config, alpha_200):
    """VSS = EEV - RP, all evaluated on the same scenarios."""
    rp = solve_two_stage(config, alpha_200)
    ev = solve_expected_value(config, alpha_200.mean(axis=0))
    eev = solve_two_stage(config, alpha_200, fixed_x=ev["x"])
    return {"rp": rp["objective"], "eev": eev["objective"], "vss": eev["objective"] - rp["objective"],
            "x_rp": rp["x"], "x_ev": ev["x"]}


def solve_insurance(config, alpha_200, fixed_x=None, fixed_insurance=None):
    """Task 4: two-stage model plus the option to buy weather insurance.

    The coverage constraint is '>=' as in Task 3a. The notebook used '==',
    which forbids installing more than target_units in calm scenarios and
    therefore caps the total allocation at target_units.
    """
    c_fixed, c_var, _, _ = _crew_params(config)
    assembly_crews = range(len(c_fixed))
    target_units = config["project"]["target_units"]
    penalty_per_unit = config["project"]["penalty_per_unit"]
    premium_rate = config["emergency_crew"]["premium_rate"]
    maximum_emergency_capacity = config["emergency_crew"]["maximum_emergency_capacity"]
    insurance_premium = config["insurance"]["insurance_premium"]
    maximum_reimbursement = config["insurance"]["maximum_reimbursement"]
    scenarios_200 = range(alpha_200.shape[0])
    prob = [1.0 / alpha_200.shape[0] for s in scenarios_200]

    m_insurance = mip.Model("insurance")
    m_insurance.verbose = 0
    x_insurance = _first_stage_vars(m_insurance, config, fixed_x)
    if fixed_insurance is None:
        insurance = m_insurance.add_var(name="insurance", var_type=mip.BINARY)
    else:
        insurance = m_insurance.add_var(name="insurance", lb=fixed_insurance, ub=fixed_insurance)
    emergency_insurance = [m_insurance.add_var(name=f"emergency crew scenario {s}", lb=0,
                                               ub=maximum_emergency_capacity) for s in scenarios_200]
    penalty_insurance = [m_insurance.add_var(name=f"penalty cost {s}", lb=0) for s in scenarios_200]
    reimbursement = [m_insurance.add_var(name=f"reimbursement {s}", lb=0, ub=maximum_reimbursement)
                     for s in scenarios_200]
    cost_insurance = [m_insurance.add_var(name=f"expected cost {s}", lb=-np.inf) for s in scenarios_200]

    m_insurance.objective = minimize(sum(prob[s] * cost_insurance[s] for s in scenarios_200)
                                     + insurance_premium * insurance)
    for s in scenarios_200:
        installed_units = sum(alpha_200[s, i] * x_insurance[i] for i in assembly_crews)
        m_insurance += installed_units + emergency_insurance[s] + penalty_insurance[s] >= target_units
        m_insurance += cost_insurance[s] == (sum(c_fixed[i] * x_insurance[i] + c_var[i] * x_insurance[i] * alpha_200[s, i]
                                                 for i in assembly_crews)
                                             + penalty_per_unit * penalty_insurance[s]
                                             + premium_rate * emergency_insurance[s] - reimbursement[s])
        m_insurance += reimbursement[s] <= insurance * maximum_reimbursement
        m_insurance += reimbursement[s] <= penalty_per_unit * penalty_insurance[s]
    m_insurance.optimize()
    buy_insurance = round(insurance.x)
    return {"x": np.array([v.x for v in x_insurance]), "insurance": buy_insurance,
            "objective": m_insurance.objective_value,
            "expected_emergency": sum(prob[s] * emergency_insurance[s].x for s in scenarios_200),
            "expected_reimbursement": sum(prob[s] * reimbursement[s].x for s in scenarios_200),
            "scenario_costs": np.array([cost_insurance[s].x for s in scenarios_200])
                              + insurance_premium * buy_insurance}