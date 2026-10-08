"""Out-of-sample evaluation and risk measures (Tasks 2a and 5).

total_cost replays the second stage for one scenario with the cheapest
recourse order: with insurance, penalty units up to the reimbursement cap
cost nothing net, so they come first; then the emergency crew (16 EUR),
then uncovered penalties (40 EUR).
"""
import numpy as np


def total_cost(config, x, alpha_scenario, insurance):
    c_fixed, c_var = config["crews"]["c_fixed"], config["crews"]["c_var"]
    assembly_crews = range(len(c_fixed))
    target_units = config["project"]["target_units"]
    penalty_per_unit = config["project"]["penalty_per_unit"]
    premium_rate = config["emergency_crew"]["premium_rate"]
    maximum_emergency_capacity = config["emergency_crew"]["maximum_emergency_capacity"]
    insurance_premium = config["insurance"]["insurance_premium"]
    maximum_reimbursement = config["insurance"]["maximum_reimbursement"]

    installed_units = sum(alpha_scenario[i] * x[i] for i in assembly_crews)
    gap = max(target_units - installed_units, 0)

    if insurance == 1:
        free_penalty_units = min(gap, maximum_reimbursement / penalty_per_unit)
        rest = gap - free_penalty_units
        emergency_units = min(rest, maximum_emergency_capacity)
        shortfall_units = free_penalty_units + (rest - emergency_units)
        reimbursement_s = min(penalty_per_unit * shortfall_units, maximum_reimbursement)
    else:
        emergency_units = min(gap, maximum_emergency_capacity)
        shortfall_units = gap - emergency_units
        reimbursement_s = 0

    fixed_cost = sum(c_fixed[i] * x[i] for i in assembly_crews)
    variable_cost = sum(c_var[i] * alpha_scenario[i] * x[i] for i in assembly_crews)

    return (fixed_cost + variable_cost
            + premium_rate * emergency_units
            + penalty_per_unit * shortfall_units
            - reimbursement_s
            + insurance_premium * insurance)


def evaluate_costs(config, x, alpha, insurance=0):
    """Realised total cost of decision x in every scenario of alpha."""
    return np.array([total_cost(config, x, alpha[s], insurance) for s in range(alpha.shape[0])])


def probability_of_shortfall(config, x, alpha):
    """Share of scenarios where the primary crews alone install fewer than target_units."""
    target_units = config["project"]["target_units"]
    return np.mean(alpha @ np.asarray(x) < target_units)


def var_q(outcomes, q):
    """Value-at-Risk of a cost: the (100 - q)% quantile, i.e. the upper tail."""
    return np.percentile(outcomes, 100 - q)


def cvar_q(outcomes, q):
    """Conditional VaR: mean cost in the worst q% of scenarios."""
    return np.mean(outcomes[outcomes >= var_q(outcomes, q)])


def risk_summary(config, costs):
    q = config["risk"]["q"]
    threshold = config["risk"]["cost_threshold"]
    return {"mean": float(np.mean(costs)),
            "p_exceed": float(np.mean(costs > threshold)),
            "var": float(var_q(costs, q)),
            "cvar": float(cvar_q(costs, q))}