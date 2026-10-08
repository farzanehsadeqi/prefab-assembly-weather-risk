"""End-to-end analysis: calibrate, optimise and evaluate for every case.

Cases:
  course  - the simulator of the assignment (rho 0.65, Weibull 2 / 25)
  summer  - simulator calibrated on Bielefeld 100 m wind, May-August
  autumn  - the same, October-November

For each case the four models are solved on 200 scenarios and every
decision is evaluated on 10,000 fresh simulated scenarios and, for the
calibrated cases, on all historical 20-day windows of that season.

Usage:
  python scripts/run_pipeline.py            # use cached wind data
  python scripts/run_pipeline.py --refresh  # download wind data again
"""
import argparse
import json

import numpy as np
import pandas as pd

from prefabrisk.calibration import calibrate_season
from prefabrisk.config import PROJECT_ROOT, load_config
from prefabrisk.plots import plot_allocations, plot_alpha_distribution, plot_cost_distributions

from prefabrisk.models import (solve_chance_constrained, solve_deterministic, solve_insurance,
                               solve_two_stage, value_of_stochastic_solution)
from prefabrisk.risk import evaluate_costs, probability_of_shortfall, risk_summary
from prefabrisk.scenarios import completion_coefficients, generate_weather_scenarios, historical_windows
from prefabrisk.weather import load_wind_history

RESULTS_DIR = PROJECT_ROOT / "results"


def build_cases(config, wind_daily):
    """Simulator parameters and historical windows for each case."""
    cases = {"course": {"simulator": dict(config["course_simulator"]), "historical": None}}
    variable = config["weather_data"]["variable"]
    num_days = config["project"]["num_days"]
    for season, months in config["weather_data"]["seasons"].items():
        p = calibrate_season(wind_daily, variable, months)
        cases[season] = {
            "simulator": {"rho": p["rho"], "wind_values": p["wind_values"]},
            "calibration": {k: v for k, v in p.items() if k != "wind_values"},
            "historical": historical_windows(wind_daily, variable, months, num_days),
        }
    return cases


def run_case(config, case):
    sc = config["scenarios"]
    num_days = config["project"]["num_days"]
    thresholds = config["crews"]["wind_thresholds"]

    def alphas(n, seed):
        weather = generate_weather_scenarios(n, seed, num_days, **case["simulator"])
        return completion_coefficients(weather, thresholds)

    alpha_estimation = alphas(sc["n_sample_10k"], sc["seed_estimation"])
    alpha_200 = alphas(sc["n_sample_200"], sc["seed_optimization"])
    alpha_10k = alphas(sc["n_sample_10k"], sc["seed_validation"])
    alpha_hist = None
    if case["historical"] is not None:
        alpha_hist = completion_coefficients(case["historical"], thresholds)

    det = solve_deterministic(config, alpha_estimation.mean(axis=0))
    cc = solve_chance_constrained(config, alpha_200)
    two = solve_two_stage(config, alpha_200)
    vss = value_of_stochastic_solution(config, alpha_200)
    ins = solve_insurance(config, alpha_200)

    decisions = {
        "deterministic": (det["x"], 0, det["objective"]),
        "chance_constrained": (cc["x"], 0, cc["objective"]),
        "two_stage": (two["x"], 0, two["objective"]),
        "insurance": (ins["x"], ins["insurance"], ins["objective"]),
    }
    out = {"expected_alpha": alpha_estimation.mean(axis=0).tolist(),
           "chance_constrained_status": cc["status"],
           "vss": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vss.items()},
           "insurance_expected_reimbursement": ins["expected_reimbursement"],
            "decisions": {}, "costs": {}, "alpha_10k": alpha_10k, "alpha_hist": alpha_hist}
    for name, (x, insurance, objective) in decisions.items():
        if x is None:
            out["decisions"][name] = {"x": None, "status": "infeasible"}
            continue
        entry = {"x": x.tolist(), "insurance": int(insurance), "in_sample_objective": objective}
        costs_sim = evaluate_costs(config, x, alpha_10k, insurance)
        entry["simulated"] = risk_summary(config, costs_sim)
        entry["simulated"]["p_shortfall"] = float(probability_of_shortfall(config, x, alpha_10k))
        out["costs"][name] = {"simulated": costs_sim}
        if alpha_hist is not None:
            costs_hist = evaluate_costs(config, x, alpha_hist, insurance)
            entry["historical"] = risk_summary(config, costs_hist)
            entry["historical"]["p_shortfall"] = float(probability_of_shortfall(config, x, alpha_hist))
            out["costs"][name]["historical"] = costs_hist
        out["decisions"][name] = entry
    return out


def print_case(config, name, result):
    names = config["crews"]["crew_names"]
    print(f"\n=== {name} ===  E[alpha] = {np.round(result['expected_alpha'], 3)}   "
          f"VSS = {result['vss']['vss']:,.0f}")
    rows = {}
    for model, d in result["decisions"].items():
        if d["x"] is None:
            rows[model] = {"note": "infeasible"}
            continue
        row = dict(zip(names, np.round(d["x"]).astype(int)))
        row["ins"] = d["insurance"]
        for source in ["simulated", "historical"]:
            if source in d:
                tag = "sim" if source == "simulated" else "hist"
                row[f"mean_{tag}"] = round(d[source]["mean"])
                row[f"CVaR_{tag}"] = round(d[source]["cvar"])
                row[f"P(short)_{tag}"] = round(d[source]["p_shortfall"], 3)
        rows[model] = row
    print(pd.DataFrame(rows).T.to_string())


def make_figures(config, summary, plot_data):
    figures = RESULTS_DIR / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    names = config["crews"]["crew_names"]
    seasons = list(config["weather_data"]["seasons"])

    # 1. alpha of the most wind-sensitive crew (lowest threshold)
    crew = int(np.argmin(config["crews"]["wind_thresholds"]))
    alphas = {s: {"historical": plot_data[s]["alpha_hist"], "calibrated": plot_data[s]["alpha_10k"],
                  "course": plot_data["course"]["alpha_10k"]} for s in seasons}
    plot_alpha_distribution(alphas, crew, names[crew], config["project"]["num_days"],
                            figures / "alpha_distribution.png")

    # 2. two-stage allocation per case
    allocations = {case: summary[case]["decisions"]["two_stage"]["x"] for case in summary}
    plot_allocations(allocations, names, figures / "allocation_two_stage.png",
                     "Two-stage optimal allocation (emergency crew as recourse)")

    # 3. cost distribution of the final decision (with insurance option)
    costs = {s: plot_data[s]["costs"]["insurance"] for s in seasons}
    plot_cost_distributions(costs, config["risk"]["q"], config["risk"]["cost_threshold"],
                            figures / "cost_distribution.png",
                            "Total cost of the optimal decision with insurance option")
    print(f"Saved figures to {figures}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="download wind data again")
    args = parser.parse_args()

    config = load_config()
    wind_daily = load_wind_history(config, refresh=args.refresh)
    cases = build_cases(config, wind_daily)

    summary, plot_data = {}, {}
    for name, case in cases.items():
        result = run_case(config, case)
        print_case(config, name, result)
        plot_data[name] = {k: result.pop(k) for k in ["costs", "alpha_10k", "alpha_hist"]}
        if "calibration" in case:
            result["calibration"] = case["calibration"]
        summary[name] = result

    make_figures(config, summary, plot_data)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {RESULTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()