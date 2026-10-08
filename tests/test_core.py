"""Sanity checks. Run with: python -m pytest -q"""
import numpy as np
import pandas as pd

from prefabrisk.calibration import calibrate_season
from prefabrisk.config import load_config
from prefabrisk.models import solve_insurance, solve_two_stage, value_of_stochastic_solution
from prefabrisk.risk import cvar_q, evaluate_costs, var_q
from prefabrisk.scenarios import (completion_coefficients, generate_weather_scenarios,
                                  sample_weather_cycle)

config = load_config()
THRESHOLDS = config["crews"]["wind_thresholds"]


def course_alphas(n, seed):
    weather = generate_weather_scenarios(n, seed, 20, **config["course_simulator"])
    return completion_coefficients(weather, THRESHOLDS)


def test_simulator_matches_course_notebook():
    np.random.seed(42)
    first_days = np.round(sample_weather_cycle()[:3], 2)
    assert np.allclose(first_days, [40.62, 31.8, 34.53])


def test_completion_coefficients_are_shares_of_days():
    weather = np.array([[10.0, 35.0, 50.0, 20.0]])
    alpha = completion_coefficients(weather, [43.3, 31.7])
    assert np.allclose(alpha, [[0.75, 0.5]])


def test_calibration_recovers_rho():
    rng = np.random.default_rng(0)
    days = pd.date_range("2000-01-01", "2019-12-31")
    z = np.zeros(len(days))
    for t in range(1, len(days)):
        z[t] = 0.6 * z[t - 1] + rng.normal(0, np.sqrt(1 - 0.6**2))
    wind = pd.DataFrame({"v": 10 + np.exp(0.4 * z) * 15}, index=days)
    p = calibrate_season(wind, "v", [10, 11])
    assert abs(p["rho"] - 0.6) < 0.05


def test_replayed_costs_equal_optimised_costs():
    alpha_200 = course_alphas(200, 45)
    ins = solve_insurance(config, alpha_200)
    replay = evaluate_costs(config, ins["x"], alpha_200, ins["insurance"])
    assert np.isclose(replay.mean(), ins["objective"], rtol=1e-6)


def test_vss_non_negative_and_fixed_x_reproduces_optimum():
    alpha_200 = course_alphas(200, 45)
    two = solve_two_stage(config, alpha_200)
    fixed = solve_two_stage(config, alpha_200, fixed_x=two["x"])
    assert np.isclose(fixed["objective"], two["objective"], rtol=1e-6)
    assert value_of_stochastic_solution(config, alpha_200)["vss"] >= -1e-6


def test_var_and_cvar_use_upper_tail():
    costs = np.arange(1, 101, dtype=float)
    assert var_q(costs, 5) > np.median(costs)
    assert cvar_q(costs, 5) >= var_q(costs, 5)