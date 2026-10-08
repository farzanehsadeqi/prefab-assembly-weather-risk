"""Wind scenarios and crew completion coefficients.

sample_weather_cycle is the course simulator, unchanged except for one
option: with wind_values given, the last step uses the empirical quantile
function of observed winds instead of the Weibull formula.
"""
import numpy as np
import scipy.stats as stats


def sample_weather_cycle(num_days=20, rho=0.65, shape=2.0, scale=25.0, wind_values=None):
    """
    Simulates a 20-day sequence of daily maximum wind speeds using an AR(1) process
    mapped into a Weibull (course) or empirical (calibrated) distribution.
    """
    z = np.zeros(num_days)
    innovations = np.random.normal(0, np.sqrt(1 - rho**2), num_days)
    z[0] = np.random.normal(0, 1)
    for t in range(1, num_days):
        z[t] = rho * z[t-1] + innovations[t]
    u = stats.norm.cdf(z)
    u = np.clip(u, 1e-6, 1 - 1e-6)
    if wind_values is None:
        return (-np.log(1 - u))**(1 / shape) * scale
    return np.quantile(wind_values, u)


def generate_weather_scenarios(n_scenarios, seed, num_days=20, **simulator_params):
    np.random.seed(seed)
    return np.array([sample_weather_cycle(num_days, **simulator_params) for _ in range(n_scenarios)])


def completion_coefficients(weather_scenarios, wind_thresholds):
    """alpha[s, i] = share of days in scenario s on which crew i can work."""
    n_scenarios, num_days = weather_scenarios.shape
    alpha = np.zeros((n_scenarios, len(wind_thresholds)))
    for s in range(n_scenarios):
        wind_speed = weather_scenarios[s]
        operational_days = np.array([np.sum(wind_speed <= t) for t in wind_thresholds])
        alpha[s] = operational_days / num_days
    return alpha


def historical_windows(wind_daily, variable, months, num_days=20):
    """All runs of num_days consecutive days inside the season, year by year."""
    windows = []
    season = wind_daily.loc[wind_daily.index.month.isin(months), variable]
    for year, wind_speed in season.groupby(season.index.year):
        values = wind_speed.to_numpy()
        for start in range(len(values) - num_days + 1):
            windows.append(values[start:start + num_days])
    return np.array(windows)