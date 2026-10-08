"""Calibrate the course wind simulator on real data.

The course simulator is a Gaussian AR(1) series z mapped to wind speed
through a Weibull quantile function. On Bielefeld data a Weibull marginal
(2 or 3 parameters) misses the share of calm days at the crew thresholds
by 3-6 percentage points, so the calibrated simulator keeps the AR(1)
dependence but uses the empirical distribution of the season's daily
maxima as marginal (Gaussian copula with empirical marginal):
  1. rank each day within the season and map it to z = Phi^-1(rank / (n + 1))
  2. rho = lag-1 correlation of z between consecutive calendar days
  3. the sorted observed wind values serve as the quantile function
A two-parameter Weibull fit is still reported, for comparison only.
"""
import numpy as np
import pandas as pd
import scipy.stats as stats


def season_series(wind_daily, variable, months):
    return wind_daily.loc[wind_daily.index.month.isin(months), variable]


def fit_weibull(wind_speed):
    shape, _, scale = stats.weibull_min.fit(wind_speed, floc=0)
    return shape, scale


def to_normal_scores(wind_speed):
    u = stats.rankdata(wind_speed) / (len(wind_speed) + 1)
    return pd.Series(stats.norm.ppf(u), index=wind_speed.index)


def fit_rho(z):
    """Lag-1 correlation, using only pairs of consecutive calendar days.

    This skips the jump from the last day of a season to the first day
    of the same season one year later.
    """
    days = z.index.to_series()
    consecutive = (days.shift(-1) - days) == pd.Timedelta(days=1)
    z_today = z[consecutive]
    z_tomorrow = z.shift(-1)[consecutive]
    return np.corrcoef(z_today, z_tomorrow)[0, 1]


def calibrate_season(wind_daily, variable, months):
    wind_speed = season_series(wind_daily, variable, months)
    rho = fit_rho(to_normal_scores(wind_speed))
    shape, scale = fit_weibull(wind_speed)
    return {
        "rho": float(rho),
        "wind_values": np.sort(wind_speed.to_numpy()),   # empirical marginal
        "weibull_shape": float(shape),                    # reported only
        "weibull_scale": float(scale),
        "n_days": int(len(wind_speed)),
    }