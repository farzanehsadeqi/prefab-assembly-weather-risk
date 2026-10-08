"""Fetch daily maximum wind speeds for the site from Open-Meteo.

Open-Meteo needs no API key. The archive endpoint serves ERA5 reanalysis
from 1940 up to a few days ago. Wind speeds are in km/h, days are local time.

Three candidate variables are stored so they can be compared:
  - wind_speed_10m_max  : daily max sustained wind at 10 m (from the API)
  - wind_gusts_10m_max  : daily max gust at 10 m (from the API)
  - wind_speed_100m_max : daily max of hourly wind at 100 m, closer to the
                          height where a tower crane boom works (computed here,
                          because the API offers 100 m wind only hourly)
The one used in the model is set in config/params.yaml.
"""
import pandas as pd
import requests

from prefabrisk.config import PROJECT_ROOT

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARIABLES = ["wind_speed_10m_max", "wind_gusts_10m_max"]
HOURLY_VARIABLE = "wind_speed_100m"
CACHE_FILE = PROJECT_ROOT / "data" / "wind_daily.csv"


def _base_params(site, start_date, end_date):
    return {
        "latitude": site["latitude"],
        "longitude": site["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "wind_speed_unit": "kmh",
        "timezone": site["timezone"],
    }


def fetch_wind_history(site, start_date, end_date):
    params = _base_params(site, start_date, end_date)
    params["daily"] = ",".join(DAILY_VARIABLES)
    r = requests.get(ARCHIVE_URL, params=params, timeout=120)
    r.raise_for_status()
    wind_daily = pd.DataFrame(r.json()["daily"])
    wind_daily["time"] = pd.to_datetime(wind_daily["time"])
    return wind_daily.set_index("time")


def fetch_wind100_history(site, start_date, end_date):
    """Daily maximum of hourly 100 m wind, downloaded one year at a time."""
    frames = []
    first_year = pd.Timestamp(start_date).year
    last_year = pd.Timestamp(end_date).year
    for year in range(first_year, last_year + 1):
        start = max(start_date, f"{year}-01-01")
        end = min(end_date, f"{year}-12-31")
        params = _base_params(site, start, end)
        params["hourly"] = HOURLY_VARIABLE
        r = requests.get(ARCHIVE_URL, params=params, timeout=120)
        r.raise_for_status()
        frames.append(pd.DataFrame(r.json()["hourly"]))
        print(f"  100 m wind: {year} done")
    hourly = pd.concat(frames)
    hourly["time"] = pd.to_datetime(hourly["time"])
    daily_max = hourly.set_index("time")[HOURLY_VARIABLE].resample("D").max()
    return daily_max.rename("wind_speed_100m_max")


def load_wind_history(config, refresh=False):
    """Read the cached CSV, or download it if missing or refresh=True."""
    if CACHE_FILE.exists() and not refresh:
        return pd.read_csv(CACHE_FILE, index_col="time", parse_dates=True)
    site, wd = config["site"], config["weather_data"]
    wind_daily = fetch_wind_history(site, wd["start_date"], wd["end_date"])
    wind_daily = wind_daily.join(fetch_wind100_history(site, wd["start_date"], wd["end_date"]))
    wind_daily = wind_daily.dropna()
    CACHE_FILE.parent.mkdir(exist_ok=True)
    wind_daily.to_csv(CACHE_FILE)
    return wind_daily