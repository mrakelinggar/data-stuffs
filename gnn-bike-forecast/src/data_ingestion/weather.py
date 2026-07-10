"""
NOAA LCD Weather Data Ingestion

Downloads/caches hourly Local Climatological Data (LCD) for a single NOAA
station-year and returns a cleaned, gap-flagged hourly DataFrame.
"""

from pathlib import Path

import pandas as pd
import requests
from loguru import logger

LCD_BASE_URL = "https://www.ncei.noaa.gov/data/local-climatological-data/access"

# Combined USAF+WBAN id used by NOAA's LCD file naming -- NOT the GHCN-style
# "USW00094728" id sometimes used to refer to this same station elsewhere.
CENTRAL_PARK_STATION_ID = "72505394728"

_RAW_COLUMN_MAP = {
    "HourlyDryBulbTemperature": "temp_f",
    "HourlyPrecipitation": "precip_in",
    "HourlyWindSpeed": "wind_mph",
    "HourlyRelativeHumidity": "humidity_pct",
}


def _download_lcd_csv(station_id: str, year: int, dest: Path) -> Path:
    """Stream-download the raw LCD station-year CSV to dest if not already cached."""
    if dest.exists():
        logger.info(f"LCD cache hit: {dest}")
        return dest

    url = f"{LCD_BASE_URL}/{year}/{station_id}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading LCD data from {url}")
    try:
        with requests.get(url, stream=True, timeout=(10, 120)) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception:
        logger.error(f"Failed to download {url}; cleaning up partial file {dest}")
        dest.unlink(missing_ok=True)
        raise

    logger.info(f"Saved LCD data to {dest}")
    return dest


def _flag_long_gaps(is_na: pd.Series, max_ffill_hours: int) -> pd.Series:
    """
    Flag every row belonging to a run of >= max_ffill_hours consecutive
    missing values (the whole run, not just the rows past the threshold).
    Rows inside a shorter run (forward-filled but not flagged) are False,
    as are all non-missing rows.
    """
    run_id = (~is_na).cumsum()
    run_totals = is_na.groupby(run_id).transform("sum")
    return is_na & (run_totals >= max_ffill_hours)


def _clean_numeric(series: pd.Series) -> pd.Series:
    """
    LCD numeric columns carry trailing quality-flag characters (e.g. "32s")
    and use "T" for trace precipitation. Map trace -> 0.0 *before* stripping
    flag characters (a bare "T" is itself a trailing-letter pattern and would
    otherwise be stripped to an empty string), then strip any letters
    trailing a digit, and coerce the rest to numeric (unparseable values
    become NaN, handled by the gap-fill logic downstream).
    """
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.replace({"T": "0.0"})
    cleaned = cleaned.str.replace(r"(?<=\d)[A-Za-z]+$", "", regex=True)
    cleaned = cleaned.replace({"": None})
    return pd.to_numeric(cleaned, errors="coerce")


def fetch_noaa_lcd(
    station_id: str = CENTRAL_PARK_STATION_ID,
    year: int = 2024,
    data_dir: Path = Path("data/raw/weather"),
    raw_filename: str = "lcd_central_park_2024.csv",
    max_ffill_hours: int = 3,
) -> pd.DataFrame:
    """
    Fetch (or load from cache) NOAA LCD hourly data for one station-year.

    Returns
    -------
    DataFrame indexed by hourly timestamp (tz-naive) with columns:
        temp_f, precip_in, wind_mph, humidity_pct, weather_imputed (bool)
    Gaps are forward-filled so the returned frame has zero NaNs. Rows
    sitting inside a run of >= max_ffill_hours consecutive missing hours are
    flagged weather_imputed=True; shorter gaps are filled unflagged.
    """
    data_dir = Path(data_dir)
    raw_path = data_dir / raw_filename
    _download_lcd_csv(station_id, year, raw_path)

    raw = pd.read_csv(raw_path, low_memory=False)

    # LCD interleaves hourly (FM-15), daily (SOD), and monthly (SOM) summary
    # rows in one file -- keep hourly observations only, or the reindex step
    # below would see duplicate timestamps / non-hourly pollution.
    raw = raw[raw["REPORT_TYPE"].astype(str).str.strip() == "FM-15"].copy()

    raw["DATE"] = pd.to_datetime(raw["DATE"])
    raw = raw.rename(columns=_RAW_COLUMN_MAP)

    value_cols = list(_RAW_COLUMN_MAP.values())
    df = raw[["DATE", *value_cols]].copy()
    for col in value_cols:
        df[col] = _clean_numeric(df[col])

    # Collapse any remaining duplicate/sub-hourly timestamps within the same
    # clock hour, then reindex to a complete hourly grid.
    df = df.set_index("DATE").resample("h").mean()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_idx)
    df.index.name = "timestamp"

    is_na = df.isna().any(axis=1)
    df["weather_imputed"] = _flag_long_gaps(is_na, max_ffill_hours)
    df[value_cols] = df[value_cols].ffill()

    return df


def main() -> None:
    df = fetch_noaa_lcd()
    logger.info(f"Fetched {len(df)} hourly rows")
    logger.info(f"Imputed rows: {int(df['weather_imputed'].sum())}")
    print(df.describe())


if __name__ == "__main__":
    main()
