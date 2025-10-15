from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TimeSeriesSpec:
    time_col: str
    value_col: str
    freq: Optional[str] = None  # e.g., 'M', 'Q', 'D'


def parse_period_column(series: pd.Series) -> pd.Series:
    """Parse numeric/string period like 2011.06 -> timestamp at period end of month.

    Falls back to pandas to_datetime for general strings.
    """

    s = series.copy()
    # Handle numeric like 2011.06 (year.month)
    def _to_ts(x):
        try:
            if isinstance(x, (int, float)) and not pd.isna(x):
                year = int(x)
                month = int(round((x - year) * 100))
                month = max(1, min(12, month))
                return pd.Timestamp(year=year, month=month, day=1)
            return pd.to_datetime(x)
        except Exception:
            return pd.NaT

    return s.map(_to_ts)


def make_univariate_ts(df: pd.DataFrame, spec: TimeSeriesSpec) -> pd.Series:
    time = parse_period_column(df[spec.time_col])
    y = df[spec.value_col]
    ts = pd.Series(y.values, index=time).sort_index()
    ts = ts[~ts.index.duplicated(keep="last")]
    ts = ts.asfreq(spec.freq or pd.infer_freq(ts.index), method=None)
    return ts


def arima_forecast(ts: pd.Series, horizon: int = 12) -> Tuple[pd.Series, pd.DataFrame]:
    """Fit ARIMA via pmdarima auto_arima and forecast horizon steps.

    Returns (forecast_series, confidence_intervals_df).
    """

    try:
        import pmdarima as pm
    except ImportError as exc:
        raise ImportError("pmdarima not installed. Please add pmdarima to requirements.") from exc
    except Exception as exc:
        raise RuntimeError(f"pmdarima import failed: {exc}") from exc

    model = pm.auto_arima(
        ts,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )
    fc, conf = model.predict(n_periods=horizon, return_conf_int=True)
    index = pd.date_range(ts.index[-1] + (ts.index[-1] - ts.index[-2]), periods=horizon, freq=ts.index.freq)
    forecast = pd.Series(fc, index=index)
    conf_df = pd.DataFrame(conf, columns=["lower", "upper"], index=index)
    return forecast, conf_df


def pmdarima_status() -> Tuple[bool, Optional[str]]:
    """Return (available, version)."""

    try:
        import pmdarima  # type: ignore
        return True, getattr(pmdarima, "__version__", None)
    except Exception:
        return False, None


