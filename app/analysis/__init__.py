"""Analysis sub-package: EDA and time-series forecasting."""

from .eda import (
    detect_data_types,
    generate_insights,
    detect_problem_type,
    recommend_models,
    create_data_quality_report,
    summary_cards,
)
from .timeseries import TimeSeriesSpec, make_univariate_ts, arima_forecast, pmdarima_status
