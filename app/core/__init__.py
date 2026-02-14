"""Core data processing: loading, quality assessment, preprocessing, optimization."""

from .data_loader import DataLoader, validate_dataframe
from .data_quality import (
    calculate_unified_quality_score,
    enhanced_outlier_detection,
    enhanced_detect_and_handle_outliers,
    check_type_consistency_correctly,
)
from .preprocessing import (
    UnifiedDataPreprocessor,
    CleaningConfig,
    FeatureConfig,
    detect_column_types,
    build_preprocessor,
    build_pipeline,
    display_unified_preprocessing_report,
)
from .optimizer import optimize_dtypes, smart_sampling, detect_and_handle_outliers, auto_feature_selection
