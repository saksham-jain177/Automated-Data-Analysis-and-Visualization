from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class AppSettings(BaseSettings):
    """Application settings loaded from environment variables.

    This class centralizes configurable parameters to avoid hardcoding and
    improve reproducibility. Environment variables use the ADV_ prefix.
    """

    random_state: int = Field(42, description="Default random seed for reproducibility")
    test_size: float = Field(0.2, description="Proportion for test split")
    cv_folds: int = Field(5, description="Number of cross-validation folds")
    automl_enabled: bool = Field(False, description="Enable FLAML AutoML by default")
    automl_time_budget: int = Field(30, description="AutoML time budget in seconds")
    max_plot_samples: int = Field(5000, description="Max rows sampled for plots")
    corr_method: str = Field("pearson", description="Correlation method for heatmaps")
    model_cache_dir: str = Field("models", description="Directory to save trained models")

    
    # Local LLM configuration (Ollama)
    llm_api_base: str = Field("http://localhost:11434/v1", description="Base URL for local LLM (Ollama)")
    llm_model: str = Field("llama3", description="Model name to use (e.g. llama3, mistral)")
    llm_api_key: str = Field("ollama", description="Dummy API key for local LLM compatibility")
    
    # Agentic preprocessing configuration
    imputation_method: str = Field("median", description="Method for missing data imputation: mean, median, knn, mode")
    outlier_method: str = Field("iqr", description="Method for outlier detection: iqr, zscore, none")
    outlier_threshold: float = Field(1.5, description="Threshold for outlier detection (IQR multiplier or Z-score)")
    aggressive_cleaning: bool = Field(False, description="Enable aggressive cleaning (always handle outliers)")

    model_config = SettingsConfigDict(
        env_prefix="ADV_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
    )


def get_settings() -> AppSettings:
    """Return settings loaded from environment variables."""

    return AppSettings()


