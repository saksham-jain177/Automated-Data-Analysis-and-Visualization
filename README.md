# 📊 Automated Data Analysis & Visualization

**Intelligent, configurable platform for data preprocessing, ML modeling, and forecasting**

### 🎯 Overview

Streamlit app with **agentic data preprocessing**, modular ML pipelines, and advanced configuration. No hardcoded defaults—all strategies are configurable via environment variables.

### 📚 Documentation

- 📖 **[USER_GUIDE.md](USER_GUIDE.md)** - Complete user guide
- 🏗️ **[VISION.md](VISION.md)** - Architecture and vision

### Key Features

#### 🤖 Agentic Data Preprocessing
- **Quality Assessment**: 0-100 score with detailed report
- **Intelligent Imputation**: median, mean, KNN, mode (configurable)
- **Outlier Detection**: IQR, Z-score, or none (configurable)
- **Type Correction**: Auto-detect and fix column types
- **No hardcoded defaults**: All strategies via `ADV_` env vars

#### 🔬 Machine Learning
- **Sklearn Pipelines**: ColumnTransformer with proper preprocessing
- **Cross-validation**: Stratified for classification, standard for regression
- **Feature Importance**: Permutation-based, model-agnostic
- **AutoML (FLAML)**: Time-budgeted model search
- **Advanced Models**: XGBoost, LightGBM, Random Forest, etc.

#### 📈 Analysis & Forecasting
- **Time Series**: ARIMA via pmdarima with auto period parsing
- **Multi-format**: CSV, Excel, JSON, Parquet
- **Smart Sampling**: Handle large datasets efficiently
- **Auto Dashboard**: Recommended charts with HTML export

#### 💬 AI Assistant
- **Chat with data**: OpenRouter API integration
- **Guided/Advanced modes**: Toggle complexity
- **Tutorial system**: Sample datasets and onboarding

### 🚀 Quick Start (30 Seconds)

**IMPORTANT: Always run from the virtual environment!**

```bash
# Windows:
run.bat

# Or manually:
.\.venv\Scripts\streamlit run auto.py

# Linux/Mac:
source .venv/bin/activate
streamlit run auto.py
```

**First time setup:**
1. Create venv: `python -m venv .venv`
2. Activate: `.\.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux/Mac)
3. Install: `pip install -r requirements.txt`
4. Run: `streamlit run auto.py` or use `run.bat` (Windows)

**That's it!** The app handles everything automatically.

### 📖 Detailed Installation

```bash
# 1. Clone the repository
git clone <repo-url>
   cd Automated-Data-Analysis-and-Visualization

# 2. Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
   pip install -r requirements.txt

# 4. Optional: Set up environment variables
cp .env.example .env
# Edit .env with your settings (API keys, etc.)

# 5. Run the app
streamlit run auto.py
```

**Browser will open automatically at** `http://localhost:8501`

### Auto Dashboard & NL Charts

- The app suggests a few charts automatically in Guided mode and allows you to download them as a single HTML report.
- Use the "Quick chart command" box to render charts with simple commands:
  - `hist <numeric_col>`
  - `scatter <x> vs <y>`
  - `bar avg <y> by <x>`

### Time Series Forecasting

- Select your time column (e.g., `Period`) and value column (e.g., `Data_value`).
- Click "Run forecast" to fit ARIMA and plot the next periods with confidence intervals.
- If `pmdarima` is missing, install with `pip install pmdarima`.

### Configuration (ENV VARS)

All behavior is configurable via environment variables with `ADV_` prefix:

**Data Preprocessing:**
- `ADV_IMPUTATION_METHOD` (median) - mean, median, knn, mode
- `ADV_OUTLIER_METHOD` (iqr) - iqr, zscore, none
- `ADV_OUTLIER_THRESHOLD` (1.5) - IQR multiplier or Z-score threshold
- `ADV_AGGRESSIVE_CLEANING` (false) - Always handle outliers

**Machine Learning:**
- `ADV_RANDOM_STATE` (42)
- `ADV_CV_FOLDS` (5)
- `ADV_AUTOML_ENABLED` (false)
- `ADV_AUTOML_TIME_BUDGET` (30)

**UI & Visualization:**
- `ADV_GUIDED_MODE_DEFAULT` (true)
- `ADV_MAX_PLOT_SAMPLES` (5000)
- `ADV_CORR_METHOD` (pearson)

**AI Assistant:**
- `ADV_OPENROUTER_API_KEY` (unset)
- `ADV_OPENROUTER_MODEL` (openrouter/auto)

### Project Structure

```
app/
  __init__.py
  config.py            # Pydantic settings (no hardcoding)
  preprocessing.py     # ColumnTransformer pipelines
  modeling.py          # CV, importance, AutoML hooks
  ui.py                # Streamlit UI (Guided + Advanced)
  chat.py              # OpenRouter chat helper (optional)
auto.py                # Entry point delegating to app.ui
```

### Roadmap / Ideas

- Regression tasks detection and metrics
- SHAP explanations for tree/linear models
- Model persistence and download
- Data quality checks and drift detection
 - Natural language chart generation via chat commands

### Contributing

PRs welcome. Please keep code modular, typed where helpful, and avoid hardcoding. Add concise comments explaining function purpose.

### License

MIT (see `LICENSE`).
