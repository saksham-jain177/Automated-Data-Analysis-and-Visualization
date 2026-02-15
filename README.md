# 📊 Automated Data Analysis & Visualization

**Intelligent, modular platform for data preprocessing, ML modeling, and AI-powered insights.**

### 🎯 Overview

A modern Streamlit application designed for automated data science. It features **agentic data cleaning**, **modular ML pipelines**, **RAG-powered chat**, and a **sidebar navigation** workflow. Built for performance with caching and privacy controls.

### ✨ Key Features

#### 🛡️ modular & Optimized Core

- **Agentic Preprocessing**: Configurable imputation, outlier detection, and type correction.
- **Cached Pipeline**: Heavy processing runs once and is cached for instant page nav.
- **Privacy-First**: Data stays local. External LLM calls (OpenRouter) require explicit opt-in.

#### 🤖 RAG-Powered AI Chat

- **Context-Aware**: Uses TF-IDF retrieval to find relevant data chunks for the LLM.
- **Grounded Answers**: The AI answers based on _your_ data, not just general knowledge.
- **Transparency**: View the exact data chunks retrieved for each answer.

#### 🔬 Machine Learning & AutoML

- **Automated Modeling**: Auto-selects best models (XGBoost, LightGBM, Random Forest).
- **Time Series**: ARIMA forecasting with auto-period detection.
- **Explainability**: Permutation feature importance and model evaluation metrics.

#### 📊 Advanced Visualization

- **Natural Language Charts**: "scatter price vs age", "histogram of salary"
- **Smart Sampling**: Handles large datasets efficiently.
- **Interactive UI**: Plotly charts with zoom/pan.

### 🚀 Quick Start

**Prerequisites**: Python 3.9+

```bash
# 1. Clone & Setup
git clone https://github.com/saksham-jain177/Automated-Data-Analysis-and-Visualization
cd Automated-Data-Analysis-and-Visualization
python -m venv .venv

# 2. Activate
# Windows:
.\.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# 3. Install
pip install -r requirements.txt

# 4. Run
streamlit run auto.py
```

### 📂 Project Structure

The project is refactored into domain-specific packages for maintainability:

```text
app/
├── core/           # Data loading, quality, caching, optimization
├── analysis/       # EDA, insights generation, time-series logic
├── ml/             # Machine learning models, AutoML, evaluation
├── viz/            # Chart generation, NL parsing
├── chat/           # RAG retrieval (TF-IDF), LLM client
└── ui/             # Streamlit interface
    ├── sections/   # Modular UI pages (Data Setup, Explore, ML, Report)
    └── app.py      # Main UI orchestrator
```

### ⚙️ Configuration

Configure the app via `.env` file or environment variables. All settings have `ADV_` prefix.

**Local AI (Ollama):**

- `ADV_LLM_API_BASE`: Base URL (default: `http://localhost:11434/v1`)
- `ADV_LLM_MODEL`: Model name (default: `llama3`)
- `ADV_LLM_API_KEY`: Dummy key (default: `ollama`)

### 🤖 Local AI Setup (Ollama)

1. **Install Ollama**: Download from [ollama.com](https://ollama.com).
2. **Pull a Model**: Run `ollama pull llama3` (or any other model supported by Ollama).
3. **Run Ollama**: Keep `ollama serve` running in the background.
4. **Configure App**: The app defaults to `http://localhost:11434/v1` and `llama3`. If you use a different model or port, set `ADV_LLM_MODEL` or `ADV_LLM_API_BASE` in `.env`.

**Data Processing Defaults:**

- `ADV_IMPUTATION_METHOD`: median, mean, knn, mode
- `ADV_OUTLIER_METHOD`: iqr, zscore, none
- `ADV_CV_FOLDS`: 5

### 🤝 Contributing

Contributions are welcome! Please ensure you follow the modular structure.

- **UI changes** go in `app/ui/`
- **Logic changes** go in `app/core/`, `app/ml/`, etc.

### 📄 License

MIT
