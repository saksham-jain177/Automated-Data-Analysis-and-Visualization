# 📚 User Guide: Automated Data Analysis & Visualization

> **For Non-Technical Users** - Everything you need to know to analyze your data like a pro!

---

## 🎯 Quick Start (30 seconds)

1. **Upload your file** (CSV, Excel, JSON, Parquet)
2. **Click "Quick Analyze"** (in Guided Mode)
3. **View insights, charts, and predictions automatically**

That's it! The app does the heavy lifting for you.

---

## 📊 What Can This App Do?

### ✅ Automatic Data Analysis
- **Cleans your data** (removes duplicates, fixes errors, fills missing values)
- **Generates insights** (finds patterns, trends, anomalies)
- **Creates visualizations** (beautiful charts and graphs)
- **Scores data quality** (tells you if your data is good to use)

### 🤖 Machine Learning
- **Predicts future values** (sales, prices, trends)
- **Classifies data** (spam/not spam, high/low risk)
- **Finds important features** (what matters most in your data)
- **Auto-selects best models** (no technical knowledge needed)

### 📈 Time Series Forecasting
- **Predicts future trends** based on historical data
- **Works with dates/time periods** (daily, monthly, yearly)
- **Shows confidence intervals** (how certain the predictions are)

### 💬 Chat Assistant
- **Ask questions about your data** in plain English
- **Get code suggestions** for custom analysis
- **Powered by LLM service Provider** OpenRouter

---

## 🧩 Understanding the Interface

### 🎨 Guided Mode (Recommended for Beginners)
**Turn it ON** in the sidebar → Simple, clean interface with smart defaults

**What you get:**
- Automatic recommendations
- Fewer confusing options
- Step-by-step workflow
- Best practices applied automatically

### 🔧 Advanced Mode
**Turn Guided Mode OFF** → Full control over every setting

**What you get:**
- All preprocessing options
- Manual model selection
- Custom parameter tuning
- Advanced visualizations

---

## 📥 Supported File Types

| Format | Extension | Notes |
|--------|-----------|-------|
| CSV | `.csv` | Most common, comma-separated |
| TSV | `.tsv` | Tab-separated values |
| Excel | `.xlsx`, `.xls` | Microsoft Excel files |
| JSON | `.json` | JavaScript Object Notation |
| JSON Lines | `.jsonl` | One JSON object per line |
| Parquet | `.parquet` | High-performance columnar storage |

**Max file size:** 200MB

---

## 🔍 Section-by-Section Guide

### 1️⃣ **Data Upload & Preview**
**What it does:** Loads your file and shows the first few rows

**What to look for:**
- ✅ Correct number of rows and columns
- ✅ Column names make sense
- ✅ Data values look correct

**Troubleshooting:**
- If columns are wrong, check your file format
- If data looks jumbled, try a different file type

---

### 2️⃣ **Agentic Data Preprocessing** (Intelligent Cleaning)
**What it does:** Intelligently cleans and corrects your dataset using configurable strategies

**Preprocessing Pipeline:**
1. **Quality Assessment** - Generates quality score (0-100) and detailed report
2. **Duplicate Removal** - Removes exact duplicate rows
3. **Type Correction** - Auto-detects and fixes column types (dates, numbers, categories)
4. **Missing Data Imputation** - Fills missing values using your chosen strategy
5. **Outlier Handling** - Detects and caps extreme values (preserves data)
6. **Final Quality Check** - Shows improvement in data quality

**Imputation Methods:**
- **Median** (default): Robust to outliers, good for numeric data
- **Mean**: Average value, sensitive to outliers
- **KNN**: Uses similar rows to predict missing values (most accurate but slower)
- **Mode**: Most frequent value, used for categorical data

**Outlier Detection:**
- **IQR** (Interquartile Range): Standard statistical method, robust
- **Z-score**: Based on standard deviations, good for normal distributions
- **None**: Skip outlier handling

**When to use:**
- ✅ Your data has missing values
- ✅ You see duplicates or inconsistencies
- ✅ Data types are incorrect (dates stored as text)
- ✅ You want automated, intelligent cleaning

**Configuration:**
- **Guided Mode**: Uses optimal defaults (median + IQR)
- **Advanced Mode**: Choose your imputation and outlier methods

**No more hardcoded "smart defaults"** - You control the strategy!

---

### 3️⃣ **Exploratory Data Analysis (EDA)**
**What it does:** Generates insights and visualizations automatically

**What you get:**
- **Summary cards** (quick stats: rows, columns, missing data)
- **Data quality score** (0-100, higher = better)
- **Automated insights** (patterns, correlations, anomalies)
- **Visualizations** (histograms, correlation heatmaps)

**Key metrics explained:**
- **Completeness:** % of data filled in (not missing)
- **Uniqueness:** Variety of values (not all the same)
- **Consistency:** Data types and formats are correct

---

### 4️⃣ **Machine Learning Modeling**
**What it does:** Trains a model to make predictions

#### Step 1: Choose Target Column
- **Target = What you want to predict**
- Examples: "Price", "Category", "Risk Level"

#### Step 2: Problem Type (Detected Automatically)
- **Classification:** Predicting categories (Yes/No, High/Medium/Low)
- **Regression:** Predicting numbers (prices, ages, quantities)

#### Step 3: Model Selection
- **Guided Mode:** Auto-selected for you ✅
- **Advanced Mode:** Choose from list

**Popular models:**
- 🌟 **Random Forest:** Great all-around, handles messy data
- ⚡ **XGBoost:** Very accurate, industry standard
- 🚀 **LightGBM:** Fast and efficient
- 📊 **Logistic Regression:** Simple, interpretable

#### Step 4: Evaluate & Train
- **📊 Evaluate Model Accuracy:** Tests how well the model works
- **🔍 Show Feature Importance:** Reveals what matters most

**Understanding accuracy scores:**

**For Classification:**
- **Accuracy:** 0.85 = 85% correct predictions
- **F1 Score:** Balance metric (0-1, higher = better)
- Target: > 0.80 is good, > 0.90 is excellent

**For Regression:**
- **R² Score:** 0.85 = explains 85% of variance (closer to 1.0 = better)
- **MAE:** Average error (lower = better)
- Target: Depends on your data scale

---

### 5️⃣ **AutoML** (Let AI Find the Best Model)
**What it does:** Tests dozens of models automatically and picks the best one

**When to use:**
- ✅ You want maximum accuracy
- ✅ You're not sure which model to use
- ✅ You have 30+ seconds to wait

**How it works:**
1. Select time budget (30s-600s)
2. Click "Start AutoML Search"
3. Wait for results
4. **Use the model** via 4 tabs that appear

**After AutoML completes, you get 4 tabs:**

#### Tab 1: 📊 **Evaluate**
- Test the AutoML model's accuracy
- Get cross-validation scores
- See how reliable it is

#### Tab 2: 🔍 **Feature Importance**
- See which columns matter most
- Understand what drives predictions
- Top 15 most important features

#### Tab 3: 💾 **Save Model**
- Download the model as `.pkl` file
- Use it in your own Python scripts
- Code example provided

#### Tab 4: 🔮 **Make Predictions**
- Upload new CSV data
- Get predictions instantly
- Download results as CSV

**Time budget guide:**
- 30 seconds: Quick search
- 60 seconds: Good balance (recommended)
- 120+ seconds: Thorough search

---

### 6️⃣ **Time Series Forecasting**
**What it does:** Predicts future values based on historical trends

**Requirements:**
- ✅ A **time column** (dates, periods, timestamps)
- ✅ A **numeric column** to forecast (sales, temperature, etc.)

**⚠️ When Time Series CANNOT Be Used:**
- **Cross-sectional data** - snapshot at one point in time (like Wine quality dataset)
- **No temporal dimension** - data doesn't change over time
- **Missing time column** - no dates, timestamps, or time periods

**What to use instead:**
- Use **Machine Learning Modeling** for non-temporal predictions
- ML can predict outcomes without needing time (e.g., predict wine quality from chemical properties)

**Use cases:**
- Sales forecasting
- Stock price prediction
- Weather patterns
- Website traffic
- Inventory planning

**Steps:**
1. Select **time column** (when data was recorded)
2. Select **value column** (what to predict)
3. Choose **forecast horizon** (how many periods ahead)
4. Click "🚀 Run Forecast"

**Understanding the chart:**
- **Blue line:** Historical data (what actually happened)
- **Red dashed line:** Forecast (predictions)
- **Pink shaded area:** 95% confidence interval (likely range)

**What if I get an error?**
- ❌ "Cannot cast to unit='ns'" → Your time column isn't in date format
- **Fix:** Use a different column or clean your dates

---

### 7️⃣ **Chat Assistant** (Ask Questions About Your Data)
**What it does:** AI-powered assistant that answers questions

**Requirements:**
- OpenRouter API key (get one at openrouter.ai)
- Set in `.env` file: `ADV_OPENROUTER_API_KEY=your_key_here`

**Example questions:**
- "What are the top 5 most important columns?"
- "Show me a chart of sales over time"
- "What patterns do you see in this data?"
- "How can I predict customer churn?"

**Tips:**
- Be specific in your questions
- Ask one thing at a time
- Verify AI responses against actual data (use "Verify claims" expander)

---

## 🎓 Tutorial Mode

**New to data analysis?** Click **"📚 Show Tutorial"** in the sidebar

**What you get:**
- Step-by-step walkthrough
- Sample datasets (Iris, Wine, Breast Cancer)
- Hands-on practice
- Learn by doing!

---

## ⚙️ Settings & Configuration

### Environment Variables (`.env` file)
```env
# Guided mode (simplified UI)
ADV_GUIDED_MODE_DEFAULT=true

# Chat assistant
ADV_OPENROUTER_API_KEY=sk-or-v1-xxx
ADV_OPENROUTER_MODEL=openrouter/auto

# ML settings
ADV_CV_FOLDS=5
ADV_AUTOML_TIME_BUDGET=60
ADV_MAX_PLOT_SAMPLES=5000
```

### Common Settings
- **CV Folds:** How many times to test the model (5 is standard)
- **AutoML Time Budget:** Search time in seconds
- **Max Plot Samples:** Downsample large datasets for faster charts

---

## 🚨 Troubleshooting

### ❌ "pmdarima not installed"
**Problem:** Time series forecasting needs pmdarima library

**Fix:**
```bash
pip install pmdarima
```
Then restart Streamlit.

---

### ❌ "FLAML not installed"
**Problem:** AutoML needs FLAML library

**Fix:**
```bash
pip install 'flaml[automl]'
```
Then restart Streamlit.

---

### ❌ "OpenRouter API error"
**Problem:** Chat assistant can't connect

**Fix:**
1. Check your API key in `.env`
2. Verify you have credits at openrouter.ai
3. Try a different model (some are free)

---

### ❌ "Cannot cast datetime"
**Problem:** Time series can't parse your time column

**Fix:**
- Try a different time column
- Ensure dates are in standard format (YYYY-MM-DD)
- Use numeric periods (2024.01 = Jan 2024)

---

### ❌ Charts not showing
**Problem:** Visualization section is empty

**Fix:**
- Ensure you have numeric columns
- Try reducing dataset size (use cleaning → sampling)
- Check browser console for errors

---

## 💡 Best Practices

### ✅ Data Quality
1. **Clean your data first** (enable automated cleaning)
2. **Check for missing values** (look at summary cards)
3. **Remove duplicates** (cleaning handles this)
4. **Validate data types** (dates as dates, numbers as numbers)

### ✅ Modeling
1. **Start with Random Forest or XGBoost** (reliable choices)
2. **Always evaluate before trusting predictions** (cross-validation)
3. **Check feature importance** (understand what drives predictions)
4. **Use AutoML for best results** (when time allows)

### ✅ Forecasting
1. **Need consistent time intervals** (monthly, daily, etc.)
2. **More history = better forecast** (at least 24 data points)
3. **Verify forecasts make sense** (sanity check predictions)
4. **Use confidence intervals** (understand uncertainty)

---

## 🎯 Common Use Cases

### 📊 Business Analytics
1. Upload sales data
2. Quick Analyze → View trends
3. Forecast future sales (Time Series)
4. Identify key drivers (Feature Importance)

### 🏥 Healthcare Analysis
1. Upload patient data
2. Clean & validate (automated cleaning)
3. Predict outcomes (ML Modeling)
4. Understand risk factors (Feature Importance)

### 💰 Financial Analysis
1. Upload transaction data
2. Detect anomalies (EDA insights)
3. Predict fraud (Classification)
4. Forecast revenue (Time Series)

### 🛒 Customer Analytics
1. Upload customer behavior data
2. Segment customers (Clustering via chat)
3. Predict churn (Classification)
4. Identify high-value features

---

## 🔗 Additional Resources

- **GitHub:** [Project Repository](#)
- **Issues:** Report bugs or request features
- **Documentation:** README.md
- **API Docs:** OpenRouter.ai

---

## 📞 Getting Help

**If you're stuck:**
1. Check this guide first
2. Read error messages carefully (they often tell you what's wrong)
3. Use the chat assistant (ask: "What does this error mean?")
4. Check troubleshooting section above
5. Review tutorial mode for examples

**Still need help?**
- Open an issue on GitHub
- Include error messages and screenshots
- Describe what you tried

---

## 🎉 You're Ready!

This app makes data analysis accessible to everyone. Don't worry about making mistakes—experiment, explore, and learn!

**Remember:**
- Start with Guided Mode
- Use tutorial datasets to practice
- Ask the chat assistant when confused
- Clean your data for best results

**Happy analyzing!** 📊✨


