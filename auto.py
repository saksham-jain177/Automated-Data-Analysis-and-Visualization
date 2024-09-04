import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title("Automated Data Analysis and Visualization")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    st.subheader("Basic Data Information")
    st.write("Shape of the dataset:")
    st.write(df.shape)
    st.write("Columns in the dataset:")
    st.write(df.columns.tolist())
    st.write("Data types of each column:")
    st.write(df.dtypes)
    st.write("Missing values in the dataset:")
    st.write(df.isnull().sum())
    st.write("Summary Statistics:")
    st.write(df.describe())

    # Define numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Handling Missing Values
    st.subheader("Handling Missing Values")
    missing_value_strategy = st.selectbox("Select strategy to handle missing values",
                                          ["None", "Drop rows", "Fill with mean", "Fill with median", "Fill with mode"])

    if missing_value_strategy == "Drop rows":
        df = df.dropna()
    elif missing_value_strategy == "Fill with mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif missing_value_strategy == "Fill with median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif missing_value_strategy == "Fill with mode":
        df = df.fillna(df.mode().iloc[0])

    # Data Normalization and Standardization
    st.subheader("Data Normalization and Standardization")
    normalization_strategy = st.selectbox("Select normalization/standardization strategy",
                                          ["None", "Standardization (Z-score)", "Normalization (Min-Max)"])

    if normalization_strategy == "Standardization (Z-score)":
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    elif normalization_strategy == "Normalization (Min-Max)":
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Feature Engineering
    st.subheader("Feature Engineering")
    if st.checkbox("Add polynomial features"):
        st.session_state['Add polynomial features'] = True
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(df[numeric_cols])
        df_poly = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(numeric_cols))
        df = pd.concat([df.reset_index(drop=True), df_poly], axis=1)
    else:
        st.session_state['Add polynomial features'] = False

    st.subheader("Data Visualization")
    st.write("Histogram of Numeric Columns")
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    st.write("Correlation Heatmap")
    if len(numeric_cols) > 1:  # Ensure there are at least two numeric columns for correlation
        fig, ax = plt.subplots()
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.write("Not enough numeric columns for correlation heatmap.")

    st.write("Scatter Plot")
    if len(numeric_cols) >= 2:
        if 'Add polynomial features' in st.session_state and st.session_state['Add polynomial features']:
            df_temp = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(numeric_cols))
            for i in range(0, len(df_temp.columns), 2):
                x_col = df_temp.columns[i]
                y_col = df_temp.columns[i+1] if i+1 < len(df_temp.columns) else df_temp.columns[0]
                
                fig = px.scatter(df_temp, x=x_col, y=y_col, title=f"Scatter plot of {x_col} vs {y_col}")
                st.plotly_chart(fig)
        else:
            df_temp = df

            if not df_temp.empty:
                fig = px.scatter(df_temp, x=numeric_cols[0], y=numeric_cols[1])
                st.plotly_chart(fig)
            else:
                st.write("DataFrame is empty after adding polynomial features.")

    st.subheader("Feature Importance")
    if st.checkbox("Calculate feature importance"):
        target_col = st.selectbox("Select target column", df.columns)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        model = RandomForestClassifier()
        model.fit(X, y)
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importance})
        st.write(feature_importance.sort_values(by='importance', ascending=False))

    st.subheader("Machine Learning Models")
    if st.checkbox("Train and Evaluate ML Models"):
        model_type = st.selectbox("Select model type", ["Logistic Regression", "Decision Tree", "Random Forest"])
        target_col = st.selectbox("Select target column", df.columns)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode categorical variables
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            label_encoders[target_col] = le

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        if model_type == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression()
        elif model_type == "Decision Tree":
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier()
        elif model_type == "Random Forest":
            model = RandomForestClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("Classification Report:")
        st.write(classification_report(y_test, y_pred))

    st.subheader("Automated Report")
    report = ""
    report += f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.\n\n"
    report += "Summary Statistics:\n"
    report += df.describe().to_string() + "\n\n"
    report += "Missing Values:\n"
    report += df.isnull().sum().to_string() + "\n\n"
    report += "Correlation Matrix:\n"
    report += df[numeric_cols].corr().to_string() + "\n\n"
    st.text_area("Report", value=report, height=300)
    st.download_button(label="Download Report", data=report, file_name='report.txt', mime='text/plain')
