import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, mean_squared_error, r2_score, confusion_matrix
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import io
import base64

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Define LangChain prompt
prompt_template = PromptTemplate(
    input_variables=["dataset_summary", "task_type", "results_summary", "user_query"],
    template="""
    You are an expert data scientist and chatbot. The user has provided a dataset with the following summary:
    {dataset_summary}

    The task performed was: {task_type}
    Results summary: {results_summary}

    The user asked: {user_query}

    Provide a detailed response, including:
    - A summary of the dataset and its potential use cases in pharmaceutical procurement, supply chain, or banking fraud detection.
    - Interpretation of the results for the task performed.
    - Recommendations for further analysis or model improvements.
    - If forecasting is requested, suggest time-series approaches.
    Respond clearly and concisely in markdown format.
    """
)
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# Preprocess data
def preprocess_data(df, target_column=None, task_type=None, time_column=None):
    df = df.copy()
    df = df.fillna(df.mean(numeric_only=True))
    df = df.fillna(df.mode().iloc[0])
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    if task_type == "anomaly":
        X = df
        y = None
    elif task_type == "forecasting":
        y = df[target_column] if target_column in df else None
        X = df[[time_column]] if time_column in df else None
    else:
        X = df.drop(columns=[target_column]) if target_column in df else df
        y = df[target_column] if target_column in df else None

    if task_type != "forecasting":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    else:
        X_scaled = X
        scaler = None

    return X_scaled, y, scaler

# Train classification models
def train_classification(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    results = {}
    best_model = None
    best_score = 0
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        score = report['weighted avg']['f1-score']
        results[name] = {"model": model, "report": report, "y_pred": y_pred, "score": score}
        if score > best_score:
            best_score = score
            best_model = name
    return results, best_model

# Train regression models
def train_regression(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor()
    }
    results = {}
    best_model = None
    best_score = -float('inf')
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"model": model, "mse": mse, "r2": r2, "y_pred": y_pred, "score": r2}
        if r2 > best_score:
            best_score = r2
            best_model = name
    return results, best_model

# Perform anomaly detection
def detect_anomalies(X):
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    iso_labels = iso_forest.fit_predict(X)
    dbscan_labels = dbscan.fit_predict(X)
    iso_anomalies = np.sum(iso_labels == -1)
    dbscan_anomalies = np.sum(dbscan_labels == -1)
    best_model = "Isolation Forest" if iso_anomalies > dbscan_anomalies else "DBSCAN"
    return {"Isolation Forest": iso_labels, "DBSCAN": dbscan_labels}, best_model

# Perform forecasting
def perform_forecasting(y, periods=10):
    try:
        model = ARIMA(y, order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)
        return forecast, model_fit
    except:
        return None, None

# Generate visualizations
def generate_visualizations(df, task_type, results=None, target_column=None, anomalies=None, X_test=None, y_test=None, forecast=None):
    figs = []

    # Feature Distribution Plot
    plt.figure(figsize=(10, 6))
    for col in df.select_dtypes(include=[np.number]).columns[:5]:
        sns.histplot(df[col], kde=True, label=col)
    plt.title("Feature Distributions", fontsize=14)
    plt.legend()
    figs.append(plt.gcf())
    plt.close()

    if task_type == "classification" and results:
        for name, result in results.items():
            cm = confusion_matrix(y_test, result["y_pred"])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - {name}", fontsize=14)
            plt.xlabel("Predicted", fontsize=12)
            plt.ylabel("Actual", fontsize=12)
            figs.append(plt.gcf())
            plt.close()

    if task_type == "regression" and results:
        for name, result in results.items():
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, result["y_pred"], alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.title(f"Predicted vs Actual - {name}", fontsize=14)
            plt.xlabel("Actual", fontsize=12)
            plt.ylabel("Predicted", fontsize=12)
            figs.append(plt.gcf())
            plt.close()

    if task_type == "anomaly" and anomalies:
        plt.figure(figsize=(10, 6))
        plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=anomalies["Isolation Forest"], cmap='coolwarm', alpha=0.5)
        plt.title("Anomaly Detection - Isolation Forest", fontsize=14)
        plt.xlabel(df.columns[0], fontsize=12)
        plt.ylabel(df.columns[1], fontsize=12)
        figs.append(plt.gcf())
        plt.close()

    if task_type == "forecasting" and forecast is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label="Historical Data")
        plt.plot(range(len(y_test), len(y_test) + len(forecast)), forecast, label="Forecast", color='red')
        plt.title("Time Series Forecast", fontsize=14)
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.legend()
        figs.append(plt.gcf())
        plt.close()

    return figs

# Convert plot to base64
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# Streamlit App
def main():
    # Set page config
    st.set_page_config(page_title="ML & LLM Chatbot", page_icon="📊", layout="wide")

    # Custom CSS for enhanced UI
    st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 5px; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: bold; }
    .stTextInput, .stTextArea { border: 1px solid #4CAF50; border-radius: 5px; }
    .sidebar .sidebar-content { background-color: #ffffff; border-right: 1px solid #ddd; }
    h1, h2, h3 { color: #2c3e50; }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("Configure the app and upload your dataset.")
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
        generate_sample = st.button("Generate Sample Dataset", key="generate_sample")
        task_type = st.selectbox("Select Task", ["Classification", "Regression", "Anomaly Detection", "Forecasting"])

    # Main content
    st.title("📈 ML & LLM Chatbot for Advanced Analytics")
    st.markdown("""
    Welcome to the advanced ML and LLM-powered chatbot! Perform **classification**, **regression**, **anomaly detection**, or **forecasting** on your dataset.
    Upload a CSV or generate a sample dataset, then interact via queries. The app selects the best model based on performance metrics and provides insights.
    """)

    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None

    # Handle sample dataset generation
    if generate_sample:
        with st.spinner("Generating sample dataset..."):
            np.random.seed(42)
            n_samples = 1000
            sample_data = {
                "transaction_amount": np.random.normal(1000, 200, n_samples),
                "drug_quantity": np.random.randint(1, 100, n_samples),
                "supply_chain_cost": np.random.normal(5000, 1000, n_samples),
                "is_fraud": np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
                "category": np.random.choice(["pharma", "supply", "banking"], n_samples),
                "timestamp": pd.date_range(start="2023-01-01", periods=n_samples, freq="D")
            }
            st.session_state.df = pd.DataFrame(sample_data)
            st.session_state.df.to_csv("sample_dataset.csv", index=False)
            st.success("Sample dataset generated!")

    # Load dataset
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully!")

    # Display dataset preview
    if st.session_state.df is not None:
        st.subheader("📋 Dataset Preview")
        st.dataframe(st.session_state.df.head(), use_container_width=True)

        # Generate dataset summary for LLM
        dataset_summary = f"""
        Columns: {list(st.session_state.df.columns)}
        Shape: {st.session_state.df.shape}
        Numerical Columns: {list(st.session_state.df.select_dtypes(include=[np.number]).columns)}
        Categorical Columns: {list(st.session_state.df.select_dtypes(include=['object']).columns)}
        Missing Values: {st.session_state.df.isnull().sum().to_dict()}
        """

        # Tabs for different tasks
        tab1, tab2, tab3 = st.tabs(["Chatbot", "Results", "Visualizations"])

        with tab1:
            st.subheader("💬 Chatbot")
            user_query = st.text_area("Enter your query (e.g., 'Predict is_fraud', 'Forecast supply_chain_cost')", height=100)
            if user_query:
                with st.spinner("Processing query..."):
                    # Determine task type from user input or selection
                    selected_task = task_type.lower()
                    if "classification" in user_query.lower():
                        selected_task = "classification"
                    elif "regression" in user_query.lower():
                        selected_task = "regression"
                    elif "anomaly" in user_query.lower():
                        selected_task = "anomaly"
                    elif "forecast" in user_query.lower():
                        selected_task = "forecasting"

                    # Select target and time column
                    target_column = None
                    time_column = None
                    if selected_task in ["classification", "regression", "forecasting"]:
                        target_column = st.selectbox("Select target column", st.session_state.df.columns, key="target")
                    if selected_task == "forecasting":
                        time_column = st.selectbox("Select time column", st.session_state.df.columns, key="time")

                    if (selected_task in ["classification", "regression"] and target_column) or selected_task == "anomaly" or (selected_task == "forecasting" and target_column and time_column):
                        # Preprocess data
                        X, y, scaler = preprocess_data(st.session_state.df, target_column, selected_task, time_column)
                        results = None
                        best_model = None
                        figs = []
                        results_summary = ""

                        if selected_task == "classification":
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            results, best_model = train_classification(X_train, X_test, y_train, y_test)
                            results_summary = f"Best model: {best_model}, F1-Score: {results[best_model]['score']:.4f}"
                            figs = generate_visualizations(X, selected_task, results, target_column, X_test=X_test, y_test=y_test)

                        elif selected_task == "regression":
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            results, best_model = train_regression(X_train, X_test, y_train, y_test)
                            results_summary = f"Best model: {best_model}, R2: {results[best_model]['score']:.4f}"
                            figs = generate_visualizations(X, selected_task, results, target_column, X_test=X_test, y_test=y_test)

                        elif selected_task == "anomaly":
                            results, best_model = detect_anomalies(X)
                            results_summary = f"Best model: {best_model}, Anomalies detected: {np.sum(results[best_model] == -1)}"
                            figs = generate_visualizations(X, selected_task, anomalies=results)

                        elif selected_task == "forecasting":
                            forecast, model_fit = perform_forecasting(y)
                            if forecast is not None:
                                results_summary = f"Forecast for {len(forecast)} periods completed."
                                figs = generate_visualizations(st.session_state.df, selected_task, forecast=forecast, y_test=y)
                            else:
                                results_summary = "Forecasting failed. Ensure the target column is suitable for time-series analysis."

                        # Run LLM chain
                        llm_response = llm_chain.run(
                            dataset_summary=dataset_summary,
                            task_type=selected_task,
                            results_summary=results_summary,
                            user_query=user_query
                        )
                        st.markdown("### LLM Response")
                        st.markdown(llm_response)

        with tab2:
            st.subheader("📊 Model Results")
            if results:
                if selected_task == "classification":
                    for name, result in results.items():
                        st.write(f"**{name} (F1-Score: {result['score']:.4f})**")
                        st.json(result["report"])
                    st.success(f"Best Model: {best_model}")
                elif selected_task == "regression":
                    for name, result in results.items():
                        st.write(f"**{name}** - MSE: {result['mse']:.4f}, R2: {result['r2']:.4f}")
                    st.success(f"Best Model: {best_model}")
                elif selected_task == "anomaly":
                    st.write(f"**Isolation Forest Anomalies**: {np.sum(results['Isolation Forest'] == -1)}")
                    st.write(f"**DBSCAN Anomalies**: {np.sum(results['DBSCAN'] == -1)}")
                    st.success(f"Best Model: {best_model}")
                elif selected_task == "forecasting" and forecast is not None:
                    st.write("**ARIMA Forecast**")
                    st.dataframe(pd.DataFrame(forecast, columns=["Forecast"]), use_container_width=True)

        with tab3:
            st.subheader("📈 Visualizations")
            for fig in figs:
                st.image(fig_to_base64(fig), use_column_width=True)

if __name__ == "__main__":
    main()