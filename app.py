"""
Customer Churn Prediction Dashboard
------------------------------------
A Streamlit web app that lets a user upload a churn dataset (CSV/XLSX),
automatically preprocesses it, runs EDA, trains a Random Forest model,
and shows predictions, probabilities, risk levels, and model diagnostics.

HOW TO RUN (you are using IDLE, so please read this):
IDLE cannot "run" a Streamlit app directly (pressing F5 will just execute
the script and do nothing visible, or throw a warning). Streamlit apps must
be launched from a terminal / command prompt using the `streamlit run`
command. Steps:

1. Save this file as app.py (already done if you downloaded it as app.py).
2. Open Command Prompt (Windows) or Terminal (Mac/Linux).
3. Navigate to the folder containing app.py, e.g.:
       cd C:\\Users\\lenovo\\Desktop\\churn_app
4. Install the required libraries (only once):
       pip install streamlit pandas numpy scikit-learn seaborn matplotlib plotly openpyxl
5. Run the app:
       streamlit run app.py
6. It will open automatically in your browser at http://localhost:8501

You can keep editing app.py in IDLE, save it, and Streamlit will offer to
"Rerun" automatically in the browser tab.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# PAGE CONFIG + STYLING
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    h1, h2, h3 {font-weight: 700;}
    .stTabs [data-baseweb="tab"] {font-size: 16px; font-weight: 600;}
    .risk-high {color:#EF553B; font-weight:700;}
    .risk-medium {color:#FFA15A; font-weight:700;}
    .risk-low {color:#00CC96; font-weight:700;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Customer Churn Prediction Dashboard")
st.caption("Upload a customer dataset → preprocess → explore → predict churn with Random Forest")


def metric_card(label, value):
    """Render a KPI card using plain HTML/CSS so the colors are always
    fully under our control (Streamlit's built-in st.metric widget
    applies its own internal styling that can override custom CSS)."""
    st.markdown(
        f"""
        <div style="background-color:#161b22;border:1px solid #2d333b;
                    border-radius:12px;padding:14px 12px;min-height:88px;">
            <div style="color:#9ca3af;font-size:0.8rem;font-weight:600;
                        margin-bottom:6px;line-height:1.2;">{label}</div>
            <div style="color:#ffffff;font-size:1.6rem;font-weight:700;
                        line-height:1.2;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------------------------------------------------
# SIDEBAR - FILE UPLOAD
# ----------------------------------------------------------------------
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or XLSX file", type=["csv", "xlsx"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Settings")
test_size_pct = st.sidebar.slider("Test set size (%)", 10, 40, 20)
n_estimators = st.sidebar.select_slider("Number of trees (n_estimators)", options=[100, 200, 300, 400, 500], value=200)

st.sidebar.markdown("---")
high_thresh = st.sidebar.slider("High risk threshold", 0.5, 0.95, 0.70, 0.05)
medium_thresh = st.sidebar.slider("Medium risk threshold", 0.2, 0.5, 0.40, 0.05)

# Reset trained model if a new file is uploaded
if uploaded_file is not None:
    current_name = uploaded_file.name
    if st.session_state.get("last_file") != current_name:
        for key in ["model", "X_test", "y_test", "y_pred", "y_prob", "id_test", "X_columns"]:
            st.session_state.pop(key, None)
        st.session_state["last_file"] = current_name

# ----------------------------------------------------------------------
# MAIN APP
# ----------------------------------------------------------------------
if uploaded_file is not None:

    # ---- Load data ----
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    # ---- Identify customer ID column ----
    id_col = None
    for c in df.columns:
        if c.lower().replace("_", "") in ["customerid", "id"]:
            id_col = c
            break

    if id_col:
        customer_ids = df[id_col]
        df_proc = df.drop(columns=[id_col])
    else:
        customer_ids = pd.Series(range(len(df)), name="RowID")
        df_proc = df.copy()

    if "Churn" not in df_proc.columns:
        st.error("⚠️ This dataset must contain a 'Churn' column (Yes/No or 1/0) to train the model.")
        st.stop()

    # ------------------------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------------------------
    st.subheader("🧹 Automatic Preprocessing")

    # Convert TotalCharges-like numeric-looking object columns
    for col in df_proc.columns:
        if df_proc[col].dtype == "object" and col != "Churn":
            converted = pd.to_numeric(df_proc[col], errors="coerce")
            # If most values convert successfully, treat as numeric
            if converted.notnull().mean() > 0.9:
                df_proc[col] = converted

    missing_before = int(df_proc.isnull().sum().sum())

    # Fill numeric missing values with median
    for col in df_proc.select_dtypes(include=[np.number]).columns:
        if df_proc[col].isnull().sum() > 0:
            df_proc[col] = df_proc[col].fillna(df_proc[col].median())

    # Fill categorical missing values with mode
    for col in df_proc.select_dtypes(include=["object"]).columns:
        if df_proc[col].isnull().sum() > 0:
            df_proc[col] = df_proc[col].fillna(df_proc[col].mode()[0])

    missing_after = int(df_proc.isnull().sum().sum())

    p1, p2, p3 = st.columns(3)
    with p1:
        metric_card("Missing values (before)", missing_before)
    with p2:
        metric_card("Missing values (after)", missing_after)
    with p3:
        metric_card("Duplicate rows", int(df_proc.duplicated().sum()))

    # Robust target encoding
    churn_col = None
    for col in df_proc.columns:
        if col.strip().lower() == "churn":
            churn_col = col
            break

    if churn_col is None:
        st.error("⚠️ Dataset must contain a 'Churn' column.")
        st.stop()

    if churn_col != "Churn":
        df_proc.rename(columns={churn_col: "Churn"}, inplace=True)

    df_proc["Churn"] = (
        df_proc["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "yes": 1,
            "no": 0,
            "1": 1,
            "0": 0,
            "true": 1,
            "false": 0
        })
    )

    if df_proc["Churn"].isna().any():
        invalid_values = df_proc.loc[df_proc["Churn"].isna(), "Churn"].unique()
        st.error("Invalid values found in the Churn column.")
        st.write("Unsupported values:", invalid_values)
        st.stop()

    df_proc["Churn"] = df_proc["Churn"].astype(int)

    # Encode categorical features (keep a display copy before encoding)
    df_display = df_proc.copy()
    df_encoded = df_proc.copy()
    encoders = {}
    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object":
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le

    st.success("Preprocessing complete: missing values handled, categorical columns encoded, target mapped to 0/1.")

    # ------------------------------------------------------------------
    # EDA
    # ------------------------------------------------------------------
    st.subheader("📈 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["Correlation Heatmap", "Churn Distribution", "Feature Explorer"])

    with tab1:
        numeric_df = df_encoded.apply(pd.to_numeric, errors="coerce").fillna(0)
        if numeric_df.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(9, 7))
            sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax, center=0)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns available for correlation heatmap.")

    with tab2:
        churn_counts = df_display["Churn"].map({0: "No", 1: "Yes"}).value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        fig2 = px.bar(
            churn_counts, x="Churn", y="Count", color="Churn", text="Count",
            color_discrete_map={"Yes": "#EF553B", "No": "#00CC96"},
            title="Churn Distribution"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        numeric_cols = [c for c in df_display.select_dtypes(include=[np.number]).columns if c != "Churn"]
        if numeric_cols:
            selected_col = st.selectbox("Choose a numeric feature to compare against Churn", numeric_cols)
            fig3 = px.histogram(
                df_display, x=selected_col,
                color=df_display["Churn"].map({0: "No", 1: "Yes"}),
                barmode="overlay", opacity=0.7,
                title=f"{selected_col} distribution by Churn status",
                labels={"color": "Churn"}
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No numeric columns available to explore.")

    # ------------------------------------------------------------------
    # MODEL TRAINING
    # ------------------------------------------------------------------
    st.subheader("🤖 Train the Model")

    X = df_encoded.drop("Churn", axis=1)
    y = df_encoded["Churn"]

    train_clicked = st.button("Train Random Forest Model", type="primary")

    if train_clicked:
        with st.spinner("Training Random Forest model..."):
            X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
                X, y, customer_ids, test_size=test_size_pct / 100,
                random_state=42, stratify=y
            )

            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            st.session_state["model"] = model
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test
            st.session_state["y_pred"] = y_pred
            st.session_state["y_prob"] = y_prob
            st.session_state["id_test"] = id_test
            st.session_state["X_columns"] = X.columns
            st.session_state["df_display"] = df_display

        st.success("✅ Model trained successfully!")

    # ------------------------------------------------------------------
    # RESULTS (shown once a model has been trained this session)
    # ------------------------------------------------------------------
    if "model" in st.session_state:
        model = st.session_state["model"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]
        y_prob = st.session_state["y_prob"]
        id_test = st.session_state["id_test"]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # ---- KPI Cards ----
        st.subheader("📌 Key Performance Indicators")
        k1, k2, k3 = st.columns(3)
        with k1:
            metric_card("Total Customers", len(df_display))
        with k2:
            metric_card("Churn Rate", f"{df_display['Churn'].mean() * 100:.1f}%")
        with k3:
            metric_card("Accuracy", f"{acc * 100:.1f}%")

        k4, k5, k6 = st.columns(3)
        with k4:
            metric_card("Precision", f"{prec * 100:.1f}%")
        with k5:
            metric_card("Recall", f"{rec * 100:.1f}%")
        with k6:
            metric_card("ROC-AUC", f"{auc:.3f}")

        # ---- Confusion Matrix + ROC Curve ----
        st.subheader("Model Diagnostics")
        m1, m2 = st.columns(2)

        with m1:
            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3.2))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                        xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)

        with m2:
            st.write("**ROC Curve**")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Random Forest (AUC={auc:.3f})"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                                          line=dict(dash="dash", color="red")))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        st.write("**Classification Report**")
        report = classification_report(y_test, y_pred, target_names=["No", "Yes"], output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

        # ---- Feature Importance ----
        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            "Feature": st.session_state["X_columns"],
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        fig_imp = px.bar(
            importance, x="Importance", y="Feature", orientation="h",
            title="Feature Importance (Random Forest)"
        )
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(t=40))
        st.plotly_chart(fig_imp, use_container_width=True)

        # ---- Customer-wise Prediction + Risk Categorization ----
        st.subheader("👥 Customer-wise Churn Prediction")

        def risk_category(p):
            if p >= high_thresh:
                return "High"
            elif p >= medium_thresh:
                return "Medium"
            else:
                return "Low"

        results = pd.DataFrame({
            "CustomerID": id_test.values,
            "Actual Churn": y_test.map({0: "No", 1: "Yes"}).values,
            "Predicted Churn": pd.Series(y_pred, index=y_test.index).map({0: "No", 1: "Yes"}).values,
            "Churn Probability": np.round(y_prob, 3),
        })
        results["Risk Level"] = results["Churn Probability"].apply(risk_category)

        risk_filter = st.multiselect(
            "Filter by Risk Level", ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )
        filtered_results = results[results["Risk Level"].isin(risk_filter)].sort_values(
            "Churn Probability", ascending=False
        )

        r1, r2, r3 = st.columns(3)
        with r1:
            metric_card("High Risk Customers", int((results["Risk Level"] == "High").sum()))
        with r2:
            metric_card("Medium Risk Customers", int((results["Risk Level"] == "Medium").sum()))
        with r3:
            metric_card("Low Risk Customers", int((results["Risk Level"] == "Low").sum()))

        st.dataframe(filtered_results, use_container_width=True, height=400)

        csv_bytes = filtered_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Predictions as CSV",
            data=csv_bytes,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

    else:
        st.info("Click **'Train Random Forest Model'** above to generate predictions and metrics.")

else:
    st.info("Upload a CSV or XLSX file from the sidebar to get started.")
    st.markdown("""
    ### Expected dataset format
    Your file should look like the Telco Customer Churn dataset, containing:
    - A customer ID column (optional, e.g. `customerID`)
    - Feature columns (e.g. `gender`, `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, etc.)
    - A target column named **`Churn`** with values `Yes`/`No` (or `1`/`0`)
    """)
