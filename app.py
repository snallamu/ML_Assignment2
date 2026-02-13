# ------------------------------------------------------------
# Main Streamlit Application
#
# Compares multiple classification models on the Student
# Performance dataset and displays evaluation metrics,
# a classification report, and qualitative observations.
#
# Required Streamlit features implemented:
#   a. Dataset upload option (CSV — test data only)
#   b. Model selection dropdown
#   c. Evaluation metrics display
#   d. Confusion matrix and classification report
# ------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)

from model.dataprep import load_and_prepare_data, prepare_uploaded_data
from model.metrics import build_models, evaluate_model


# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Classification",
    layout="wide"
)

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("Student Performance Classification Models")
st.write(
    "This application trains and evaluates multiple machine learning "
    "classification models on the Student Performance dataset. "
    "Use the default UCI dataset or upload your own test CSV below."
)

st.markdown("---")

# ------------------------------------------------------------
# Sidebar — Controls
# ------------------------------------------------------------
st.sidebar.header("Configuration")

# b. Model selection dropdown
MODEL_OPTIONS = [
    "All Models",
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost",
]
selected_model = st.sidebar.selectbox(
    "Select Model to Evaluate",
    options=MODEL_OPTIONS,
    index=0,
    help="Choose a specific model or run all six at once."
)

st.sidebar.markdown("---")

# a. Dataset upload option (CSV — test data only, as per assignment spec)
st.sidebar.subheader("Upload Test Data (Optional)")
st.sidebar.write(
    "Upload a CSV file containing test features and a `performance_level` "
    "column (0–4). Leave blank to use the built-in UCI dataset."
)
uploaded_file = st.sidebar.file_uploader(
    "Upload test CSV",
    type=["csv"],
    help="CSV must contain the same features as the UCI Student dataset "
         "plus a 'performance_level' target column (values 0–4)."
)

st.sidebar.markdown("---")
run_button = st.sidebar.button("Run Models", use_container_width=True)

# ------------------------------------------------------------
# Class label mapping (for readability in reports)
# ------------------------------------------------------------
CLASS_LABELS = {
    0: "Very Poor (0–9)",
    1: "Poor (10–11)",
    2: "Satisfactory (12–13)",
    3: "Good (14–15)",
    4: "Excellent (16–20)",
}

# ------------------------------------------------------------
# Observations per model
# ------------------------------------------------------------
OBSERVATIONS = {
    "Logistic Regression": (
        "Served as a reliable baseline model with consistent performance, "
        "but its linear decision boundary limits its ability to capture the "
        "non-linear relationships present in this multi-class problem."
    ),
    "Decision Tree": (
        "Captured feature interactions effectively; however, its performance "
        "showed fluctuation across runs, indicating a tendency to overfit the "
        "training data without pruning."
    ),
    "KNN": (
        "Produced moderate results but proved sensitive to feature scaling "
        "and the choice of neighbourhood size, which affected prediction "
        "stability across different sample distributions."
    ),
    "Naive Bayes": (
        "Executed very quickly and provided acceptable baseline results, "
        "though the strong feature-independence assumption reduced accuracy "
        "on the correlated demographic and academic features in this dataset."
    ),
    "Random Forest": (
        "Demonstrated strong and consistent performance by aggregating "
        "predictions across 100 decision trees, which effectively reduced "
        "variance and improved generalisation over single-tree models."
    ),
    "XGBoost": (
        "Delivered the best overall results, benefiting from sequential "
        "boosting, built-in regularisation, and its efficient capacity to "
        "learn complex patterns from tabular data."
    ),
}


# ------------------------------------------------------------
# Helper — render one model's full results
# ------------------------------------------------------------
def display_model_results(name, model, X_test, y_test):
    st.subheader(f"Model: {name}")

    # --- Metrics ---
    metrics = evaluate_model(model, X_test, y_test)
    metrics_display = {k: round(v, 4) for k, v in metrics.items()}
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    for col, (metric_name, value) in zip(
        [col1, col2, col3, col4, col5, col6], metrics_display.items()
    ):
        col.metric(label=metric_name, value=f"{value:.4f}")

    st.markdown("")

    # d. Confusion matrix + classification report (side by side)
    col_left, col_right = st.columns([1, 1])

    y_pred = model.predict(X_test)
    label_names = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS)]

    # Confusion Matrix
    with col_left:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=list(CLASS_LABELS.values())
        )
        disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
        ax.set_title(f"{name}", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Classification Report
    with col_right:
        st.markdown("**Classification Report**")
        report_str = classification_report(
            y_test, y_pred,
            target_names=label_names,
            zero_division=0
        )
        st.code(report_str, language=None)

    # Observation
    st.info(f"**Observation:** {OBSERVATIONS.get(name, '')}")
    st.markdown("---")


# ------------------------------------------------------------
# Main execution block
# ------------------------------------------------------------
if run_button:

    with st.spinner("Loading data and training models..."):

        # ── Data source: uploaded CSV or built-in UCI dataset ──
        if uploaded_file is not None:
            try:
                test_df = pd.read_csv(uploaded_file)
                st.success(
                    f"Uploaded CSV loaded: {test_df.shape[0]} rows, "
                    f"{test_df.shape[1]} columns."
                )
                # Prepare uploaded data (uses fixed UCI training split)
                X_train, X_test, y_train, y_test = prepare_uploaded_data(
                    test_df
                )
                st.info(
                    "Training on the UCI dataset; evaluating on your "
                    "uploaded test set."
                )
            except Exception as e:
                st.error(f"Could not process uploaded file: {e}")
                st.stop()
        else:
            X_train, X_test, y_train, y_test = load_and_prepare_data()
            st.info("Using built-in UCI Student Performance dataset.")

        # ── Build and train all models ──
        all_models = build_models()
        for model in all_models.values():
            model.fit(X_train, y_train)

    st.success("Training complete!")
    st.markdown("---")

    # ── c. Overall Model Comparison Table ──
    st.header("Model Comparison Table")

    rows = []
    for name, model in all_models.items():
        m = evaluate_model(model, X_test, y_test)
        m["Model"] = name
        rows.append(m)

    results_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(results_df.round(4), use_container_width=True)
    st.markdown("---")

    # ── Per-model detailed view ──
    st.header("Detailed Model Results")

    if selected_model == "All Models":
        for name, model in all_models.items():
            display_model_results(name, model, X_test, y_test)
    else:
        if selected_model in all_models:
            display_model_results(
                selected_model,
                all_models[selected_model],
                X_test,
                y_test
            )
        else:
            st.warning(f"Model '{selected_model}' not found.")

    # ── Summary Observations Table ──
    st.header("Summary: Observations on Model Performance")
    obs_df = pd.DataFrame(
        list(OBSERVATIONS.items()),
        columns=["ML Model", "Observation"]
    )
    st.table(obs_df)
