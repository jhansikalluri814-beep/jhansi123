from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "logistic_regression_model.joblib"
RANDOM_STATE = 42


def load_classification_data():
    """Load a built-in binary classification dataset."""
    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")
    target_names = list(dataset.target_names)
    return X, y, target_names


def train_and_save_model():
    X, y, target_names = load_classification_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
            ),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred)

    bundle = {
        "model": model,
        "feature_names": list(X.columns),
        "target_names": target_names,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
    }
    joblib.dump(bundle, MODEL_PATH)
    return bundle


@st.cache_resource
def get_model_bundle():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train_and_save_model()


def render_metrics(bundle):
    st.subheader("Model Test Result")
    st.metric("Test Accuracy", f"{bundle['accuracy']:.2%}")

    st.write("Confusion Matrix")
    st.dataframe(
        pd.DataFrame(
            bundle["confusion_matrix"],
            index=[f"Actual {name}" for name in bundle["target_names"]],
            columns=[f"Predicted {name}" for name in bundle["target_names"]],
        ),
        use_container_width=True,
    )

    st.write("Classification Report")
    st.dataframe(pd.DataFrame(bundle["classification_report"]).T, use_container_width=True)


def render_prediction_ui(bundle):
    X, _, _ = load_classification_data()
    feature_names = bundle["feature_names"]

    st.subheader("Prediction")
    st.caption("Adjust the input values, then click Predict.")

    with st.form("prediction_form"):
        selected_input_mode = st.radio(
            "Input mode",
            ["Use average sample values", "Enter all feature values"],
            horizontal=True,
        )

        values = {}
        if selected_input_mode == "Use average sample values":
            st.info("The app will use the dataset mean for every feature.")
            mean_values = X.mean()
            values = {feature: float(mean_values[feature]) for feature in feature_names}
        else:
            cols = st.columns(3)
            for index, feature in enumerate(feature_names):
                values[feature] = cols[index % 3].number_input(
                    feature,
                    value=float(X[feature].mean()),
                    min_value=float(X[feature].min()),
                    max_value=float(X[feature].max()),
                    step=float((X[feature].max() - X[feature].min()) / 100),
                )

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([values], columns=feature_names)
        prediction = int(bundle["model"].predict(input_df)[0])
        probabilities = bundle["model"].predict_proba(input_df)[0]
        predicted_label = bundle["target_names"][prediction]

        st.success(f"Predicted class: {predicted_label}")
        st.write("Prediction Probability")
        st.dataframe(
            pd.DataFrame(
                {
                    "Class": bundle["target_names"],
                    "Probability": probabilities,
                }
            ),
            use_container_width=True,
        )


def main():
    st.set_page_config(page_title="Logistic Regression Classifier", layout="wide")
    st.title("Logistic Regression Classification Model")
    st.write(
        "This app trains a logistic regression model on a classification dataset, "
        "saves it as a `.joblib` file, tests it, and lets you make predictions."
    )

    bundle = get_model_bundle()

    st.sidebar.header("Model")
    st.sidebar.write(f"Saved model: `{MODEL_PATH.name}`")

    if st.sidebar.button("Retrain and Save Model"):
        bundle = train_and_save_model()
        get_model_bundle.clear()
        st.sidebar.success("Model retrained and saved.")

    tab_predict, tab_metrics, tab_data = st.tabs(["Predict", "Test Result", "Dataset"])

    with tab_predict:
        render_prediction_ui(bundle)

    with tab_metrics:
        render_metrics(bundle)

    with tab_data:
        X, y, target_names = load_classification_data()
        data = X.copy()
        data["target"] = y
        data["target_name"] = y.map({index: name for index, name in enumerate(target_names)})
        st.subheader("Classification Data")
        st.dataframe(data.head(100), use_container_width=True)
        st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")


if __name__ == "__main__":
    main()
