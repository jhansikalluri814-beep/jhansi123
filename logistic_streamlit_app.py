import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "logistic_iris_model.joblib")
RANDOM_STATE = 42


@st.cache_data
def load_classification_data():
    iris = load_iris()
    features = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = pd.Series(iris.target, name="target")
    return features, target, iris.target_names


def train_and_save_model(features, target):
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "x_test": x_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }

    joblib.dump({"model": model, "metrics": metrics}, MODEL_PATH)
    return model, metrics


def load_or_train_model(features, target):
    if os.path.exists(MODEL_PATH):
        saved = joblib.load(MODEL_PATH)
        return saved["model"], saved["metrics"]

    return train_and_save_model(features, target)


def predict_flower(model, input_values, class_names):
    input_array = np.array(input_values).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    probabilities = model.predict_proba(input_array)[0]
    return class_names[prediction], probabilities


def main():
    st.set_page_config(page_title="Logistic Regression Classifier", layout="wide")

    st.title("Logistic Regression Classification Model")
    st.write(
        "This app trains a logistic regression model on the Iris classification dataset, "
        "saves the model, tests it, and uses the saved model for prediction."
    )

    features, target, class_names = load_classification_data()
    model, metrics = load_or_train_model(features, target)

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Dataset Preview")
        preview = features.copy()
        preview["class"] = target.map(lambda value: class_names[value])
        st.dataframe(preview.head(12), use_container_width=True)

        st.subheader("Model Test Results")
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")

        st.write("Confusion Matrix")
        confusion_df = pd.DataFrame(
            metrics["confusion_matrix"],
            index=class_names,
            columns=class_names,
        )
        st.dataframe(confusion_df, use_container_width=True)

        st.write("Classification Report")
        report_df = pd.DataFrame(metrics["classification_report"]).transpose()
        st.dataframe(report_df, use_container_width=True)

    with right:
        st.subheader("Make a Prediction")
        st.write("Enter flower measurements below.")

        input_values = []
        for column in features.columns:
            input_values.append(
                st.number_input(
                    label=column,
                    min_value=0.0,
                    max_value=10.0,
                    value=float(features[column].mean()),
                    step=0.1,
                )
            )

        if st.button("Predict Class"):
            predicted_class, probabilities = predict_flower(
                model,
                input_values,
                class_names,
            )
            st.success(f"Predicted class: {predicted_class}")

            probability_df = pd.DataFrame(
                {
                    "class": class_names,
                    "probability": probabilities,
                }
            )
            st.bar_chart(probability_df.set_index("class"))

        st.subheader("Saved Model")
        st.code(os.path.abspath(MODEL_PATH))

        if st.button("Retrain and Save Model"):
            model, metrics = train_and_save_model(features, target)
            st.success("Model retrained and saved successfully. Refresh to see updated metrics.")


if __name__ == "__main__":
    main()
