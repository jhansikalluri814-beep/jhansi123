import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = "iris_logistic_model.pkl"
SCALER_PATH = "iris_scaler.pkl"


def train_and_save_model():
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return {
        "model": model,
        "scaler": scaler,
        "accuracy": accuracy,
        "report": report,
        "feature_names": feature_names,
        "target_names": target_names,
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train,
    }


def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict(model, scaler, values):
    values = np.array(values, dtype=float).reshape(1, -1)
    values_scaled = scaler.transform(values)
    prediction = model.predict(values_scaled)[0]
    probabilities = model.predict_proba(values_scaled)[0]
    return int(prediction), probabilities


def build_ui(train_data):
    st.set_page_config(page_title="Iris Logistic Regression", layout="centered")
    st.title("Logistic Regression Classifier for Iris")
    st.write(
        "This app trains a logistic regression model on the Iris dataset, saves the model, then lets you test custom feature values." 
    )

    st.subheader("Model training results")
    st.write(f"Accuracy on the test set: **{train_data['accuracy']:.4f}**")
    with st.expander("See full classification report"):
        st.text(train_data["report"])

    st.subheader("Input features")
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0, 0.1)
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.35, 0.1)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.3, 0.1)

    input_values = [sepal_length, sepal_width, petal_length, petal_width]
    st.write("**Selected input features:**")
    st.write(pd.DataFrame([input_values], columns=train_data["feature_names"]))

    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        st.error("Model files not found. Please rerun the script to train and save the model.")
        return

    prediction, probabilities = predict(model, scaler, input_values)
    prediction_name = train_data["target_names"][prediction]

    st.subheader("Prediction")
    st.write(f"Predicted Iris species: **{prediction_name}**")
    prob_df = pd.DataFrame(
        [probabilities],
        columns=[f"Prob: {name}" for name in train_data["target_names"]],
    )
    st.write(prob_df)

    st.subheader("Test sample preview")
    sample_index = st.number_input(
        "Select a test sample index", 0, len(train_data["X_test"]) - 1, 0, 1
    )
    sample_features = train_data["X_test"][sample_index]
    sample_label = train_data["y_test"][sample_index]
    sample_pred, sample_probs = predict(model, scaler, sample_features)
    st.write(pd.DataFrame([sample_features], columns=train_data["feature_names"]))
    st.write(f"True label: **{train_data['target_names'][sample_label]}**")
    st.write(f"Predicted label: **{train_data['target_names'][sample_pred]}**")


def main():
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        train_data = train_and_save_model()
    else:
        # Load data again for UI and evaluation if model already existed
        iris = load_iris()
        X = iris.data
        y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        y_pred = model.predict(scaler.transform(X_test))
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=iris.target_names)
        train_data = {
            "model": model,
            "scaler": scaler,
            "accuracy": accuracy,
            "report": report,
            "feature_names": iris.feature_names,
            "target_names": iris.target_names,
            "X_test": X_test,
            "y_test": y_test,
        }

    build_ui(train_data)


if __name__ == "__main__":
    main()
