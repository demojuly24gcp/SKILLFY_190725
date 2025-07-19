import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import dagshub
from dagshub import dagshub_logger
import mlflow
import joblib

# Load the wine quality dataset
aj_dir = os.path.dirname(os.path.abspath(__file__).replace('ECHICA', 'AJ'))
data_path = os.path.join(aj_dir, '../data/winequality-red.csv')
data = pd.read_csv(data_path)

# Features and target
X = data.drop('quality', axis=1)
y = data['quality']

# Feature engineering: add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42, stratify=y)

# Balance classes with SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Define base models with strong default parameters
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
et = ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)

# Voting ensemble
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('et', et)],
    voting='soft',
    n_jobs=-1
)

# Pipeline with scaling
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', ensemble)
])

# Fit pipeline on balanced data
pipeline.fit(X_train_bal, y_train_bal)

y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score (weighted): {f1:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")

# Save the trained pipeline as an artifact and local copy in ECHICA directory
model_path = os.path.join(os.path.dirname(__file__), "trained_pipeline.pkl")
joblib.dump(pipeline, model_path)

# Save the classification report as an artifact
report_path = "classification_report.txt"
with open(report_path, "w") as f:
    f.write(classification_report(y_test, y_pred))

mlflow.set_tracking_uri("https://dagshub.com/edurekajuly24gcp/SKILLFY_190725.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = "christian-echica"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "41377f1a29b4e15de301de109f1e47b8e017edfd"

dagshub.init(repo_owner='edurekajuly24gcp', repo_name='SKILLFY_190725', mlflow=True)

with mlflow.start_run():
    mlflow.set_tag("author", "Christian Echica")
    mlflow.log_param("model", "VotingClassifier")
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score_weighted", f1)
    mlflow.log_metric("precision_weighted", precision)
    mlflow.log_metric("recall_weighted", recall)
    mlflow.log_artifact(model_path)
    mlflow.log_artifact(report_path)

if __name__ == "__main__":
    import streamlit as st
    from pyngrok import ngrok
    import joblib

    # Set up ngrok
    NGROK_AUTH_TOKEN = "1XhoEKzAiOMOEJAyuWszV8h24cU_6Lqw3LYmKCGuJ9fauBh7r"
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    public_url = ngrok.connect(8501)
    st.write(f"Public URL: {public_url}")

    st.title("Wine Quality Prediction")
    st.write("Enter the wine characteristics below:")

    # Load the trained pipeline
    model_path = os.path.join(os.path.dirname(__file__), "trained_pipeline.pkl")
    if not os.path.exists(model_path):
        st.error("Trained model not found. Please train and save the model first.")
    else:
        pipeline = joblib.load(model_path)
        # Get feature names from the original dataset
        feature_names = list(pd.read_csv(data_path).drop('quality', axis=1).columns)
        # Create input fields for each feature
        user_input = []
        for feature in feature_names:
            val = st.number_input(f"{feature}", value=0.0, format="%f")
            user_input.append(val)
        # Generate polynomial features to match training
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_user = poly.fit_transform([user_input])
        # Feature selection (use the same indices as training)
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_selector.fit(X_train_bal, y_train_bal)
        importances = rf_selector.feature_importances_
        important_indices = importances.argsort()[-int(0.8 * len(importances)):]
        X_user_selected = X_user[:, important_indices]
        if st.button("Predict Quality"):
            pred = pipeline.predict(X_user_selected)
            st.success(f"Predicted Wine Quality: {pred[0]}")
