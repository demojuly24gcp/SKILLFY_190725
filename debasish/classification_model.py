import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

# Load dataset
data = pd.read_csv(r'winequality-red.csv')  # Ensure this file is in your working directory

# Features and target
X = data.drop('quality', axis=1)
y = data['quality']

# Optionally, convert quality to binary classification (good/bad)
y = y.apply(lambda q: 1 if q >= 7 else 0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Print F1 score
print("F1 Score:", f1_score(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score (weighted): {f1:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")

dagshub.init(repo_owner='edurekajuly24gcp', repo_name='SKILLFY_190725', mlflow=True)

with mlflow.start_run():
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score_weighted", f1)
    mlflow.log_metric("precision_weighted", precision)
    mlflow.log_metric("recall_weighted", recall)
