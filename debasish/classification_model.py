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