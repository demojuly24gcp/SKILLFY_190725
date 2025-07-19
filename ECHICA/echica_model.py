import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

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
