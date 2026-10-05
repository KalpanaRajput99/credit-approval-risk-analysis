import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# UCI Credit Approval dataset
# Source: https://archive.ics.uci.edu/dataset/27/credit+approval
# The UCI dataset has anonymized attributes A1-A15 and target A16.

columns = [f"A{i}" for i in range(1, 17)]
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data",
    header=None,
    names=columns,
    na_values="?"
)

# Convert target
df["A16"] = df["A16"].map({"+": "Approved", "-": "Rejected"})

# Convert known continuous attributes
numeric_cols = ["A2", "A3", "A8", "A11", "A14", "A15"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\n===== CREDIT APPROVAL ANALYSIS =====")
print("Dataset shape:", df.shape)
print("\nApproval summary:")
print(df["A16"].value_counts())
print("\nMissing values:")
print(df.isna().sum())

# Simple visualization
df["A16"].value_counts().plot(kind="bar")
plt.title("Credit Application Outcome")
plt.xlabel("Outcome")
plt.ylabel("Applications")
plt.tight_layout()
plt.show()

# Prepare ML data
X = df.drop(columns=["A16"])
y = df["A16"].map({"Approved": 1, "Rejected": 0})

categorical_cols = [c for c in X.columns if c not in numeric_cols]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\n===== MODEL PERFORMANCE =====")
print("Accuracy :", round(accuracy_score(y_test, pred), 4))
print("Precision:", round(precision_score(y_test, pred), 4))
print("Recall   :", round(recall_score(y_test, pred), 4))
print("F1 Score :", round(f1_score(y_test, pred), 4))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))
print("\nClassification Report:")
print(classification_report(y_test, pred, target_names=["Rejected", "Approved"]))
