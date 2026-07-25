"""
Radar Signal Classification using Random Forest
Dataset: Johns Hopkins University Ionosphere Database (UCI)
Classifies radar returns as 'good' (structure detected in ionosphere)
or 'bad' (signal passed through) — 34 continuous features per sample.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_curve, auc
)
import joblib

# ---------- Load data ----------
cols = [f"feat_{i+1}" for i in range(34)] + ["label"]
df = pd.read_csv("ionosphere.csv", header=None, names=cols)

X = df.drop(columns=["label"])
y = df["label"].map({"g": 1, "b": 0})  # good=1, bad=0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------- Train ----------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ---------- Evaluate ----------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["bad", "good"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# ---------- Save confusion matrix plot ----------
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(["bad", "good"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["bad", "good"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix (Acc: {acc:.2%})")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)

# ---------- Feature importance ----------
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(6, 6))
importances.head(15).sort_values().plot(kind="barh", ax=ax, color="#2563eb")
ax.set_title("Top 15 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)

# ---------- ROC curve ----------
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", color="#16a34a")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve"); ax.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)

# ---------- Save model ----------
joblib.dump(model, "rf_model.pkl")
X.to_csv("feature_reference.csv", index=False)  # for sample defaults in app

print("\nSaved: rf_model.pkl, confusion_matrix.png, feature_importance.png, roc_curve.png")
