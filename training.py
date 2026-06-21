# training.py

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score,
    precision_recall_curve,
)


sns.set_theme(style="whitegrid")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


df = pd.read_csv("data/alarm_logs.csv")
print(f"Shape: {df.shape}")
print(df.head())

theft_rate = df["label"].mean()
print(f"\nTheft rate: {theft_rate:.1%}")
print("(If we predicted 'false alarm' for every single row, we'd still be "
      f"right ~{1 - theft_rate:.0%} of the time -- but catch ZERO real thefts!)")
ax = sns.countplot(data=df, x="label")
ax.set_xticks([0, 1])
ax.set_xticklabels(["False Alarm (0)", "Real Theft (1)"])
ax.set_title("Class balance: real thefts are RARE")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}",
                 (p.get_x() + p.get_width() / 2, p.get_height()),
                 ha="center", va="bottom")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_class_balance.png", dpi=120)
plt.close()
print(f"Saved chart -> {OUTPUT_DIR}/01_class_balance.png")


section("SECTION 2: Train a baseline model")

FEATURES = [
    "motion_sensor", "pressure_pad", "glass_break",
    "infrared_beam", "hour_of_day", "after_hours", "humidity_spike",
]

X = df[FEATURES]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

print(f"Trained on {len(X_train)} rows, testing on {len(X_test)} rows")

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["False Alarm", "Theft"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix @ threshold = 0.5")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_confusion_matrix.png", dpi=120)
plt.close()
print(f"Saved chart -> {OUTPUT_DIR}/02_confusion_matrix.png")

tn, fp, fn, tp = cm.ravel()
print(f"\nRaw counts: TN={tn}  FP={fp}  FN={fn}  TP={tp}")

p = precision_score(y_test, y_pred)
r = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {p:.2f}  -> of the alarms we raised, {p:.0%} were real thefts")
print(f"Recall:    {r:.2f}  -> of all real thefts, we caught {r:.0%} of them")
print(f"F1 score:  {f1:.2f}")


y_proba = model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.05, 1.0, 0.01)
precisions, recalls, f1s = [], [], []

for t in thresholds:
  
    preds_t = (y_proba >= t).astype(int)

    precisions.append(precision_score(y_test, preds_t, zero_division=0))
    recalls.append(recall_score(y_test, preds_t, zero_division=0))
    f1s.append(f1_score(y_test, preds_t, zero_division=0))

best_f1_idx = int(np.argmax(f1s))
best_f1_threshold = thresholds[best_f1_idx]

plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions, label="Precision")
plt.plot(thresholds, recalls, label="Recall")
plt.plot(thresholds, f1s, label="F1", linewidth=2)
plt.axvline(best_f1_threshold, color="black", linestyle="--",
            label=f"Best F1 threshold = {best_f1_threshold:.2f}")
plt.xlabel("Decision threshold")
plt.ylabel("Score")
plt.title("Precision / Recall / F1 vs Threshold")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_threshold_tuning.png", dpi=120)
plt.close()
print(f"Saved chart -> {OUTPUT_DIR}/03_threshold_tuning.png")

print(f"\nBest F1 = {f1s[best_f1_idx]:.2f} at threshold = {best_f1_threshold:.2f}")


prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(6, 6))
plt.plot(rec_curve, prec_curve)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_precision_recall_curve.png", dpi=120)
plt.close()
print(f"Saved chart -> {OUTPUT_DIR}/04_precision_recall_curve.png")



COST_PER_FALSE_NEGATIVE = 5000   
COST_PER_FALSE_POSITIVE = 50    

costs = []
for t in thresholds:
    preds_t = (y_proba >= t).astype(int)

    
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, preds_t, labels=[0, 1]).ravel()

    total_cost = (fn_t * COST_PER_FALSE_NEGATIVE) + (fp_t * COST_PER_FALSE_POSITIVE)
    costs.append(total_cost)

best_cost_idx = int(np.argmin(costs))
best_cost_threshold = thresholds[best_cost_idx]

plt.figure(figsize=(8, 5))
plt.plot(thresholds, costs, label="Total operational cost (£)")
plt.axvline(best_cost_threshold, color="green", linestyle="--",
            label=f"Cost-optimal threshold = {best_cost_threshold:.2f}")
plt.axvline(best_f1_threshold, color="black", linestyle=":",
            label=f"F1-optimal threshold = {best_f1_threshold:.2f}")
plt.xlabel("Decision threshold")
plt.ylabel("Total cost (£) on the test set")
plt.title("Operational Cost vs Threshold")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_cost_tuning.png", dpi=120)
plt.close()
print(f"Saved chart -> {OUTPUT_DIR}/05_cost_tuning.png")

print(f"\nF1-optimal threshold:   {best_f1_threshold:.2f}  "
      f"(F1 = {f1s[best_f1_idx]:.2f}, cost = £{costs[best_f1_idx]:,.0f})")
print(f"Cost-optimal threshold: {best_cost_threshold:.2f}  "
      f"(F1 = {f1s[best_cost_idx]:.2f}, cost = £{costs[best_cost_idx]:,.0f})")


section(f"DONE. All charts saved in ./{OUTPUT_DIR}/")
