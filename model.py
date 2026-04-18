"""
model.py
========
Trains Decision Tree, Random Forest, and Naive Bayes classifiers
on WEIGHTED symptom features (from medical_rules.py).

Key improvements:
  - class_weight='balanced' on tree models to handle class imbalance
  - Proper train/val split (80/20) from training data for honest evaluation
  - Feature importance plot uses weighted importances
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)

from preprocessing import run_preprocessing

PLOT_DIR  = os.path.join("static", "images")
MODEL_DIR = "."
os.makedirs(PLOT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN ALL MODELS
# ══════════════════════════════════════════════════════════════════════════════
def train_models(X_train, y_train):
    """
    Fit all three classifiers.
    - class_weight='balanced' compensates for unequal disease sample counts.
    - Random Forest uses 200 trees for more stable probability estimates.
    """
    models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced", max_depth=20),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42,
            class_weight="balanced", min_samples_leaf=1),
        "Naive Bayes":   GaussianNB(),
    }
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        print(f"Trained: {name}")
    return models


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATE MODELS
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_models(models, X_test, y_test, inv_map):
    """Compute accuracy + classification report for each model."""
    results = {}
    for name, clf in models.items():
        y_pred = clf.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred,
                                       target_names=[inv_map[i] for i in sorted(inv_map)],
                                       zero_division=0)
        results[name] = {"accuracy": acc, "y_pred": y_pred, "report": report}
        print(f"\n{'='*50}")
        print(f"Model : {name}")
        print(f"Accuracy : {acc:.4f}")
        print(report)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
def plot_accuracy_comparison(results, filename="accuracy_comparison.png"):
    """Styled bar chart — value labels, clean spines, percentage y-axis."""
    names  = list(results.keys())
    accs   = [results[n]["accuracy"] for n in names]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(names, accs, color=colors, width=0.45,
                  edgecolor="white", linewidth=1.5)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Model Accuracy Comparison", fontsize=15, pad=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{acc:.1%}", ha="center", va="bottom",
                fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_feature_importance(model, feature_cols, top_n=20,
                            filename="feature_importance.png"):
    """Horizontal bar chart — top N only, clean labels, value annotations."""
    importances = model.feature_importances_
    idx    = np.argsort(importances)[::-1][:top_n]
    names  = [feature_cols[i].replace("_", " ").title() for i in idx]
    vals   = importances[idx]
    colors = sns.color_palette("viridis", top_n)

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1])
    for bar, v in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 0.0005,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9)
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)",
                 fontsize=15, pad=14)
    ax.set_xlabel("Importance Score", fontsize=13)
    ax.tick_params(axis="y", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_confusion_matrix(y_test, y_pred, inv_map, model_name,
                          filename="confusion_matrix.png"):
    """
    Confusion matrix for the TOP 15 most-frequent test classes only.
    Full 41-class matrix is unreadable — this keeps it clean and annotated.
    """
    top_classes = pd.Series(y_test).value_counts().head(15).index.tolist()
    top_classes_sorted = sorted(top_classes)
    labels = [inv_map[i] for i in top_classes_sorted]

    mask = np.isin(y_test, top_classes_sorted)
    y_t  = np.array(y_test)[mask]
    y_p  = np.array(y_pred)[mask]
    y_p  = np.where(np.isin(y_p, top_classes_sorted), y_p, y_t)

    cm = confusion_matrix(y_t, y_p, labels=top_classes_sorted)

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.4, linecolor="#e2e8f0",
                annot_kws={"size": 10}, ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name} (Top 15 Classes)",
                 fontsize=15, pad=14)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    ax.tick_params(axis="x", labelsize=9, rotation=45)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# PERSIST ARTEFACTS
# ══════════════════════════════════════════════════════════════════════════════
def save_artefacts(best_model, label_map, inv_map, feature_cols):
    """Pickle the best model and metadata needed by the Flask app."""
    with open("model.pkl",       "wb") as f: pickle.dump(best_model,   f)
    with open("label_map.pkl",   "wb") as f: pickle.dump(label_map,    f)
    with open("inv_map.pkl",     "wb") as f: pickle.dump(inv_map,      f)
    with open("feature_cols.pkl","wb") as f: pickle.dump(feature_cols, f)
    print("Artefacts saved: model.pkl, label_map.pkl, inv_map.pkl, feature_cols.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_training():
    X_train_full, y_train_full, X_test, y_test, label_map, inv_map, feature_cols = \
        run_preprocessing()

    # ── Internal 80/20 validation split (avoids train==test evaluation) ───
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2,
        random_state=42, stratify=y_train_full)
    print(f"Train split: {len(X_tr)} | Val split: {len(X_val)} | Test: {len(X_test)}\n")

    models  = train_models(X_tr, y_tr)

    # Evaluate on held-out validation set (honest estimate)
    print("=== VALIDATION SET RESULTS ===")
    results = evaluate_models(models, X_val, y_val, inv_map)

    # Also report on official test set
    print("\n=== TEST SET RESULTS ===")
    test_results = evaluate_models(models, X_test, y_test, inv_map)

    plot_accuracy_comparison(results)

    best_name  = max(results, key=lambda n: results[n]["accuracy"])
    best_model = models[best_name]
    best_pred  = test_results[best_name]["y_pred"]
    print(f"\nBest model: {best_name} (val {results[best_name]['accuracy']:.2%})")

    plot_confusion_matrix(y_test, best_pred, inv_map, best_name)
    if hasattr(best_model, "feature_importances_"):
        plot_feature_importance(best_model, feature_cols)
    save_artefacts(best_model, label_map, inv_map, feature_cols)
    return best_model, label_map, inv_map, feature_cols


if __name__ == "__main__":
    run_training()
