"""
preprocessing.py
================
Handles all data loading, cleaning, and preprocessing steps.
Each step is clearly documented with inline comments.

KEY ADDITION: Weighted feature vectors
  Instead of raw 0/1 binary columns, each symptom column is multiplied
  by its medical importance weight (from medical_rules.py).
  This means common symptoms (headache=1) contribute less signal than
  specific ones (weakness_of_one_body_side=4), making the model
  naturally prefer specific symptom patterns for severe diseases.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

from medical_rules import get_weight

# ── Output directory for saved plots ──────────────────────────────────────────
PLOT_DIR = os.path.join("static", "images")
os.makedirs(PLOT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_data(train_path="dataset/Training.csv",
              test_path="dataset/Testing.csv"):
    """Load training and testing CSVs."""
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    # Drop the unnamed trailing column that appears in Training.csv
    train = train.loc[:, ~train.columns.str.contains('^Unnamed')]
    test  = test.loc[:,  ~test.columns.str.contains('^Unnamed')]

    print("=== RAW DATA SHAPE ===")
    print(f"Training : {train.shape}")
    print(f"Testing  : {test.shape}\n")
    return train, test


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLEAN COLUMN NAMES
# ══════════════════════════════════════════════════════════════════════════════
def clean_columns(df):
    """
    Strip whitespace from column names.
    Some columns like 'spotting_ urination' have a leading space.
    """
    df.columns = df.columns.str.strip()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. MISSING VALUE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def handle_missing(df, label="train"):
    """
    Check for missing values and fill numeric columns with 0.
    The dataset is mostly complete, but we handle it defensively.
    """
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"=== MISSING VALUES ({label}) ===")
    print(missing[missing > 0] if total_missing > 0 else "No missing values found.")
    print()

    # Fill any numeric NaN with 0 (symptom absent)
    symptom_cols = [c for c in df.columns if c != "prognosis"]
    df[symptom_cols] = df[symptom_cols].fillna(0)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. REMOVE DUPLICATES
# ══════════════════════════════════════════════════════════════════════════════
def remove_duplicates(df, label="train"):
    """Drop exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"=== DUPLICATES ({label}) ===")
    print(f"Removed {before - after} duplicate rows. Remaining: {after}\n")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. ENCODE TARGET LABEL
# ══════════════════════════════════════════════════════════════════════════════
def encode_labels(train, test):
    """
    Label-encode the 'prognosis' column, then apply symptom importance weights.
    Weighted features: each binary 0/1 column is multiplied by its medical
    weight so specific symptoms (weight=4-5) dominate over generic ones (weight=1).
    """
    all_labels = pd.concat([train["prognosis"], test["prognosis"]]).unique()
    label_map  = {label: idx for idx, label in enumerate(sorted(all_labels))}
    inv_map    = {v: k for k, v in label_map.items()}

    train["label"] = train["prognosis"].map(label_map)
    test["label"]  = test["prognosis"].map(label_map)

    feature_cols = [c for c in train.columns if c not in ("prognosis", "label")]

    X_train_raw = train[feature_cols].astype(int)
    y_train     = train["label"].astype(int)
    X_test_raw  = test[feature_cols].astype(int)
    y_test      = test["label"].astype(int)

    # Apply symptom importance weights
    # headache (weight=1) contributes 1x; weakness_of_one_body_side (weight=4) contributes 4x
    weight_vector = np.array([get_weight(c) for c in feature_cols], dtype=float)
    X_train = X_train_raw * weight_vector
    X_test  = X_test_raw  * weight_vector

    print(f"=== LABEL ENCODING + WEIGHTING ===")
    print(f"Total unique diseases : {len(label_map)}")
    print(f"Feature columns       : {len(feature_cols)}")
    print(f"Weight range          : {int(weight_vector.min())}–{int(weight_vector.max())}\n")
    return X_train, y_train, X_test, y_test, label_map, inv_map, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLASS IMBALANCE CHECK
# ══════════════════════════════════════════════════════════════════════════════
def check_class_balance(y_train, inv_map):
    """Print class distribution to detect imbalance."""
    counts = pd.Series(y_train).value_counts().sort_index()
    print("=== CLASS DISTRIBUTION (top 10) ===")
    for idx, cnt in counts.head(10).items():
        print(f"  {inv_map[idx]:<35} : {cnt}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 7. VISUALISATIONS  (all plots are clean, focused, and readable)
# ══════════════════════════════════════════════════════════════════════════════

# Shared style defaults
TITLE_SIZE  = 15
LABEL_SIZE  = 12
TICK_SIZE   = 10
DPI         = 120

def _save(fig, filename):
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_missing_summary(df, filename="missing_heatmap.png"):
    """
    Instead of a useless all-purple heatmap, show a clean bar chart of
    missing value counts. If none exist, produce an informative 'No missing
    values' graphic so the page still has something to display.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    if missing.empty:
        ax.text(0.5, 0.5, "✔  No Missing Values Found\nin the Dataset",
                ha="center", va="center", fontsize=18,
                color="#38a169", fontweight="bold",
                transform=ax.transAxes)
        ax.set_axis_off()
        ax.set_title("Missing Values Check", fontsize=TITLE_SIZE, pad=14)
    else:
        sns.barplot(x=missing.values, y=missing.index,
                    palette="Reds_r", ax=ax)
        ax.set_title("Missing Values per Column", fontsize=TITLE_SIZE)
        ax.set_xlabel("Missing Count", fontsize=LABEL_SIZE)
        ax.tick_params(axis="y", labelsize=TICK_SIZE)
    plt.tight_layout()
    return _save(fig, filename)


def plot_disease_distribution(train_df, top_n=15, filename="disease_distribution.png"):
    """
    Horizontal bar chart — top N diseases only so labels are readable.
    Sorted by count, colour-coded by frequency.
    """
    counts = train_df["prognosis"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color=sns.color_palette("husl", top_n)[::-1])
    # Value labels on bars
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=TICK_SIZE, fontweight="bold")
    ax.set_title(f"Top {top_n} Diseases by Frequency", fontsize=TITLE_SIZE, pad=12)
    ax.set_xlabel("Number of Training Samples", fontsize=LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return _save(fig, filename)


def plot_symptom_frequency(X_train, feature_cols, top_n=20,
                           filename="symptom_frequency.png"):
    """
    Top-N symptoms by occurrence — clean horizontal bars with value labels.
    Raw binary counts (before weighting) so frequency is intuitive.
    """
    # Use raw presence counts (divide by weight to get binary back if weighted)
    freq = (X_train > 0).sum().sort_values(ascending=False).head(top_n)
    labels = [c.replace("_", " ").title() for c in freq.index]

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = sns.color_palette("mako_r", top_n)
    bars = ax.barh(labels[::-1], freq.values[::-1], color=colors[::-1])
    for bar, val in zip(bars, freq.values[::-1]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=TICK_SIZE, fontweight="bold")
    ax.set_title(f"Top {top_n} Most Frequent Symptoms", fontsize=TITLE_SIZE, pad=12)
    ax.set_xlabel("Frequency (number of records)", fontsize=LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return _save(fig, filename)


def plot_correlation_heatmap(X_train, top_n=15, filename="correlation_heatmap.png"):
    """
    Correlation heatmap of the top-N most variable features ONLY.
    Reduced from 132 → 15 so every cell is readable.
    Annotated with correlation values.
    """
    # Pick top_n by variance (most informative features)
    top_cols = X_train.var().sort_values(ascending=False).head(top_n).index
    corr     = (X_train[top_cols] > 0).astype(int).corr()

    # Shorten labels for readability
    short_labels = [c.replace("_", " ").title() for c in top_cols]

    fig, ax = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle mask
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", linewidths=0.5, linecolor="#e2e8f0",
                xticklabels=short_labels, yticklabels=short_labels,
                annot_kws={"size": 9}, ax=ax,
                vmin=-1, vmax=1, center=0)
    ax.set_title(f"Symptom Correlation Heatmap (Top {top_n} Features)",
                 fontsize=TITLE_SIZE, pad=14)
    ax.tick_params(axis="x", labelsize=9, rotation=45)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    plt.tight_layout()
    return _save(fig, filename)


def plot_disease_pie(train_df, top_n=10, filename="disease_pie.png"):
    """
    Donut-style pie chart of top-N diseases.
    Cleaner than a full pie — centre shows total sample count.
    """
    counts = train_df["prognosis"].value_counts().head(top_n)
    colors = sns.color_palette("pastel", top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index,
        autopct="%1.1f%%", startangle=140,
        colors=colors, pctdistance=0.82,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2}
    )
    for t in texts:     t.set_fontsize(10)
    for a in autotexts: a.set_fontsize(9); a.set_fontweight("bold")

    # Centre label
    ax.text(0, 0, f"Top {top_n}\nDiseases", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#2d3748")
    ax.set_title(f"Disease Share — Top {top_n}", fontsize=TITLE_SIZE, pad=14)
    plt.tight_layout()
    return _save(fig, filename)


def plot_class_balance(y_train, inv_map, filename="class_balance.png"):
    """
    Horizontal bar chart showing samples per disease class.
    Highlights whether the dataset is balanced or skewed.
    """
    counts = pd.Series(y_train).value_counts().sort_values(ascending=True)
    labels = [inv_map[i].replace("(", "").replace(")", "")[:30] for i in counts.index]
    colors = ["#e53e3e" if v < 5 else "#38a169" for v in counts.values]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(labels, counts.values, color=colors)
    ax.axvline(counts.mean(), color="#2b6cb0", linestyle="--",
               linewidth=1.5, label=f"Mean = {counts.mean():.1f}")
    ax.set_title("Class Balance — Samples per Disease", fontsize=TITLE_SIZE, pad=12)
    ax.set_xlabel("Number of Samples", fontsize=LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return _save(fig, filename)


# ══════════════════════════════════════════════════════════════════════════════
# 8. FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_preprocessing():
    """Execute the complete preprocessing pipeline and return all artefacts."""
    train, test = load_data()

    train = clean_columns(train)
    test  = clean_columns(test)

    train = handle_missing(train, "train")
    test  = handle_missing(test,  "test")

    # Missing values summary (replaces useless all-purple heatmap)
    plot_missing_summary(train, "missing_heatmap.png")

    train = remove_duplicates(train, "train")
    test  = remove_duplicates(test,  "test")

    X_train, y_train, X_test, y_test, label_map, inv_map, feature_cols = \
        encode_labels(train, test)

    check_class_balance(y_train, inv_map)

    # All visualisation plots — focused, readable, meaningful
    plot_disease_distribution(train, top_n=15)
    plot_symptom_frequency(X_train, feature_cols, top_n=20)
    plot_correlation_heatmap(X_train, top_n=15)
    plot_disease_pie(train, top_n=10)
    plot_class_balance(y_train, inv_map)

    print("=== PREPROCESSING COMPLETE ===\n")
    return X_train, y_train, X_test, y_test, label_map, inv_map, feature_cols


if __name__ == "__main__":
    run_preprocessing()
