"""
Dataset exploration: compute statistics and generate figures for the thesis report.

Figures are saved to results/figures/ (git-tracked) so they can be pulled locally.

Usage:
  python scripts/explore_data.py
"""
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

PROCESSED_DIR = Path("data/processed")
FIGURES_DIR   = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {"Toys & Games": "toys", "CDs & Vinyl": "cds"}

# ── Plotting style ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
COLORS = {"Toys & Games": "#4C72B0", "CDs & Vinyl": "#DD8452"}

# ── Load datasets ──────────────────────────────────────────────────────────────
datasets = {}
for label, key in DATASETS.items():
    path = PROCESSED_DIR / key / "dataset.pkl"
    with open(path, "rb") as f:
        datasets[label] = pickle.load(f)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Summary statistics table (printed + saved as CSV)
# ══════════════════════════════════════════════════════════════════════════════
rows = []
for label, ds in datasets.items():
    train = ds.train
    # Reconstruct total interactions: train interactions + 1 val + 1 test per user
    n_val_users  = len(ds.val)
    n_test_users = len(ds.test)
    n_train_interactions = len(train)
    n_total = n_train_interactions + n_val_users + n_test_users

    user_counts = train.groupby("user_idx")["item_idx"].count()
    item_counts = train.groupby("item_idx")["user_idx"].count()
    density = n_total / (ds.n_users * ds.n_items) * 100

    rows.append({
        "Dataset":            label,
        "Users":              ds.n_users,
        "Items":              ds.n_items,
        "Interactions":       n_total,
        "Density (%)":        round(density, 4),
        "Avg inter./user":    round(n_total / ds.n_users, 2),
        "Avg inter./item":    round(n_total / ds.n_items, 2),
        "Min inter./user":    int(user_counts.min()) + 2,   # +2 for val+test
        "Max inter./user":    int(user_counts.max()) + 2,
    })

stats_df = pd.DataFrame(rows).set_index("Dataset")
print("\n=== Dataset Statistics ===")
print(stats_df.to_string())
stats_df.to_csv(FIGURES_DIR / "dataset_stats.csv")
print(f"\nSaved statistics to {FIGURES_DIR / 'dataset_stats.csv'}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. User interaction count distribution (log-log)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

for ax, (label, ds) in zip(axes, datasets.items()):
    # full interactions per user = train + 1 val + 1 test (approx: train + 2)
    user_counts = ds.train.groupby("user_idx")["item_idx"].count() + 2
    counts, bins = np.histogram(user_counts, bins=50)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.bar(bin_centers, counts, width=(bins[1] - bins[0]) * 0.9,
           color=COLORS[label], alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Interactions per user (log scale)")
    ax.set_ylabel("Number of users (log scale)")
    ax.set_title(label)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    mean_val = user_counts.mean()
    ax.axvline(mean_val, color="firebrick", linestyle="--", linewidth=1.4,
               label=f"Mean = {mean_val:.1f}")
    ax.legend(fontsize=10)

fig.suptitle("User Interaction Count Distribution", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "user_degree_distribution.pdf", bbox_inches="tight")
fig.savefig(FIGURES_DIR / "user_degree_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: user_degree_distribution")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Item interaction count distribution (log-log)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

for ax, (label, ds) in zip(axes, datasets.items()):
    item_counts = ds.train.groupby("item_idx")["user_idx"].count()
    counts, bins = np.histogram(item_counts, bins=50)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.bar(bin_centers, counts, width=(bins[1] - bins[0]) * 0.9,
           color=COLORS[label], alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Interactions per item (log scale)")
    ax.set_ylabel("Number of items (log scale)")
    ax.set_title(label)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    mean_val = item_counts.mean()
    ax.axvline(mean_val, color="firebrick", linestyle="--", linewidth=1.4,
               label=f"Mean = {mean_val:.1f}")
    ax.legend(fontsize=10)

fig.suptitle("Item Interaction Count Distribution", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "item_degree_distribution.pdf", bbox_inches="tight")
fig.savefig(FIGURES_DIR / "item_degree_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: item_degree_distribution")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Dataset size comparison (grouped bar chart)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

categories = ["Users", "Items", "Interactions"]
x = np.arange(len(categories))
width = 0.32

for i, (label, ds) in enumerate(datasets.items()):
    train = ds.train
    n_val_users  = len(ds.val)
    n_test_users = len(ds.test)
    n_total = len(train) + n_val_users + n_test_users
    values = [ds.n_users, ds.n_items, n_total]
    bars = ax.bar(x + (i - 0.5) * width, values, width,
                  label=label, color=COLORS[label], alpha=0.88, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{val:,}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylabel("Count")
ax.set_title("Dataset Scale Comparison", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
fig.tight_layout()
fig.savefig(FIGURES_DIR / "dataset_comparison.pdf", bbox_inches="tight")
fig.savefig(FIGURES_DIR / "dataset_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: dataset_comparison")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Cumulative interaction coverage (Lorenz-style curve)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, (label, ds) in zip(axes, datasets.items()):
    # Users sorted by activity (ascending)
    user_counts = ds.train.groupby("user_idx")["item_idx"].count().sort_values()
    cumulative_users = np.linspace(0, 100, len(user_counts))
    cumulative_inter = np.cumsum(user_counts.values) / user_counts.sum() * 100

    ax.plot(cumulative_users, cumulative_inter,
            color=COLORS[label], linewidth=2.2, label=label)
    ax.plot([0, 100], [0, 100], "k--", linewidth=1, alpha=0.5, label="Perfect equality")

    # Mark the point where top 20% users account for X% of interactions
    idx_80 = np.searchsorted(cumulative_users, 80)
    pct_at_80 = cumulative_inter[idx_80]
    ax.annotate(f"Bottom 80% of users\n→ {pct_at_80:.1f}% of interactions",
                xy=(80, pct_at_80), xytext=(45, pct_at_80 - 20),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=9.5, color="gray")

    ax.set_xlabel("Cumulative % of users (sorted by activity)")
    ax.set_ylabel("Cumulative % of interactions")
    ax.set_title(label)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

fig.suptitle("User Activity Inequality (Lorenz Curve)", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "lorenz_curve.pdf", bbox_inches="tight")
fig.savefig(FIGURES_DIR / "lorenz_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: lorenz_curve")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Train / val / test split sizes
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))

split_labels = ["Train", "Validation", "Test"]
x = np.arange(len(split_labels))

for i, (label, ds) in enumerate(datasets.items()):
    sizes = [len(ds.train), len(ds.val), len(ds.test)]
    bars = ax.bar(x + (i - 0.5) * width, sizes, width,
                  label=label, color=COLORS[label], alpha=0.88, edgecolor="white")
    for bar, val in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{val:,}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(split_labels, fontsize=12)
ax.set_ylabel("Number of records")
ax.set_title("Train / Validation / Test Split Sizes", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
fig.tight_layout()
fig.savefig(FIGURES_DIR / "split_sizes.pdf", bbox_inches="tight")
fig.savefig(FIGURES_DIR / "split_sizes.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: split_sizes")

print(f"\nAll figures saved to {FIGURES_DIR}/")
