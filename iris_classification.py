"""
Iris Flower Classification -- Complete ML Pipeline
===================================================
Dataset  : sklearn.datasets.load_iris()
Models   : K-Nearest Neighbors · Decision Tree · Logistic Regression (bonus)
Author   : Yassine Gab
"""

# -- 0. Imports --------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# -- Plot style ---------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#6C63FF", "#FF6584", "#43C59E"]
sns.set_palette(PALETTE)

# ============================================================
# STEP 1 -- Load & Explore the Dataset
# ============================================================
print("=" * 60)
print("  STEP 1 -- LOAD & EXPLORE THE DATASET")
print("=" * 60)

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print(f"\n  Shape          : {df.shape}")
print(f"  Feature names  : {list(iris.feature_names)}")
print(f"  Target classes : {list(iris.target_names)}")
print(f"\n  Class distribution:\n{df['species'].value_counts()}")
print(f"\n  First 5 rows:\n{df.head()}")
print(f"\n  Statistical summary:\n{df.describe().round(2)}")
print(f"\n  Missing values  : {df.isnull().sum().sum()}")


# -- Pairplot -------------------------------------------------
print("\n[Generating pairplot …]")
pair_fig = sns.pairplot(df, hue="species", diag_kind="kde",
                        palette=PALETTE, plot_kws={"alpha": 0.7})
pair_fig.figure.suptitle("Iris -- Feature Pair Plot", y=1.02, fontsize=14, fontweight="bold")
pair_fig.figure.savefig("pairplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> pairplot.png")


# -- Correlation heat-map -------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(df.drop(columns="species").corr(), annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("correlation_heatmap.png", dpi=150)
plt.close()
print("  Saved -> correlation_heatmap.png")


# ============================================================
# STEP 2 -- Split the Data
# ============================================================
print("\n" + "=" * 60)
print("  STEP 2 -- SPLIT THE DATA")
print("=" * 60)

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\n  Training samples : {X_train.shape[0]}")
print(f"  Test samples     : {X_test.shape[0]}")

# Feature scaling (important for KNN)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ============================================================
# STEP 3 -- Train Models & Evaluate
# ============================================================
print("\n" + "=" * 60)
print("  STEP 3 -- TRAIN & EVALUATE MODELS")
print("=" * 60)

models = {
    "K-Nearest Neighbors (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree":             DecisionTreeClassifier(max_depth=4, random_state=42),
    "Logistic Regression":       LogisticRegression(max_iter=500, random_state=42),
}

results = {}

for name, model in models.items():
    # KNN & LR benefit from scaled data; DT is scale-invariant but we use it uniformly
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    results[name] = {"accuracy": acc, "cm": cm, "y_pred": y_pred}

    print(f"\n-- {name}")
    print(f"   Accuracy : {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))


# -- Confusion matrices figure ---------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Confusion Matrices -- All Models", fontsize=14, fontweight="bold")

for ax, (name, res) in zip(axes, results.items()):
    disp = ConfusionMatrixDisplay(res["cm"], display_labels=iris.target_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=9)

fig.tight_layout()
fig.savefig("confusion_matrices.png", dpi=150)
plt.close()
print("\n  Saved -> confusion_matrices.png")


# -- Accuracy comparison bar chart -----------------------------
fig, ax = plt.subplots(figsize=(8, 4))
names = list(results.keys())
accs  = [v["accuracy"] * 100 for v in results.values()]
colors = PALETTE

bars = ax.barh(names, accs, color=colors, edgecolor="white", height=0.5)
ax.set_xlim(80, 102)
ax.set_xlabel("Accuracy (%)", fontsize=11)
ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold")
for bar, acc in zip(bars, accs):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{acc:.2f}%", va="center", fontsize=10, fontweight="bold")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig("accuracy_comparison.png", dpi=150)
plt.close()
print("  Saved -> accuracy_comparison.png")


# -- Decision Tree visualisation -------------------------------
dt_model = models["Decision Tree"]
fig, ax = plt.subplots(figsize=(18, 7))
plot_tree(dt_model, feature_names=iris.feature_names,
          class_names=iris.target_names, filled=True,
          rounded=True, fontsize=9, ax=ax)
ax.set_title("Decision Tree Structure (max_depth=4)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("decision_tree.png", dpi=150)
plt.close()
print("  Saved -> decision_tree.png")


# ============================================================
# STEP 4 -- BONUS: Decision Boundary (2D -- petal features)
# ============================================================
print("\n" + "=" * 60)
print("  STEP 4 -- BONUS: 2-D DECISION BOUNDARIES")
print("=" * 60)

feature_idx = [2, 3]  # petal length & petal width
feat_labels  = [iris.feature_names[i] for i in feature_idx]

X2 = iris.data[:, feature_idx]
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y, test_size=0.20, random_state=42, stratify=y
)
sc2 = StandardScaler()
X2_train_sc = sc2.fit_transform(X2_train)
X2_test_sc  = sc2.transform(X2_test)

boundary_models = {
    "KNN (k=5)":           KNeighborsClassifier(n_neighbors=5),
    "Decision Tree":       DecisionTreeClassifier(max_depth=4, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
fig.suptitle("Decision Boundaries (Petal Length vs Petal Width)", fontsize=14, fontweight="bold")

h = 0.02
for ax, (name, bm) in zip(axes, boundary_models.items()):
    bm.fit(X2_train_sc, y2_train)

    x_min, x_max = X2_test_sc[:, 0].min() - 1, X2_test_sc[:, 0].max() + 1
    y_min, y_max = X2_test_sc[:, 1].min() - 1, X2_test_sc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = bm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3,
                colors=PALETTE, levels=[-0.5, 0.5, 1.5, 2.5])
    for cls, color in enumerate(PALETTE):
        mask = y2_test == cls
        ax.scatter(X2_test_sc[mask, 0], X2_test_sc[mask, 1],
                   c=color, edgecolors="k", s=60, label=iris.target_names[cls])

    acc2 = accuracy_score(y2_test, bm.predict(X2_test_sc))
    ax.set_title(f"{name}\nAcc = {acc2*100:.1f}%", fontsize=11, fontweight="bold")
    ax.set_xlabel("Petal Length (scaled)", fontsize=9)
    ax.set_ylabel("Petal Width (scaled)", fontsize=9)
    ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("decision_boundaries.png", dpi=150)
plt.close()
print("  Saved -> decision_boundaries.png")


# -- KNN -- choosing the best k ---------------------------------
print("\n[Finding optimal k for KNN …]")
k_range = range(1, 21)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_sc, y_train)
    k_scores.append(accuracy_score(y_test, knn.predict(X_test_sc)))

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(k_range, k_scores, marker="o", color="#6C63FF", linewidth=2)
ax.axvline(x=k_scores.index(max(k_scores)) + 1, color="#FF6584",
           linestyle="--", label=f"Best k={k_scores.index(max(k_scores))+1}")
ax.set_xlabel("k (Number of Neighbors)", fontsize=11)
ax.set_ylabel("Test Accuracy", fontsize=11)
ax.set_title("KNN -- Accuracy vs. k", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig("knn_k_selection.png", dpi=150)
plt.close()
print("  Saved -> knn_k_selection.png")


# -- Final summary ---------------------------------------------
print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"\n  {'Model':<32} {'Accuracy':>10}")
print("  " + "-" * 44)
for name, res in results.items():
    print(f"  {name:<32} {res['accuracy']*100:>9.2f}%")
print("\n  All plots saved successfully!")
print("=" * 60)
