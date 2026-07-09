# 🌸 Iris Flower Classification — Machine Learning Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-3.x-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

*A complete supervised machine learning pipeline for classifying Iris flower species using KNN, Decision Tree, and Logistic Regression — with 5-fold cross-validation.*

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Results](#-results)
- [Visualizations](#-visualizations)
- [Key Findings](#-key-findings)
- [Author](#-author)

---

## 🔍 Overview

This project implements a full end-to-end machine learning classification pipeline on the classic **Iris dataset**. It covers every step of the ML workflow — from data exploration and preprocessing, through model training and evaluation, to rich visual analysis including decision boundary plots.

**Goals:**
- Load, inspect and explore the Iris dataset
- Split data into training and test sets (80/20)
- Train three classifiers: KNN, Decision Tree, and Logistic Regression
- Evaluate each model using accuracy scores, classification reports, and confusion matrices
- Apply **5-fold stratified cross-validation** and compare with single-split results
- Visualize results and interpret model behaviour

---

## 📦 Dataset

The **Iris dataset** is a classic multiclass classification benchmark included in `sklearn.datasets`.

| Property        | Details                                      |
|-----------------|----------------------------------------------|
| **Source**      | `sklearn.datasets.load_iris()`               |
| **Samples**     | 150 (50 per class)                           |
| **Features**    | 4 numerical features                         |
| **Target**      | 3 species (Setosa, Versicolor, Virginica)    |
| **Missing data**| None                                         |

### Features

| Feature             | Unit | Description                     |
|---------------------|------|---------------------------------|
| Sepal Length        | cm   | Length of the sepal             |
| Sepal Width         | cm   | Width of the sepal              |
| Petal Length        | cm   | Length of the petal             |
| Petal Width         | cm   | Width of the petal              |

### Class Distribution

```
setosa        50 samples
versicolor    50 samples
virginica     50 samples
```
> Perfectly balanced — no class imbalance handling required.

---

## 🗂️ Project Structure

```
IrisProject/
├── iris_classification.py     # Main ML pipeline script
├── README.md                  # Project documentation
├── .gitignore                 # Git ignore rules
│
└── outputs/                   # Generated plots (after running the script)
    ├── pairplot.png               # Feature pair plot by species
    ├── correlation_heatmap.png    # Feature correlation matrix
    ├── confusion_matrices.png     # Confusion matrices for all 3 models
    ├── accuracy_comparison.png    # Model accuracy bar chart
    ├── decision_tree.png          # Decision tree structure diagram
    ├── decision_boundaries.png    # 2D decision boundaries (bonus)
    ├── knn_k_selection.png        # KNN hyperparameter tuning plot
    └── cross_validation.png       # CV vs single split comparison chart
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Clone the repository

```bash
git clone https://github.com/yassinegab/IrisProject.git
cd IrisProject
```

### Install dependencies

```bash
pip install scikit-learn pandas matplotlib seaborn numpy
```

---

## 🚀 Usage

Run the complete pipeline with a single command:

```bash
python iris_classification.py
```

The script will:
1. Load and explore the dataset
2. Generate exploratory visualizations
3. Split data and train all three models
4. Print accuracy scores and classification reports to the console
5. Save all plots as `.png` files in the working directory

**Expected output:**

```
============================================================
  STEP 1 -- LOAD & EXPLORE THE DATASET
============================================================
  Shape          : (150, 5)
  Feature names  : ['sepal length (cm)', ...]
  Missing values : 0

  ...

============================================================
  FINAL SUMMARY
============================================================
  Model                              Accuracy
  --------------------------------------------
  K-Nearest Neighbors (k=5)            93.33%
  Decision Tree                        93.33%
  Logistic Regression                  93.33%
```

---

## 🧪 Methodology

### 1. Data Exploration
- Loaded the Iris dataset via `sklearn.datasets.load_iris()`
- Created a `pandas` DataFrame for structured inspection
- Printed shape, feature names, class distribution, summary statistics
- Generated a pairplot and correlation heatmap

### 2. Data Splitting
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```
- 80% training / 20% test split
- Stratified to preserve class proportions in both sets
- `random_state=42` ensures reproducibility

### 3. Feature Scaling
```python
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
```
All features were standardized (zero mean, unit variance). This is critical for distance-based models like KNN.

### 4. Models Trained

| Model                  | Key Hyperparameters            |
|------------------------|-------------------------------|
| K-Nearest Neighbors    | `k=5` (optimal found via search) |
| Decision Tree          | `max_depth=4`, `random_state=42` |
| Logistic Regression    | `max_iter=500`, `random_state=42` |

### 5. Evaluation Metrics
- **Accuracy score** — overall percentage of correct predictions
- **Classification report** — per-class precision, recall, F1-score
- **Confusion matrix** — breakdown of predictions per class

### 6. Cross-Validation (5-Fold Stratified)
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    model),
])
scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
print(f"CV mean: {scores.mean():.4f} ± {scores.std():.4f}")
```
- Each model is wrapped in a **Pipeline** with a `StandardScaler` — this ensures the scaler is re-fitted on each training fold, **preventing data leakage**
- `StratifiedKFold` preserves the 50/50/50 class balance in every fold
- Results are compared against the single 80/20 split to assess consistency

---

## 📊 Results

All three models achieved **identical accuracy** of **93.33%** on the single 80/20 test split. Cross-validation reveals consistently higher mean accuracy, confirming the models generalise well.

| Model                    | Single Split | CV Mean  | CV Std  |
|--------------------------|:------------:|:--------:|:-------:|
| K-Nearest Neighbors (k=5)| **93.33%**  | **97.33%** | ±2.49% |
| Decision Tree (depth=4)  | **93.33%**  | **95.33%** | ±3.40% |
| Logistic Regression      | **93.33%**  | **95.33%** | ±4.52% |

> **Interpretation:** CV mean scores are 2–4% higher than the single split, suggesting the 80/20 split happened to include a slightly harder test fold. The small standard deviations (2–5%) indicate stable, consistent performance across all folds.

### Individual CV Fold Scores

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|:---:|:---:|:---:|:---:|:---:|
| KNN (k=5)          | 100% | 96.67% | 93.33% | 100% | 96.67% |
| Decision Tree      | 100% | 96.67% | 93.33% | 96.67% | 90% |
| Logistic Regression| 100% | 96.67% | 90%   | 100%  | 90%  |

### Confusion Matrix Breakdown

```
                Predicted
                Setosa  Versicolor  Virginica
Actual Setosa     10        0           0      ← Perfect
Actual Versicolor  0       9/10        1/0     ← 1 or 0 error
Actual Virginica   0       1/0         9/10    ← 1 or 0 error
```

- **Setosa** is perfectly classified by all models (100% precision & recall).
- **Versicolor** and **Virginica** show minor confusion with each other — expected due to overlapping feature distributions.

### 🧠 Evaluation Metrics & Interpretation

#### **Which metric is most important in this case?**
- **Macro F1-Score**: Since this dataset is perfectly balanced (equal numbers of Setosa, Versicolor, and Virginica), overall **Accuracy** is highly informative. However, **Macro F1-Score** is the most important metric because it takes the harmonic mean of precision and recall for each class individually and averages them. This ensures that any poor performance on a specific class is not masked by high performance on other classes.
- **Precision vs. Recall**: 
  - **Precision** is crucial if false positives are costly (e.g., mislabeling a common flower as a rare, protected species).
  - **Recall** is crucial if false negatives are costly (e.g., missing a toxic variant). 
  - In a standard taxonomic context with no single dominant cost, the balanced **F1-Score** represents the best overall measure.

#### **What does the confusion matrix tell you?**
1. **Linear Separability of Setosa**: All models achieve a perfect score (10/10) for Setosa. There are 0 false positives and 0 false negatives. This proves Setosa is fully distinct from the other two species.
2. **Boundary Overlap (Versicolor vs. Virginica)**: The confusion matrix exposes the limits of the models:
   - For **KNN**: 2 Virginica flowers were misclassified as Versicolor.
   - For **Logistic Regression & Decision Trees**: 1 Versicolor was misclassified as Virginica and 1 Virginica was misclassified as Versicolor.
   - This indicates that Versicolor and Virginica share overlapping feature profiles, particularly in their petal length and width boundary regions.

- All misclassifications occur at the Versicolor/Virginica boundary

---

## 🖼️ Visualizations

### Feature Pair Plot
Shows the pairwise relationship between all 4 features, colour-coded by species. Setosa is clearly linearly separable; Versicolor and Virginica overlap slightly on sepal features but are distinguishable on petal features.

### Correlation Heatmap
Reveals that **petal length** and **petal width** are highly correlated (r ≈ 0.96), and both strongly correlate with the target class — making them the most informative features.

### Confusion Matrices
Visualizes where each model succeeds and struggles. All models misclassify 2 out of 30 test samples, always between Versicolor and Virginica.

### Decision Boundaries (Bonus)
2D visualization using petal length and petal width (the two most discriminative features). All three models draw clear decision regions with sharp boundaries.

### KNN — k Selection (Bonus)
Test accuracy plotted for k = 1 to 20. The optimal k is found at k = 1–5, with accuracy stabilizing around 93–100%. k=5 offers a good balance between bias and variance.

### Cross-Validation Comparison
Two side-by-side charts:
- **Grouped bar chart** — single split accuracy vs. CV mean accuracy (with error bars showing ±1 std deviation) for all three models
- **Box plot** — distribution of the 5 fold scores per model, showing spread and consistency

---

## 🔑 Key Findings

1. **All three classifiers perform equally well** (93.33% single split, 95–97% CV mean), confirming the dataset's clean and separable structure.
2. **Cross-validation gives a more reliable estimate**: CV means are 2–4% higher than the single split, and the low standard deviations (±2–5%) confirm stable generalisation.
3. **Pipeline prevents data leakage**: wrapping the scaler and classifier in a `Pipeline` ensures preprocessing is refitted on each fold's training data — a best practice in production ML.
4. **Setosa is trivially separable** — all models classify it with 100% accuracy across every fold.
5. **Petal features are the most discriminative**: petal length and petal width explain most of the variance between species.
6. **KNN achieved the highest CV accuracy** (97.33%), slightly beating Decision Tree and Logistic Regression (both 95.33%).

---

## 📚 Dependencies

```
scikit-learn>=1.0
pandas>=1.3
numpy>=1.21
matplotlib>=3.4
seaborn>=0.11
```

---

## 👤 Author

**Yassine Gab**  
[GitHub](https://github.com/yassinegab)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it with attribution.

---

<div align="center">

*Built with scikit-learn · pandas · matplotlib · seaborn*

</div>
