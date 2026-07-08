# 🌸 Iris Flower Classification — Machine Learning Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-3.x-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

*A complete supervised machine learning pipeline for classifying Iris flower species using KNN, Decision Tree, and Logistic Regression.*

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
    └── knn_k_selection.png        # KNN hyperparameter tuning plot
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

### Bonus
- **Decision boundary visualization** using 2 features (petal length & petal width)
- **KNN hyperparameter tuning** — accuracy plotted for k = 1 to 20

---

## 📊 Results

All three models achieved **identical accuracy** of **93.33%** on the test set.

| Model                    | Test Accuracy | Notes                           |
|--------------------------|:-------------:|---------------------------------|
| K-Nearest Neighbors (k=5)| **93.33%**   | 2 misclassifications            |
| Decision Tree (depth=4)  | **93.33%**   | Interpretable structure         |
| Logistic Regression      | **93.33%**   | Strong linear separability      |

### Confusion Matrix Breakdown

```
                Predicted
                Setosa  Versicolor  Virginica
Actual Setosa     10        0           0      ← Perfect
Actual Versicolor  0       9/10        1/0     ← 1 or 0 error
Actual Virginica   0       1/0         9/10    ← 1 or 0 error
```

- **Setosa** is perfectly classified by all models (100% precision & recall)
- **Versicolor** and **Virginica** show minor confusion with each other — expected due to overlapping feature distributions
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

---

## 🔑 Key Findings

1. **All three classifiers perform equally well** (93.33%) on this dataset, reflecting the dataset's simplicity and clean structure.
2. **Setosa is trivially separable** — all models classify it with 100% accuracy.
3. **Petal features are the most discriminative**: petal length and petal width explain most of the variance between species.
4. **Feature scaling matters**: KNN and Logistic Regression both require standardization; the Decision Tree is scale-invariant but was scaled uniformly for fairness.
5. The small dataset (150 samples) means all models achieve near-peak performance — further gains would require more data or ensemble methods.

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
