# 🌠 Predicting Asteroid Hazard Potential: A Machine Learning Approach

> **Queen's University** | NASA NeoWs Dataset | Binary Classification Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green?logo=xgboost)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Overview

This project implements a **full end-to-end machine learning pipeline** to classify Near-Earth Objects (NEOs) as **Potentially Hazardous Asteroids (PHAs)** using NASA's NeoWs dataset. The pipeline covers everything from raw data exploration and preprocessing to ensemble modeling, threshold tuning, and model serialization.

The core challenge is a **severe class imbalance** problem where hazardous asteroids represent a small minority of the dataset — making recall the primary optimization metric, since missing a real threat has far greater consequences than a false alarm.

---

## 🎯 Objectives

| # | Goal |
|---|------|
| 1 | Build a benchmarked binary classifier for PHA screening |
| 2 | Identify the most predictive orbital and physical features via feature importance analysis |
| 3 | Handle class imbalance using SMOTE oversampling |
| 4 | Prevent data leakage by excluding definitional features (MOID, Absolute Magnitude) from training |
| 5 | Evaluate ensemble methods (Soft Voting, Stacking) against individual models |

---

## 📂 Dataset

- **Source:** [NASA Asteroids Classification — Kaggle](https://www.kaggle.com/datasets/shrutimehta/nasa-asteroids-classification)
- **Origin:** NASA's Near Earth Object Web Service (NeoWS) API
- **Size:** ~4,700 rows × 40 columns
- **Target Variable:** `Hazardous` (binary: True / False)

### Feature Categories

| Category | Examples |
|---|---|
| **Physical** | Absolute Magnitude, Estimated Diameter (km, m, miles, feet) |
| **Close-Approach** | Relative Velocity (km/s), Miss Distance (AU, km), Close Approach Date |
| **Orbital** | Eccentricity, Semi-Major Axis, Inclination, Orbital Period, MOID, Perihelion Distance |
| **Administrative** | Name, Neo Reference ID, Orbit ID, Orbit Determination Date |

> ⚠️ **Leakage Prevention:** `Minimum Orbit Intersection (MOID)` and `Absolute Magnitude` were intentionally **excluded** from training features. NASA's official PHA definition is directly derived from these two values (MOID < 0.05 AU **and** H < 22), so including them would cause the model to trivially memorize the label definition rather than learn generalizable patterns.

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Feature distribution plots (Hazardous vs. Non-Hazardous)
- Boxplots for outlier detection
- Correlation heatmap of numerical features
- Class imbalance visualization (bar + pie charts)
- MOID vs. Absolute Magnitude scatter plot with NASA PHA boundary overlay

### 2. Sanity Check — Rule-Based Baseline
Before training, a rule-based baseline was established using the official NASA PHA criteria. Models scoring at or near this baseline were considered to have learned the label definition, not a generalizable signal.

### 3. Data Preprocessing
- Removal of redundant/duplicate columns (diameter in different units, velocity in different units)
- Dropping identifier and date columns
- Removal of duplicate rows
- Target variable encoded as binary (0 / 1)
- Train / Validation / Test split

### 4. Model Training
Each model was wrapped in an **imbalanced-learn Pipeline** containing:
- `StandardScaler` — feature normalization
- `SMOTE` — synthetic minority oversampling to handle class imbalance
- Classifier — tuned via `RandomizedSearchCV` with `StratifiedKFold` (5-fold CV)

| Model | Key Hyperparameters Tuned |
|---|---|
| Logistic Regression | `C`, `solver`, `class_weight`, SMOTE `k_neighbors` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split`, `max_features` |
| XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `scale_pos_weight` |
| SVM | `C`, `kernel`, `gamma`, `class_weight` |

### 5. Ensemble Methods
- **Soft Voting:** Averages predicted probabilities across all four base models
- **Stacking:** Out-of-fold (OOF) base model predictions used to train a Logistic Regression meta-learner

### 6. Overfitting Analysis
- Train / Validation / Test recall compared for each model
- Learning curves plotted for all four base models
- Gap threshold of **0.05** used to flag overfitting

### 7. Threshold Tuning
Precision-Recall curves were analyzed to find the optimal decision threshold targeting **≥ 0.95 Recall** — prioritizing sensitivity for hazard screening.

---

## 📊 Evaluation Metrics

Given the class imbalance and domain criticality, the primary metric is **Recall** (minimizing false negatives). The full evaluation suite includes:

- Accuracy, Precision, **Recall**, F1-Score, **ROC-AUC**
- Confusion Matrices for all models
- ROC Curves (all models overlaid)
- Precision-Recall Curves with threshold analysis

---

## 🛠️ Tech Stack

| Library | Usage |
|---|---|
| `pandas`, `numpy` | Data manipulation and numerical operations |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | ML models, preprocessing, evaluation, pipelines |
| `xgboost` | Gradient boosting classifier |
| `imbalanced-learn` | SMOTE oversampling, imbalanced pipeline |
| `joblib` | Model serialization |

---

## 📁 Project Structure

```
asteroid-hazard-prediction/
│
├── Asteroid_Hazard_Potential_Final.ipynb   # Main notebook (full pipeline)
├── nasa.csv                                # Raw dataset (from Kaggle)
│
├── models/
│   ├── lr_model.pkl                        # Saved Logistic Regression model
│   ├── rf_model.pkl                        # Saved Random Forest model
│   ├── xgb_model.pkl                       # Saved XGBoost model
│   ├── svm_model.pkl                       # Saved SVM model
│   └── stack_meta.pkl                      # Saved Stacking meta-learner
│
├── results/
│   └── model_comparison_results.csv        # Final metrics for all models
│
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib
```

### Run the Notebook

1. Clone the repository:
```bash
git clone https://github.com/your-username/asteroid-hazard-prediction.git
cd asteroid-hazard-prediction
```

2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/shrutimehta/nasa-asteroids-classification) and place `nasa.csv` in the project root.

3. Open and run the notebook:
```bash
jupyter notebook Asteroid_Hazard_Potential_Final.ipynb
```

### Load a Saved Model

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('models/rf_model.pkl')

# Predict on new data
# X_new = pd.DataFrame(...)  # must match training feature schema
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]
```

---

## 💡 Key Design Decisions

| Decision | Rationale |
|---|---|
| Drop MOID & Absolute Magnitude | These directly define the NASA PHA label — including them causes data leakage |
| Use Recall as primary metric | A missed hazardous asteroid is far more costly than a false alarm |
| SMOTE inside pipeline | Prevents data leakage by only oversampling within each CV fold |
| OOF predictions for stacking | Ensures the meta-learner is never trained on predictions from data it was already fit on |
| Threshold tuning | Fixed 0.5 threshold is suboptimal for imbalanced problems; precision-recall trade-off is analyzed explicitly |

---


## 🙏 Acknowledgements

- [NASA Jet Propulsion Laboratory](https://www.jpl.nasa.gov/) for the NeoWs API
- [Kaggle — NASA Asteroids Classification Dataset](https://www.kaggle.com/datasets/shrutimehta/nasa-asteroids-classification)
- Scikit-learn, XGBoost, and imbalanced-learn open-source communities
