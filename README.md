# Advance House Prediction

---

An end-to-end regression project on the KC House Sales dataset, covering exploratory data analysis,
geospatial visualization, Gradient Boosting modeling, and dimensionality reduction via PCA.

---

## Problem

Predict residential property sale prices based on structural, locational, and condition-based features
across King County, Washington — enabling data-driven valuation for real estate analytics.

---

## Dataset

**Source:** KC House Data (Kaggle)
**Size:** 21,613 records | 21 features
**Target variable:** `price` (sale price in USD)
**Key features:** `sqft_living`, `bedrooms`, `bathrooms`, `grade`, `waterfront`, `lat`, `long`, `yr_built`

---

## Analysis Workflow

### 1. Exploratory Data Analysis
- Bedroom distribution via bar chart
- Scatter plots examining price relationships with `sqft_living`, `lat`, `long`, `bedrooms`, and `waterfront`
- Geographic distribution of listings via joint latitude-longitude plot

### 2. Data Preprocessing
- Missing value imputation using `SimpleImputer` (mean strategy)
- Date column converted to binary feature (2014 vs. other)
- Dropped non-predictive columns (`id`, raw `date`)

### 3. Modeling — Gradient Boosting Regressor
- 90/10 train-test split
- Pipeline with mean imputation + `GradientBoostingRegressor`
- Hyperparameters: `n_estimators=400`, `max_depth=5`, `learning_rate=0.1`
- Train vs. test loss tracked across boosting iterations

### 4. Dimensionality Reduction — PCA
- Applied PCA on scaled feature matrix to analyze variance structure across 19 features

---

## Key Observations

- `sqft_living` shows the strongest positive correlation with price
- Waterfront properties command significantly higher prices
- Geographic clustering visible in lat/long joint plot — location drives valuation
- Gradient Boosting train loss converges steadily; test loss tracked across all 400 estimators

---

## Tech Stack

**Python** — `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`  
**Models** — Gradient Boosting Regressor, PCA  
**Preprocessing** — SimpleImputer, Pipeline, train-test split
