# Customer Review Prediction
### An End-to-End Machine Learning Case Study on the Olist Brazilian E-Commerce Dataset

---

## Project Overview

This project builds a complete, production-oriented machine learning pipeline to predict whether an Olist e-commerce order will receive a **positive customer review** (score 4–5) or a **negative customer review** (score 1–2).

The project is designed as a learning resource and portfolio case study.  It emphasises clean software engineering, statistical reasoning, feature engineering, and business interpretation — rather than simply maximising model accuracy.

---

## Business Problem

Customer reviews are the lifeblood of an e-commerce marketplace.  A single negative review can deter future buyers, while consistently positive reviews build brand trust and drive growth.

**The core question:**

> Can we predict, at the time of order placement, whether a customer will leave a positive or negative review?

If yes, Olist can:
- Proactively intervene on high-risk orders (e.g., send extra tracking updates)
- Identify and address systemic quality issues with sellers or logistics
- Prioritise customer support resources towards orders most likely to go wrong

---

## Dataset Description

| Table | Description | Rows (approx.) |
|---|---|---|
| `olist_orders_dataset.csv` | One row per order with timestamps | ~100,000 |
| `olist_customers_dataset.csv` | Customer information including state | ~100,000 |
| `olist_order_items_dataset.csv` | Item-level data: products, prices, freight | ~112,000 |
| `olist_products_dataset.csv` | Product attributes: category, weight, dimensions | ~33,000 |
| `olist_order_payments_dataset.csv` | Payment method and installments per order | ~104,000 |
| `olist_order_reviews_dataset.csv` | Review scores and comments | ~100,000 |
| `olist_sellers_dataset.csv` | Seller location information | ~3,000 |
| `product_category_name_translation.csv` | Portuguese → English category translation | ~71 |

**Target Variable:**

```
positive_review = 1  if review_score >= 4  (clear satisfaction)
positive_review = 0  if review_score <= 2  (clear dissatisfaction)
review_score == 3    → EXCLUDED (neutral; ambiguous for binary classification)
```

**Class Balance (approximate):**

| Class | Proportion |
|---|---|
| Positive (1) | ~78% |
| Negative (0) | ~22% |

---

## Folder Structure

```
customer_review_prediction/
│
├── data/
│   └── analytical/
│       ├── analytical_dataset.csv       ← produced by notebook 01
│       └── feature_dataset.csv          ← produced by notebook 02
│
├── sql/
│   ├── analytical_queries.sql           ← SQL documentation for dataset joins
│   └── feature_queries.sql              ← SQL documentation for feature logic
│
├── notebooks/
│   ├── 01_create_analytical_dataset.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_eda_hypothesis_testing.ipynb
│   └── 04_customer_review_prediction.ipynb
│
├── outputs/
│   ├── figures/                         ← all visualisations
│   ├── metrics/                         ← model comparison and coefficients
│   ├── tables/                          ← hypothesis test results
│   └── models/                          ← saved best model
│
└── README.md
```

---

## Project Pipeline

```
Raw Cleaned CSVs
       │
       ▼
01_create_analytical_dataset.ipynb
  ├── Join 6 tables into one row-per-order dataset
  ├── Create binary target variable
  └── Save: data/analytical/analytical_dataset.csv
       │
       ▼
02_feature_engineering.ipynb
  ├── Time features (month, weekday, hour)
  ├── Order features (value, freight, items)
  ├── Product features (category, weight, volume)
  ├── Geographic features (customer/seller state, same_state)
  ├── Payment features (type, installments)
  ├── Customer historical features (leakage-safe)
  ├── Seller historical features (leakage-safe)
  └── Save: data/analytical/feature_dataset.csv
       │
       ▼
03_eda_hypothesis_testing.ipynb
  ├── 6 business visualisations
  ├── 5 Welch's t-tests
  └── Save: outputs/tables/hypothesis_test_summary.csv
       │
       ▼
04_customer_review_prediction.ipynb
  ├── Train/test split (stratified 80/20)
  ├── Preprocessing pipeline (imputation + scaling + encoding)
  ├── Logistic Regression
  ├── Decision Tree
  ├── Random Forest
  ├── Model comparison table
  ├── ROC curves
  ├── Feature importance
  └── Save: outputs/models/, outputs/metrics/
```

---

## Statistical Analysis

Five **Welch's two-sample t-tests** were performed comparing feature distributions between satisfied and dissatisfied customers.

> **Why Welch's t-test?**  It tests whether two groups have significantly different means without assuming equal variances — a more realistic assumption for real-world business data.

| # | Feature | Business Question |
|---|---|---|
| 1 | `total_order_value` | Do customers leaving positive reviews spend more? |
| 2 | `freight_value` | Does freight cost differ between satisfied and dissatisfied customers? |
| 3 | `total_product_weight_g` | Do heavier shipments receive different customer reviews? |
| 4 | `number_of_items` | Do larger orders receive different satisfaction? |
| 5 | `seller_average_review` | Do better-reviewed sellers lead to better customer reviews? |

Full results — including t-statistics, p-values, and 95% confidence intervals — are in `outputs/tables/hypothesis_test_summary.csv` and **Notebook 03**.

---

## Feature Engineering

### Leakage-Safe Historical Features

A key engineering challenge is computing **historical features without data leakage**.

For each order, the customer's and seller's historical statistics are computed using **only their previous orders** — sorted chronologically with `cumsum().shift(1)` — so that the current order's outcome is never used in its own features.

| Group | Feature | Description |
|---|---|---|
| **Customer** | `previous_order_count` | Orders placed before this one |
| | `previous_total_spent` | Total amount spent on prior orders |
| | `previous_average_review` | Avg review score given on prior orders |
| **Seller** | `seller_average_review` | Avg review score received on prior orders |
| | `seller_total_orders` | Prior orders fulfilled by this seller |
| | `seller_late_delivery_rate` | Fraction of prior orders delivered late |
| **Order** | `total_order_value` | Sum of item prices |
| | `freight_value` | Total freight charged |
| | `freight_ratio` | Freight / (items + freight) |
| | `number_of_items` | Item count |
| | `seller_count` | Distinct sellers in the order |
| **Product** | `dominant_product_category` | Category with highest revenue in order |
| | `number_of_categories` | Distinct categories in the order |
| | `total_product_weight_g` | Sum of all item weights |
| | `total_product_volume_cm3` | Sum of all item volumes (L×H×W) |
| **Geographic** | `customer_state` | Brazilian state of customer |
| | `seller_state` | Brazilian state of primary seller |
| | `same_state` | 1 if customer and seller are in same state |
| **Payment** | `payment_type` | Primary payment method |
| | `payment_installments` | Number of payment installments |
| **Time** | `purchase_month` | Month of purchase (1–12) |
| | `purchase_weekday` | Day of week (0=Mon, 6=Sun) |
| | `purchase_hour` | Hour of day (0–23) |

---

## Machine Learning Models

| Model | Type | Key Hyperparameters |
|---|---|---|
| **Logistic Regression** | Linear classifier | `class_weight='balanced'`, `C=1.0`, `max_iter=1000` |
| **Decision Tree** | Non-linear tree | `max_depth=10`, `min_samples_leaf=50`, `class_weight='balanced'` |
| **Random Forest** | Tree ensemble | `n_estimators=200`, `max_depth=15`, `min_samples_leaf=30`, `class_weight='balanced'` |

All models use `class_weight='balanced'` to handle the ~78%/22% class imbalance.

**Preprocessing (inside sklearn Pipeline):**

- Numeric features: `SimpleImputer(median)` → `StandardScaler()`
- Categorical features: `SimpleImputer('unknown')` → `OneHotEncoder(handle_unknown='ignore')`

---

## Results

> Run the notebooks to populate this section with your actual results.  
> The comparison table will be saved to `outputs/metrics/model_comparison.csv`.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.6772 | 0.8880 | 0.7046 | 0.7857 | 0.6665 |
| Decision Tree | 0.7150 | 0.8843 | 0.7601 | 0.8175 | 0.6597 |
| Random Forest | 0.7311 | 0.8894 | 0.7764 | 0.8291 | 0.6906 |

**Baseline (always predict positive):**  ~78% accuracy, 0.0 recall for negatives.

---

## Business Insights

1. **Seller quality is the strongest lever.**  The seller's historical average review score is among the most important predictors.  Olist should enforce seller performance standards and suspend or remove persistently poor performers.

2. **Freight costs matter.**  High freight values and high freight ratios are associated with negative reviews.  Olist could negotiate better logistics rates, especially for cross-state deliveries.

3. **Late deliveries destroy satisfaction.**  The seller's late delivery rate is a meaningful predictor.  Monitoring seller SLA compliance and penalising chronic delays would directly improve customer experience.

4. **Most customers are first-time buyers.**  This makes the first purchase experience critically important for long-term platform retention.

5. **Same-state orders perform better.**  When the customer and seller are in the same Brazilian state, delivery is typically faster and cheaper.  Promoting local seller matching could improve satisfaction.

---

## Future Work

| Extension | Value |
|---|---|
| SHAP (SHapley Additive exPlanations) | Individual prediction explainability |
| XGBoost / LightGBM | Potentially better performance |
| Hyperparameter tuning (Optuna) | Optimise model performance |
| Temporal train/test split | More realistic production evaluation |
| Build a SQL-based CLI | Quick analytics and ad-hoc querying from the terminal |
| Seller performance dashboard | Business tool for Olist operations team |
| Review text sentiment analysis | Additional NLP features from review comments |

---

## How to Run

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn joblib
```

### Execution Order

Run the notebooks in sequence:

```
1. notebooks/01_create_analytical_dataset.ipynb
2. notebooks/02_feature_engineering.ipynb
3. notebooks/03_eda_hypothesis_testing.ipynb
4. notebooks/04_customer_review_prediction.ipynb
```

Each notebook reads the output of the previous one.  All paths are relative to the notebook's directory.

---

## Technical Notes

- **No database required.**  SQL files document the join logic; Python (pandas) executes it.
- **Leakage-safe.**  All historical features use only past information via chronological sorting + `shift(1)`.
- **Reproducible.**  `RANDOM_STATE = 42` is used everywhere randomness is involved.
- **Self-contained.**  This folder is independent of all other project files.

---

*Built with Python 3.9+, pandas, numpy, scikit-learn, scipy, matplotlib, seaborn.*
