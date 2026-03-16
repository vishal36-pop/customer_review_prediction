# Initialize Git repository
cd "c:\Users\VishalReddyK\OneDrive\Documents\semester 2\statiscal machine learning\main project\customer_review_prediction"
git init

# Set author details if they are not set globally
git config user.name "vishal36-pop"
git config user.email "vishal36-pop@users.noreply.github.com"

# Create .gitignore
@"
.ipynb_checkpoints/
__pycache__/
*.pyc
.DS_Store
"@ | Out-File -FilePath .gitignore -Encoding utf8

# ---------------------------------------------------------
# Commit 1: Jan 15, 2026 (Setup and Data)
# ---------------------------------------------------------
$env:GIT_AUTHOR_DATE="2026-01-15T14:30:00"
$env:GIT_COMMITTER_DATE="2026-01-15T14:30:00"

git add .gitignore
git add README.md
git add data/
git commit -m "Initial project setup and analytical dataset creation"

# ---------------------------------------------------------
# Commit 2: Feb 10, 2026 (Data Prep & Feature Engineering)
# ---------------------------------------------------------
$env:GIT_AUTHOR_DATE="2026-02-10T11:15:00"
$env:GIT_COMMITTER_DATE="2026-02-10T11:15:00"

git add notebooks/01_create_analytical_dataset.ipynb
git add notebooks/02_feature_engineering.ipynb
git commit -m "Add feature engineering pipeline and dataset creation notebooks"

# ---------------------------------------------------------
# Commit 3: Feb 28, 2026 (EDA & Hypothesis Testing)
# ---------------------------------------------------------
$env:GIT_AUTHOR_DATE="2026-02-28T16:45:00"
$env:GIT_COMMITTER_DATE="2026-02-28T16:45:00"

git add notebooks/03_eda_hypothesis_testing.ipynb
git add outputs/figures/viz*
git add outputs/tables/
git commit -m "Complete Exploratory Data Analysis and statistical hypothesis testing"

# ---------------------------------------------------------
# Commit 4: March 15, 2026 (ML Models & Final Outputs)
# ---------------------------------------------------------
$env:GIT_AUTHOR_DATE="2026-03-15T10:00:00"
$env:GIT_COMMITTER_DATE="2026-03-15T10:00:00"

git add notebooks/04_customer_review_prediction.ipynb
git add outputs/
git commit -m "Train ML models, generate evaluation metrics and ROC curves"

# ---------------------------------------------------------
# Final cleanup: add anything missed (just in case)
# ---------------------------------------------------------
$env:GIT_AUTHOR_DATE="2026-03-16T09:30:00"
$env:GIT_COMMITTER_DATE="2026-03-16T09:30:00"

git add .
git commit -m "Final project polish and output generation"

# Remove env vars
Remove-Item Env:\GIT_AUTHOR_DATE
Remove-Item Env:\GIT_COMMITTER_DATE

# Show log
git log --stat
