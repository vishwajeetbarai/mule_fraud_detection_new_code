# Bank Account Fraud Detection (NeurIPS 2022 Benchmark)

Fraud detection with **XGBoost** and **LightGBM** on the real
[Bank Account Fraud](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)
benchmark (`BaseDataset.csv`). **No synthetic or mock data is used anywhere in this repo.**

> Scope note: this dataset is about **fraud at account opening** (is this new
> application fraudulent?). It does not contain transaction graphs, so this project
> does not do money-flow / mule-network analysis.

---

## 1. Dataset

| Item | Value |
|---|---|
| File | `data/BaseDataset.csv` (~213 MB, not stored in Git) |
| Rows | 1,000,000 account applications |
| Columns | 31 features + target `fraud_bool` |
| Fraud rate | about 1.10% (11,029 frauds) |
| Time column | `month` (0 to 7), used only for splitting, never as a feature |

Download it from Kaggle and place it at `data/BaseDataset.csv`.
To keep it elsewhere, use `--data PATH` or set the `BASE_DATASET_PATH` environment variable.

---

## 2. Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Put BaseDataset.csv in the data/ folder

# 4. Quick test on 10% of the real rows (a few minutes)
python src/train.py --sample-frac 0.1 --skip-cv --skip-shap

# 5. Sanity tests
pytest tests

# 6. Full run (all rows + cross-validation + SHAP; can take a long time)
python src/train.py
```

Useful options:

| Option | Meaning |
|---|---|
| `--data PATH` | Use a CSV from another location |
| `--output-dir DIR` | Write results somewhere other than `outputs/` |
| `--sample-frac 0.1` | Use a random 10% of the real rows (smoke tests only) |
| `--skip-cv` | Skip temporal cross-validation |
| `--skip-shap` | Skip SHAP explanations |
| `--seed 42` | Random seed |

---

## 3. How the pipeline works

```mermaid
flowchart TD
    A["BaseDataset.csv<br/>1,000,000 real applications"] --> B["Load and check<br/>data_loader.py"]
    B --> C["Split by month<br/>train 0-4 / valid 5 / test 6-7"]
    C --> D["Clean and build features<br/>features.py"]
    D --> E["Train XGBoost and LightGBM<br/>early stopping on month 5"]
    E --> F["Pick alert threshold<br/>on month 5 at 5% FPR"]
    F --> G["Score unseen months 6-7"]
    G --> H["Metrics<br/>ROC-AUC, PR-AUC, TPR at 1% and 5% FPR"]
    H --> I["Explain and save<br/>SHAP, plots, CSVs in outputs/"]
    C --> J["Temporal cross-validation<br/>test months 3 to 7"]
    J --> I
```

| Step | What happens |
|---|---|
| Split | Time-based, never random. Train = months 0-4, validation = month 5, test = months 6-7. This mimics real life: learn from the past, face the future. |
| Missing values | `-1` in 5 columns and negative `intended_balcon_amount` mean "unknown". They become NaN, plus a `_missing` flag column. |
| Features | Velocity ratios (6h/24h/4w), credit limit to income, total address months, phone-valid count. Text columns are one-hot encoded using levels learned from the training months only. |
| Dropped | `device_fraud_count` (always 0), `month` (split key only), and the target. |
| Imbalance | `scale_pos_weight` (about 99) so the rare fraud class is not ignored. |
| Threshold | Chosen on the validation month for about 5% false positives, then applied unchanged to the test months. |
| Cross-validation | Expanding window: train on earlier months, test on the next month (months 3 to 7). |
| Explainability | Gain importance for both models and SHAP for LightGBM. |

---

## 4. Metrics explained

Each application gets a fraud score between 0 and 1. A cutoff decides what gets flagged.

| Metric | Meaning in simple words |
|---|---|
| **ROC-AUC** | Chance that a random fraud gets a higher score than a random good customer. 0.5 = coin flip, 1.0 = perfect. |
| **PR-AUC** | How precise the alerts are across all cutoffs. More honest than ROC-AUC when fraud is rare (about 1%). A random model scores about 0.01. |
| **TPR at 5% FPR** | Share of all frauds caught while wrongly flagging only 5% of good customers. This is the benchmark's standard operating point. |
| **TPR at 1% FPR** | Same idea with a stricter limit of 1% false alarms. |
| Precision / Recall / F1 | Computed at the threshold chosen on the validation month. |

---

## 5. Repository layout

```
.
├── data/
│   ├── README.md          where to put BaseDataset.csv
│   └── .gitkeep
├── src/
│   ├── config.py          paths, schema, split months, hyper-parameters
│   ├── data_loader.py     CSV loading, validation, temporal split
│   ├── features.py        leakage-safe preprocessing and feature engineering
│   ├── evaluation.py      metrics and plots
│   └── train.py           end-to-end entry point
├── tests/
│   └── test_pipeline.py   sanity and leakage tests on real data
├── scripts/
│   └── git_cleanup.sh     one-time cleanup of old synthetic files
├── outputs/               generated results (models/ is git-ignored)
├── requirements.txt
└── README.md
```

---

## 6. Outputs

Running `python src/train.py` writes to `outputs/`:

| File | Content |
|---|---|
| `model_metrics_table.csv` | Test-set metrics for both models |
| `cv_results_5fold_temporal.csv`, `cv_summary_5fold_temporal.csv` | Cross-validation results |
| `feature_importance_xgboost.csv`, `feature_importance_lightgbm.csv` | Gain importance |
| `shap_feature_importance_lgbm.csv` | Mean absolute SHAP values |
| `roc_curve.png`, `precision_recall_curve.png` | Performance curves |
| `fraud_probability_distribution.png` | Score distribution, fraud vs good |
| `lightgbm_feature_importance.png`, `shap_bar_lgbm.png`, `shap_beeswarm_lgbm.png` | Explainability plots |
| `test_predictions.csv`, `models/` | Predictions and saved models (git-ignored) |

---

## 7. Results

Run the full pipeline and paste your numbers from `outputs/model_metrics_table.csv`:

| Model | ROC-AUC | PR-AUC | TPR at 1% FPR | TPR at 5% FPR |
|---|---|---|---|---|
| XGBoost | _fill in_ | _fill in_ | _fill in_ | _fill in_ |
| LightGBM | _fill in_ | _fill in_ | _fill in_ | _fill in_ |

---

## 8. Configuration

All settings are in `src/config.py`: dataset path, split months, XGBoost/LightGBM
parameters, target FPR, and `EXTRA_EXCLUDE_FEATURES`.

## 9. Responsible use

The data includes sensitive attributes such as `customer_age`. Add any attribute you
want removed to `EXTRA_EXCLUDE_FEATURES` in `config.py`. This is a research and
portfolio project, not a production fraud system.

## 10. License and data credit

Dataset: *Bank Account Fraud (BAF) Dataset Suite*, NeurIPS 2022 (Jesus et al.).
Please follow the dataset's own license and citation terms.
