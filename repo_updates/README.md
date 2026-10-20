# Bank Account Fraud Detection (NeurIPS 2022 Benchmark)

Gradient-boosted fraud detection (XGBoost and LightGBM) on the real
**Bank Account Fraud `BaseDataset.csv`**: 1,000,000 account-opening applications,
31 features, `fraud_bool` target (~1.1% fraud). No synthetic data is used.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download `BaseDataset.csv` from
[Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)
into `data/` (git-ignored, 213 MB). Or use `--data PATH` / env var `BASE_DATASET_PATH`.

## Run

```bash
python src/train.py                                              # full research run
python src/train.py --loss focal                                 # train with Focal Loss
python src/train.py --resample smote                             # in-fold leakage-safe SMOTE
python src/train.py --sample-frac 0.1 --skip-cv --skip-shap         # quick smoke test on 10% sample
pytest tests                                                     # sanity & leakage unit tests
```

## Research Components

| Component | Technical Implementation |
|---|---|
| **1. Behavioral Graph & Velocity Dynamics** | Bipartite degree products (`device_distinct_emails_8w` × `dob_distinct_emails_4w`), graph hub centrality proxy, device recycling velocity, velocity surge acceleration ratios, leverage strain, and frozen training baseline drift z-scores. |
| **2. Imbalanced Learning & Cost Optimization** | Custom binary Focal Loss (`--loss focal`), strictly in-fold SMOTE resampling (zero data leakage), and expected monetary cost optimization curve over banking loss matrix ($C_{FN}=\$1500$, $C_{FP}=\$30$, $C_{TP}=\$10$). |
| **3. Trustworthiness & Regulatory XAI** | Global SHAP feature interaction dependence plots, Beeswarm/Bar summaries, and individual Adverse Action / Audit Waterfall cards compliant with RBI Digital Lending Guidelines and Basel III / Fed SR 11-7 model risk governance. |
| **4. Rigorous Academic Evaluation & Robustness** | DeLong's asymptotic test for correlated ROC-AUC differences, 5-fold temporal CV paired t-test and Wilcoxon signed-rank tests with Cohen's d effect size, and adversarial evasion stress-testing across perturbation budgets $\varepsilon \in [0.05, 0.30]$. |

## Repository layout

```
src/config.py        paths, schema, cost matrix, focal loss hyperparameters
src/data_loader.py   CSV loading, compact dtypes, temporal splitting (0-4 / 5 / 6-7)
src/features.py      leakage-safe preprocessing, behavioral graph metrics, drift baselines
src/evaluation.py    metrics, DeLong's test, cost-sensitive optimization, adversarial stress, regulatory audit cards
src/train.py         research orchestrator, custom focal loss, in-fold SMOTE, temporal CV
tests/               unit tests for features, leakage prevention, DeLong test, and cost optimization
outputs/             metrics, statistical tests, cost curves, SHAP, and stress testing outputs
```

## Generated Outputs (`outputs/`)

- `model_metrics_table.csv` — Primary benchmark metrics (ROC-AUC, PR-AUC, TPR@1%FPR, TPR@5%FPR).
- `cost_optimization_results.csv` & `cost_optimization_curve.png` — Operational loss per application across decision thresholds.
- `delong_roc_test_results.csv` — DeLong's statistical significance test between LightGBM and XGBoost.
- `cv_paired_significance_test.csv` — Cross-validation paired t-test, Wilcoxon test, and Cohen's d effect size.
- `adversarial_stress_test_summary.csv` & `adversarial_stress_curve.png` — Model degradation under adversarial feature evasion.
- `regulatory_compliance_audit_case.png` — Regulatory adverse action audit notice for high-risk accounts.
- `shap_bar_lgbm.png`, `shap_beeswarm_lgbm.png`, `shap_interaction_top_pair.png` — Interpretability figures.

