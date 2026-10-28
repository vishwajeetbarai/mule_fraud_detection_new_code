# Autonomous Mule Account Fraud Detection & Drift-Adaptive Architecture
### An Enterprise & Publication-Grade Benchmark on the 1M-Record NeurIPS 2022 Bank Account Fraud Dataset

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: NeurIPS BAF](https://img.shields.io/badge/Dataset-NeurIPS%202022%20BAF-orange.svg)](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)
[![Test Suite](https://img.shields.io/badge/Pytest-18%20Passed-brightgreen.svg)]()
[![Model Risk: Basel III / SR 11-7](https://img.shields.io/badge/Compliance-Basel%20III%20%7C%20Fed%20SR%2011--7-darkblue.svg)]()

---

## 📌 Executive Summary & Project Background

**What is a Mule Account?**
In retail banking, anti-money laundering (AML), and financial crime, a **money mule** is an account created or compromised to receive illicit funds (from phishing, ransomware, narcotics, or identity theft) and swiftly funnel them through layered financial channels. Because money mules often appear to be standard customers during initial KYC checks, traditional credit scoring models and rules-based fraud engines fail to flag them until bust-out occurs, incurring severe regulatory fines and financial write-offs.

**What is this Project?**
This repository contains a **1-year Master's Level Software Engineering & Technology (SET) Research Project** for an enterprise-ready, publication-grade Machine Learning architecture that detects opening-stage mule accounts. Evaluated on the benchmark **NeurIPS 2022 Bank Account Fraud (BAF) `BaseDataset.csv`** (1,000,000 real opening applications; ~1.1% fraud prevalence; 8 chronological months), the system demonstrates that static machine learning models degrade drastically in production due to concept drift, fraud syndicate evasion, and class imbalance.

To solve this, this repository implements an end-to-end framework featuring:
1. **Behavioral Graph Syndicate Projections & Velocity Spike Calculus** to capture coordinated fraud rings.
2. **Leakage-Safe Temporal Cross-Validation** with strictly frozen reference preprocessing.
3. **Continuous Concept Drift Detection (PSI & 2-Sample KS Testing)** for feature shift and prediction decay.
4. **Demographic Fairness & Algorithmic Bias Auditing** (EEOC 80% Four-Fifths Rule compliance across age and devices).
5. **Cost-Sensitive Financial Loss Optimization** aligned with Basel III operational risk matrices ($10–12% operational savings).
6. **Regulatory Explainable AI (XAI)** generating adverse action notices compliant with RBI Digital Lending & Fed SR 11-7 guidelines.

---

## 🔬 Core Research Novelties & Architecture

```
                       ┌────────────────────────────────────────────────────────┐
                       │  Chronological Bank Applications (Months 0 - 7)        │
                       └──────────────────────────┬─────────────────────────────┘
                                                  │
                                                  ▼
                       ┌────────────────────────────────────────────────────────┐
                       │  Leakage-Safe Preprocessing & Graph Engineering        │
                       │  • Multiplex Entity Collisions (Device × DOB × Branch) │
                       │  • Non-linear Velocity Spikes & Rate Derivatives       │
                       │  • Frozen Historical Baseline Drift Normalization      │
                       └──────────────────────────┬─────────────────────────────┘
                                                  │
                      ┌───────────────────────────┴────────────────────────────┐
                      ▼                                                        ▼
       ┌──────────────────────────────┐                         ┌──────────────────────────────┐
       │   XGBoost / LightGBM         │                         │  Concept Drift Detection     │
       │   • Focal Loss / Resampling  │                         │  • Population Stability (PSI)│
       │   • 5-Fold Expanding Temp-CV │                         │  • 2-Sample KS Kolmogorov    │
       └──────────────┬───────────────┘                         │  • Model Score Distribution  │
                      │                                         └──────────────┬───────────────┘
                      ▼                                                        ▼
       ┌──────────────────────────────┐                         ┌──────────────────────────────┐
       │ Financial Cost Optimization  │                         │  Fairness & Bias Audit       │
       │ Min Operational Loss Matrix  │                         │  • Disparate Impact (EEOC)   │
       │ CFN=$1500, CFP=$30, CTP=$10  │                         │  • Statistical Parity        │
       └──────────────┬───────────────┘                         │  • Age & Device OS Slices    │
                      │                                         └──────────────┬───────────────┘
                      └───────────────────────────┬────────────────────────────┘
                                                  ▼
                       ┌────────────────────────────────────────────────────────┐
                       │  Trustworthy Regulatory Governance & Adverse Action   │
                       │  • Tree-SHAP Waterfall Case Audit Cards                │
                       │  • DeLong's Correlated ROC-AUC Asymptotic Test         │
                       │  • Paired Cross-Validation Student's t / Wilcoxon      │
                       └────────────────────────────────────────────────────────┘
```

### 1. Behavioral Graph & Velocity Spike Dynamics ([`src/features.py`](file:///d:/Downloads/SET%20Project/mule_newcode/src/features.py))
- **Fraud Ring Multiplex Collisions**: Synthesizes bipartite entity node degrees across distinct emails, devices, dates-of-birth, and geographic branches into a proxy hub-centrality and clustering index (`fraud_ring_network_density`, `ring_clique_density_proxy`).
- **Velocity Dynamics**: Computes super-linear acute velocity spikes scaled against monthly background rates:
  $$\text{velocity\_spike\_index} = \frac{v_{6h}^{1.5}}{\sqrt{v_{4w} + 1.0}}$$
  along with acceleration derivatives: $\text{velocity\_momentum\_spike} = \frac{\dot{v}_{6h} - \dot{v}_{24h}}{\dot{v}_{4w}}$.
- **Tenure Friction & Device Multiplexing**: `device_to_address_collision_ratio` flags multiple identities utilizing identical devices from unstable residential addresses.

### 2. Concept Drift Detection Module ([`src/drift.py`](file:///d:/Downloads/SET%20Project/mule_newcode/src/drift.py))
- **Population Stability Index (PSI)**: Establishes reference quantile bins strictly on training/calibration months:
  $$\text{PSI} = \sum_{b=1}^{B} (Actual\%_b - Expected\%_b) \times \ln\left(\frac{Actual\%_b}{Expected\%_b}\right)$$
  Categorizes feature stability: $\text{PSI} < 0.10$ (Stable), $0.10 \le \text{PSI} < 0.25$ (Moderate Drift), $\text{PSI} \ge 0.25$ (Significant Drift).
- **Two-Sample Kolmogorov-Smirnov (KS) Test**: Formulates two-sided non-parametric hypothesis testing against cumulative distribution shifts.
- **Continuous Score Drift**: Monitors shifts in output probability predictions to anticipate calibration decay before ground truth labels arrive.

### 3. Model Fairness & Demographic Bias Audit ([`src/evaluation.py`](file:///d:/Downloads/SET%20Project/mule_newcode/src/evaluation.py))
- Evaluates **Disparate Impact (Four-Fifths / 80% Rule from the US EEOC & Basel Fair Lending standards)**, **Statistical Parity Difference**, and **Equal Opportunity** across:
  - **Protected Demographic Slices**: Age cohorts (`<25`, `25–35`, `35–50`, `50–65`, `65+`).
  - **Device Environments**: `windows`, `linux`, `macintosh`, `x11`, `other`.
- Produces automated audit logs and dual-axis graphical reports tracking approval rates vs. False Positive Rates (FPR) per slice.

### 4. Operational Cost Matrix & Basel III Financial Loss Optimization
In financial banking operations, missing a mule account is orders of magnitude worse than creating customer verification friction:
- $C_{\text{FN}} = \$1,500$ (undiscovered credit bust-out or laundering loss)
- $C_{\text{FP}} = \$30$ (friction cost of step-up verification / KYC document review)
- $C_{\text{TP}} = \$10$ (automated step-up friction cost)
- $C_{\text{TN}} = \$0$ (frictionless approval)

The pipeline scans probability thresholds to identify the minimum operational loss per application, saving **10% to 12% in operational costs** compared to standard uncalibrated 0.5 decision thresholds.

### 5. Rigorous Academic Statistical Validation
- **DeLong's Asymptotic Test**: Tests whether differences between LightGBM and XGBoost ROC curves are statistically significant or attributable to sampling variance ($p$-values, $z$-statistic, 95% Confidence Intervals).
- **Paired Cross-Validation Significance**: Computes Student's paired $t$-test, non-parametric Wilcoxon signed-rank test, and Cohen's $d$ effect sizes across the 5 temporal folds.
- **Adversarial Robustness Stress-Testing**: Assesses model degradation under intentional fraudster evasion noise $\varepsilon \in [0.05, 0.30]$ across critical device and velocity metrics.

---

## 📂 Repository Layout

```
├── data/                                 # BaseDataset.csv (NeurIPS 2022, 1M rows)
├── outputs/                              # Research reports, CSV tables, plots, audit cards
│   ├── model_metrics_table.csv           # Final temporal test metrics (AUC, PR-AUC, TPR@5%FPR)
│   ├── concept_drift_audit_summary.csv   # PSI, KS statistic, and drift alerts per feature
│   ├── fairness_audit_customer_age.csv   # Disparate impact & parity audit on age groups
│   ├── fairness_audit_device_os.csv      # Disparate impact & parity audit on device OS
│   ├── cost_optimization_results.csv     # Financial operational loss matrix optimization
│   ├── delong_roc_test_results.csv       # DeLong's test between LightGBM and XGBoost
│   ├── cv_results_5fold_temporal.csv     # Expanding-window temporal CV fold metrics
│   ├── cv_summary_5fold_temporal.csv     # Mean & standard deviation across CV folds
│   ├── regulatory_compliance_audit_case.png # RBI / Basel III adverse action notice
│   └── *.png                             # ROC, PR, Cost curve, Fairness, SHAP plots
├── src/
│   ├── config.py                         # Central configuration, schemas, loss matrix, hyperparameters
│   ├── data_loader.py                    # Memory-optimized CSV loading and temporal splitting
│   ├── features.py                       # Leakage-safe preprocessor, graph metrics, velocity spikes
│   ├── drift.py                          # PSI, KS-test, score drift tracker, and stream monitor
│   ├── evaluation.py                     # DeLong's test, cost-sensitive matrix, fairness audits, XAI
│   ├── train.py                          # Research benchmark orchestrator & temporal CV runner
│   ├── adaptive_threshold.py             # Prequential dynamic threshold recalibrator
│   └── tdta_experiment.py                # Standalone Temporal Drift-Aware Threshold Adaptation suite
├── tests/
│   ├── test_pipeline.py                  # Pytest unit tests for features, drift math, fairness, DeLong
│   └── test_tdta.py                      # Tests for adaptive thresholding and calibration safeguards
├── requirements.txt                      # Minimum package dependencies
└── README.md                             # Comprehensive technical documentation
```

---

## 🚀 Getting Started & Reproduction Guide

### 1. Environment Setup

```powershell
# Clone the repository
git clone https://github.com/vishwajeetbarai/mule_fraud_detection_new_code.git
cd mule_fraud_detection_new_code

# Create and activate Python virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # On Linux/macOS: source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Dataset Acquisition
Download the official **Bank Account Fraud (BAF) `BaseDataset.csv`** (213 MB) from [Kaggle NeurIPS 2022 Benchmark](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) and place it into the `data/` directory:
```
data/BaseDataset.csv
```
*(Alternatively, set the `BASE_DATASET_PATH` environment variable or use the `--data` flag).*

---

## 🧪 Testing & Execution

### Step 1: Run the Automated Unit Test Suite
Ensure that all mathematical calculations (PSI, KS test, DeLong's mid-ranks, Disparate Impact, velocity spike features, zero-leakage constraints) execute flawlessly:
```powershell
.\.venv\Scripts\python.exe -m pytest
```
*Expected Result:*
```
============================= 18 passed in 24.75s =============================
```

### Step 2: Run a Rapid Pipeline Smoke Test (5% Sample)
To verify the full end-to-end pipeline (feature engineering, XGBoost, LightGBM, DeLong test, Cost Matrix, Concept Drift, Fairness Audit, and 5-Fold Temporal CV) in ~1 minute:
```powershell
.\.venv\Scripts\python.exe src/train.py --sample-frac 0.05 --skip-shap
```

### Step 3: Run the Full Research Benchmark (100% Data)
To generate the final publication-grade models, tables, and figures on the full 1,000,000-row dataset:
```powershell
.\.venv\Scripts\python.exe src/train.py
```

### Optional CLI Arguments:
- `--loss focal`: Train using custom Focal Loss ($\gamma=2.0, \alpha=0.25$) instead of class-weighted cross-entropy.
- `--resample smote`: Apply strictly in-fold SMOTE oversampling inside training folds (zero data leakage).
- `--skip-cv`: Skip the 5-fold temporal cross-validation loop to speed up evaluation.
- `--skip-shap`: Skip Tree-SHAP calculations for quick iteration.
- `--run-tdta`: Launch the extended Temporal Drift-Aware Threshold Adaptation research experiments.

---

## 📊 Benchmark Results Summary

Evaluating on temporal test months (Months 6 & 7) using models trained strictly on historical data (Months 0–4) and calibrated on Month 5:

| Model | ROC-AUC | PR-AUC | TPR @ 5% FPR | Min Operational Cost / App | Saves vs Default (0.5) | DeLong's Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **0.8872** | **0.1384** | **0.5241** | **$11.96** | **11.6%** | $p = 0.985$ (Benchmark Match) |
| **XGBoost** | **0.8869** | **0.1342** | **0.5098** | **$12.90** | **10.0%** | Baseline |

### Key Findings from New Enhancements:
1. **Concept Drift**: Month-over-month evaluation identified that while applicant demographics (`income`, `name_email_similarity`) remain stable ($\text{PSI} < 0.08$), transaction velocity (`velocity_6h`, $\text{PSI}=1.128$) and credit risk scores ($\text{PSI}=0.268$) experience severe distribution drift.
2. **Model Fairness**: The algorithmic bias audit confirmed that both XGBoost and LightGBM meet the EEOC 80% Rule for Disparate Impact across age cohorts and operating systems, with no protected group experiencing an approval rate ratio below 0.849.
3. **Graph Feature Impact**: Synthesized fraud ring density and velocity momentum derivatives consistently rank among the top-15 gain features in Tree-SHAP importance plots.

---

## 📜 Regulatory Standards & Governance Alignment
- **Basel Committee on Banking Supervision (BCBS)**: Model Risk Management framework compliance (Principle 4: ongoing monitoring and conceptual soundness).
- **US Federal Reserve SR 11-7 / OCC 2011-12**: Guidance on Model Risk Management, conceptual soundness validation, and benchmarking.
- **Reserve Bank of India (RBI) Digital Lending Directives (2022)**: Automated Adverse Action notice generation with auditable, human-interpretable feature contribution waterfall charts.
- **US Equal Employment Opportunity Commission (EEOC) Uniform Guidelines**: Algorithmic fairness validation via the Four-Fifths (80%) Rule on Disparate Impact.

---

## 👤 Author & Academic Context
- **Project**: 1-Year Master's Degree (MSc Data Science / Advanced Software Engineering) SET Project.
- **Dataset**: Bank Account Fraud (BAF) Benchmark Suite (NeurIPS 2022 Datasets and Benchmarks Track).
- **Domain**: Financial Machine Learning, AML, Mule Account Detection, and Model Governance.
