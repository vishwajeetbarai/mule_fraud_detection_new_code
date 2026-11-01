# Autonomous Mule Account Fraud Detection & Drift-Adaptive Architecture
### An Enterprise & Publication-Grade Benchmark on the 1M-Record NeurIPS 2022 Bank Account Fraud Dataset

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: NeurIPS BAF](https://img.shields.io/badge/Dataset-NeurIPS%202022%20BAF-orange.svg)](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)
[![Test Suite](https://img.shields.io/badge/Pytest-24%20Passed-brightgreen.svg)]()
[![Model Risk: Basel III / SR 11-7](https://img.shields.io/badge/Compliance-Basel%20III%20%7C%20Fed%20SR%2011--7-darkblue.svg)]()

---

## 📌 Executive Summary & Academic Framing (Priority 1)

### What the Dataset Represents & Research Grounding
- **Dataset Source**: NeurIPS 2022 Datasets and Benchmarks Track — Bank Account Fraud (BAF) suite (`BaseDataset.csv`).
- **Nature of the Records**: The BAF dataset contains **1,000,000 tabular opening applications** spanning 8 chronological months (Months 0–7) with an average fraud prevalence of ~1.10%–1.25%. The dataset is synthetically generated using tabular CTGANs trained on real-world retail banking distributions to preserve privacy and statistical properties without exposing PII (Personally Identifiable Information).
- **Fraud Label Semantics (`fraud_bool`)**: In the BAF benchmark, `fraud_bool = 1` indicates **first-party and third-party opening-stage bank-account fraud** (identity impersonation, synthetic identity fraud, stolen credentials, and application spoofing).
- **Mule Account Distinction**:
  > [!IMPORTANT]
  > **Crucial Academic Distinction**: General bank-account fraud is **not** identical to confirmed, post-opening money mule laundering activity. A money mule is an account used to receive and funnel illicit funds downstream. 
  > In this research project, **opening-stage bank account fraud serves as the upstream operational proxy for detecting nascent mule account creation**. We do **not** claim to identify confirmed money mules from transaction ledgers in this benchmark; rather, we frame the task as **robust bank-account fraud detection with opening-stage mule account prevention as the intended operational application and downstream validation direction**.

---

## 🔬 Core Novelties & Review 3 Research Contributions

This repository addresses key research priorities essential for publication-grade Master's level evaluations:

```
                       ┌────────────────────────────────────────────────────────┐
                       │  Chronological Bank Applications (Months 0 - 7)        │
                       └──────────────────────────┬─────────────────────────────┘
                                                  │
                                                  ▼
                       ┌────────────────────────────────────────────────────────┐
                       │  Leakage-Safe Preprocessing & Graph Engineering        │
                       │  • Multiplex Entity Collisions (Device × DOB × Branch) │
                       │  • Multi-Hop Bipartite Projection (Device-Branch Link) │
                       │  • Non-linear Velocity Spikes & Rate Derivatives       │
                       └──────────────────────────┬─────────────────────────────┘
                                                  │
                      ┌───────────────────────────┴────────────────────────────┐
                      ▼                                                        ▼
       ┌──────────────────────────────┐                         ┌──────────────────────────────┐
       │   XGBoost / LightGBM         │                         │  Active Concept Drift (P2)   │
       │   • Cost-Sensitive Matrix    │                         │  • Continuous PSI / KS Shift │
       │   • Expanding Temporal CV    │                         │  • No-Drift vs Retrain vs    │
       │   • Delayed Verification Lag │                         │    Adaptive Threshold (TDTA) │
       └──────────────┬───────────────┘                         └──────────────┬───────────────┘
                      │                                                        │
                      ▼                                                        ▼
       ┌──────────────────────────────┐                         ┌──────────────────────────────┐
       │ Multi-Scenario Cost (P3)     │                         │  Enhanced Fairness (P5)      │
       │ • Scenario A (Low FN/High FP)│                         │  • Group-wise FPR, FNR, Rec  │
       │ • Scenario B (High FN/Low FP)│                         │  • Wilson 95% CIs on Slices  │
       │ • Scenario C (Severe FN)     │                         │  • Pareto Cost-Fairness      │
       │ • Capacity Triage (1%,2%,5%) │                         └──────────────┬───────────────┘
       └──────────────┬───────────────┘                                        │
                      │                                                        ▼
                      │                                         ┌──────────────────────────────┐
                      │                                         │ Conformal Calibration (P6)   │
                      │                                         │ • Guaranteed Target Coverage │
                      │                                         │ • Auto-Approve / Review Set  │
                      │                                         └──────────────┬───────────────┘
                      └───────────────────────────┬────────────────────────────┘
                                                  ▼
                       ┌────────────────────────────────────────────────────────┐
                       │ Robustness Under Evasion Perturbation Budgets (P4)    │
                       │ • Evasion Noise: 0%, 1%, 2%, 5%, 10% on Velocity/Score │
                       │ • Preserves Protected & Sensitive Slices               │
                       └────────────────────────────────────────────────────────┘
```

### 1. Active Drift Adaptation (Priority 2: Beyond Descriptive Drift)
Rather than merely describing drift via PSI and KS statistics, this pipeline benchmarks three distinct adaptation paradigms on incoming drift-affected months:
1. **Static Baseline (No Drift Handling)**: Frozen reference threshold ($0.5$ or static validation threshold).
2. **Periodic Retraining**: Re-fitting the gradient booster on the most recent historical window.
3. **Temporal Drift-Aware Threshold Adaptation (TDTA)**: Prequential quantile-matching that continuously recalibrates decision boundaries based on score distribution shifts without requiring immediate ground-truth verification labels.

### 2. Multi-Scenario Cost Optimization & Investigation Triage (Priority 3)
Banks operate under varying regulatory risk postures and investigator bandwidth constraints. The pipeline evaluates three realistic cost matrices and capacity-constrained triage:
- **Scenario A (Aggressive Consumer Growth / Low FN, High FP)**: $C_{\text{FN}}=\$500$, $C_{\text{FP}}=\$100$. Prioritizes low customer onboarding friction.
- **Scenario B (Standard Retail Bank / High FN, Low FP)**: $C_{\text{FN}}=\$1,500$, $C_{\text{FP}}=\$30$. Standard baseline balancing loan losses and review friction.
- **Scenario C (High-Risk FinTech / Severe FN, Moderate FP)**: $C_{\text{FN}}=\$3,000$, $C_{\text{FP}}=\$50$. Extreme penalty for missed fraud.
- **Capacity-Constrained Triage**: Computes precision, recall, and threshold cutoffs at fixed operational investigation capacities: **Top 1%**, **Top 2%**, and **Top 5%** of all inbound applications.

### 3. Realistic Adversarial Robustness under Perturbation Budgets (Priority 4)
Measures degradation under intentional fraudster manipulation across five perturbation budgets:
$$\varepsilon \in \{0.00, 0.01, 0.02, 0.05, 0.10\}$$
- **Targeted Features**: Actionable applicant levers (`velocity_6h`, `session_length_in_minutes`, `credit_risk_score`).
- **Protected Invariants**: Demographic attributes (`customer_age`, `device_os`, `housing_status`) are strictly preserved to ensure realistic evasion without synthetic manipulation of protected classes.

### 4. Group-Wise Fairness Audit & 95% Confidence Intervals (Priority 5)
Evaluates algorithmic equity across age cohorts (`<25`, `25–35`, `35–50`, `50–65`, `65+`) and device OS (`windows`, `linux`, `macintosh`, `x11`, `other`) reporting:
- **Group-Wise Metrics**: False Positive Rate (FPR), False Negative Rate (FNR), Recall (TPR), Precision, and Disparate Impact (EEOC 80% Rule).
- **Statistical Reliability**: Computes normal and **Wilson score 95% confidence intervals** on small groups (e.g., `Age 65+`, `x11`) to prevent small-sample misinterpretations in regulatory compliance filings.

### 5. Conformal Prediction with Finite-Sample Coverage Guarantees (Priority 6)
Implements Split Conformal Prediction for risk-calibrated triage:
- **Target Error Rate ($\alpha$)**: Defaults to $\alpha = 0.05$ (guaranteeing $\ge 95\%$ marginal coverage).
- **Automated Triage Output**:
  - **Auto-Approve**: Singleton $\{0\}$ — provably low-risk applicants approved instantly.
  - **Human-Review**: Multi-class set $\{0, 1\}$ — borderline or ambiguous applicants routed to AML investigators.
  - **Auto-Reject**: Singleton $\{1\}$ — high-confidence fraud flagged without manual delay.
- Reports empirical coverage, mean prediction set size, human-review ratio, and the percentage of all fraud captured within the review set.

---

## 📂 Repository Layout

```
├── data/                                 # BaseDataset.csv (NeurIPS 2022, 1M rows)
├── outputs/                              # Research reports, CSV tables, plots, audit cards
│   ├── model_metrics_table.csv           # Final temporal test metrics (AUC, PR-AUC, TPR@5%FPR)
│   ├── cost_scenarios_multi.csv          # Multi-scenario cost matrices (Scenarios A, B, C)
│   ├── investigation_capacity_table.csv  # Top 1%, 2%, 5% capacity-constrained triage metrics
│   ├── enhanced_fairness_audit.csv       # Group-wise FPR, FNR, Recall, Precision with 95% CIs
│   ├── concept_drift_audit_summary.csv   # PSI, KS statistic, and drift alerts per feature
│   ├── fairness_cost_pareto_frontier.csv # Pareto frontier balancing cost vs demographic disparity
│   ├── delong_roc_test_results.csv       # DeLong's test between LightGBM and XGBoost
│   ├── cv_results_5fold_temporal.csv     # Expanding-window temporal CV fold metrics
│   └── *.png                             # ROC, PR, Cost curve, Fairness, SHAP plots
├── src/
│   ├── config.py                         # Central configuration, schemas, loss matrix, hyperparameters
│   ├── data_loader.py                    # Memory-optimized CSV loading and temporal splitting
│   ├── features.py                       # Leakage-safe preprocessor, graph metrics, velocity spikes
│   ├── drift.py                          # PSI, KS-test, score drift tracker, and stream monitor
│   ├── evaluation.py                     # DeLong's test, cost-sensitive matrix, fairness audits, XAI
│   ├── review3_enhancements.py           # Multi-scenario cost, capacity triage, group-wise CIs
│   ├── conformal.py                      # Split conformal prediction and coverage verification
│   ├── pareto_fairness.py                # Multi-objective Pareto optimization (cost vs fairness)
│   ├── delayed_feedback.py               # Asynchronous fraud label verification lag simulation
│   ├── adaptive_threshold.py             # Prequential dynamic threshold recalibrator
│   └── train.py                          # Research benchmark orchestrator & temporal CV runner
├── tests/
│   ├── test_pipeline.py                  # Pytest unit tests for features, drift math, fairness, DeLong
│   ├── test_tdta.py                      # Tests for adaptive thresholding and calibration safeguards
│   ├── test_novelty_phases.py            # Tests for conformal, Pareto, and delayed feedback
│   └── test_review3_enhancements.py      # Tests for multi-scenario cost, capacity, and group CIs
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

---

## 🧪 Testing & Execution

### Step 1: Run the Automated Unit Test Suite (24 Tests)
Ensure all mathematical calculations (PSI, KS test, DeLong's mid-ranks, Disparate Impact, Wilson CIs, conformal prediction, and graph features) execute with zero errors:
```powershell
.\.venv\Scripts\python.exe -m pytest
```
*Expected Output:*
```
============================= 24 passed in 23.74s =============================
```

### Step 2: Rapid Pipeline Smoke Test (2% Sample)
To test the complete end-to-end benchmark (including Review 3 cost scenarios, investigation capacity triage, conformal prediction, and enhanced fairness):
```powershell
.\.venv\Scripts\python.exe src/train.py --sample-frac 0.02 --skip-shap --skip-cv
```

### Step 3: Run the Full Research Benchmark (100% Data)
To generate the final publication-grade models, tables, and figures on the full 1,000,000-row dataset:
```powershell
.\.venv\Scripts\python.exe src/train.py
```

### Optional CLI Arguments:
- `--loss focal`: Train using custom Focal Loss ($\gamma=2.0, \alpha=0.25$) instead of class-weighted cross-entropy.
- `--resample smote`: Apply strictly in-fold SMOTE oversampling inside training folds (zero data leakage).
- `--skip-cv`: Skip the 5-fold temporal cross-validation loop for rapid verification.
- `--skip-shap`: Skip Tree-SHAP calculations for quick iteration.
- `--run-tdta`: Launch the extended Temporal Drift-Aware Threshold Adaptation research experiments.

---

## 📊 Benchmark Results Summary

Evaluating on temporal test months (Months 6 & 7) using models trained strictly on historical data (Months 0–4) and calibrated on Month 5:

| Model | ROC-AUC | PR-AUC | TPR @ 5% FPR | Min Cost / App (Scenario B) | Cost Savings vs 0.5 | DeLong's Test |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **0.887** | **0.134** | **0.510** | **$11.85** | **19.4%** | Baseline |
| **LightGBM** | **0.887** | **0.138** | **0.524** | **$12.31** | **33.3%** | $p = 0.657$ (Comparable) |

### Review 3 Key Experimental Findings:
1. **Multi-Scenario Cost Dynamics (Priority 3)**:
   - *Scenario A ($C_{\text{FN}}=500, C_{\text{FP}}=100$)*: Optimal decision threshold rises to $\approx 0.35$–$0.45$, lowering false positives to protect applicant onboarding friction.
   - *Scenario B ($C_{\text{FN}}=1500, C_{\text{FP}}=30$)*: Optimal threshold lowers to $\approx 0.08$–$0.23$, prioritizing fraud capture and cutting operational losses by $19\%$–$33\%$.
   - *Scenario C ($C_{\text{FN}}=3000, C_{\text{FP}}=50$)*: Optimal threshold drops to $<0.10$ to aggressively intercept catastrophic laundering write-offs.
2. **Fixed Investigation Capacity (Priority 3)**:
   - At **Top 1%** capacity: Precision is $21.9\%$ with Recall at $14.8\%$.
   - At **Top 2%** capacity: Precision is $18.3\%$ with Recall at $24.6\%$.
   - At **Top 5%** capacity: Precision is $12.7\%$ with Recall at $42.6\%$, capturing nearly half of all fraudulent applications within just 5% manual triage.
3. **Conformal Prediction Guarantees (Priority 6)**:
   - At nominal $1-\alpha = 0.95$ coverage, the model achieves **$94.83\%$ empirical coverage**.
   - $37.2\%$ of inbound applicants are **auto-approved** with zero human review delay, while $57.1\%$ are routed to manual investigation, successfully intercepting $>88\%$ of all fraud attempts inside the triage pool.
4. **Demographic Equity & Wilson Confidence Intervals (Priority 5)**:
   - Disparate Impact satisfies the EEOC Four-Fifths ($80\%$) rule across all age brackets ($0.853 \le \text{DI} \le 1.000$) and operating systems ($0.874 \le \text{DI} \le 1.000$).
   - Small minority slices (e.g., `Age 65+` with $N=31$, `OS x11` with $N=27$) exhibit wider $95\%$ Wilson confidence intervals ($[\text{FPR}_{\text{low}}, \text{FPR}_{\text{high}}]$), preventing erroneous claims of bias that arise purely from small-sample statistical noise.

---

## 📜 Regulatory Standards & Model Governance
- **Basel Committee on Banking Supervision (BCBS)**: Model Risk Management framework compliance (Principle 4: continuous drift monitoring and conceptual soundness).
- **US Federal Reserve SR 11-7 / OCC 2011-12**: Guidance on Model Risk Management, conceptual soundness validation, and rigorous benchmarking.
- **Reserve Bank of India (RBI) Digital Lending Directives (2022)**: Automated adverse action notice generation with auditable, human-interpretable feature contribution waterfall charts.
- **US Equal Employment Opportunity Commission (EEOC) Uniform Guidelines**: Algorithmic fairness validation via the Four-Fifths (80%) Rule on Disparate Impact.

---

## 👤 Author & Academic Context
- **Project**: 1-Year Master's Degree (MSc Data Science / Advanced Software Engineering) SET Research Project.
- **Dataset**: Bank Account Fraud (BAF) Benchmark Suite (NeurIPS 2022 Datasets and Benchmarks Track).
- **Domain**: Financial Machine Learning, AML, Mule Account Fraud Detection, and Model Governance.
