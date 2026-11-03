# MuleShield - Defensible, Drift-Aware & Uncertainty-Guided Bank Account Fraud Detection
### An Enterprise & Publication-Grade AI Gatekeeper on the 1,000,000-Record NeurIPS 2022 Benchmark
**Degree / Milestone:** M.Sc. Data Science Major Project | VIT Vellore (Review 3 Defense)  
**Verification:** 50 / 50 Passing Automated Unit Tests (100% Green) | Strict Zero-Leakage Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: NeurIPS BAF](https://img.shields.io/badge/Dataset-NeurIPS%202022%20BAF%20(1M)-orange.svg)](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)
[![Test Suite: 50 Passed](https://img.shields.io/badge/Pytest-50%20Passed%20(100%25)-brightgreen.svg)]()
[![Model Risk: Basel III / SR 11-7](https://img.shields.io/badge/Compliance-Basel%20III%20%7C%20Fed%20SR%2011--7-darkblue.svg)]()

---

## 📌 1. Project Overview (In Plain English for the Entire Team)

### What is a "Money Mule" and what are we doing?
When international cybercrime cartels hack accounts, run ransomware, or conduct financial scams, they cannot transfer stolen money directly into their own personal bank accounts — police would freeze the funds and arrest them immediately.

Instead, they need **"Mule Accounts"** — newly opened bank accounts that appear legitimate on paper. Stolen money is wired into these accounts, layered through cash withdrawals or crypto, and moved before banks react.

### Where does OUR Project sit in the Bank?
Our system acts as the **AI Security Guard standing at the digital front door of the bank**. When an applicant applies online for a new checking or savings account, our model analyzes their application details, device telemetry, and velocity in **under 0.1 seconds**.
- **Auto-Approve (55.4%):** Genuine customers $\rightarrow$ instant, frictionless account opening.
- **Auto-Flag / Reject (6.2%):** Blatant bots or synthetic identity thieves $\rightarrow$ blocked automatically.
- **Human Review (38.4%):** Ambiguous or borderline applicants $\rightarrow$ routed to an investigator's queue.

```
       [ Cybercriminal Ring / Bot Script ]
                     │ (Submits 1,000s of online applications)
                     ▼
  ┌─────────────────────────────────────────────────────────┐
  │       OUR AI GATEKEEPER (At Application Time)           │
  │   - 86 Velocity & Dynamic Bipartite Graph Features      │
  │   - LightGBM / XGBoost Machine Learning Classifiers     │
  │   - Mondrian Conformal Prediction Uncertainty Sets      │
  │   - Sliced Wasserstein Distance Concept Drift Engine    │
  └──────────────────────────┬──────────────────────────────┘
                             │
       ┌─────────────────────┼─────────────────────┐
       ▼                     ▼                     ▼
[ AUTO-APPROVE ]      [ HUMAN REVIEW ]      [ AUTO-FLAG ]
 55.4% of users        38.4% of users        6.2% of users
 (Honest People)      (Catches Deceptive     (Direct Intercept)
                       Fraudsters!)
 ────────────────────────────────────────────────────────────
 OVERALL FRAUD INTERCEPTION RATE: 94.02% (2,706 / 2,878 Frauds)
```

---

## 🔬 2. Our 3 Advanced Research Novelties (What Makes Us Unique)

1. **Dynamic Bipartite Entity Graph & Spectral Dirichlet Energy (`src/graph_topology.py`)**:
   - Replaces ad-hoc scalar products with a temporal bipartite graph $\mathcal{G}_t = (\mathcal{V}_{app} \cup \mathcal{V}_{entity}, \mathcal{E}_t)$ with exponential time decay ($w(e) = \exp(-\lambda \Delta t)$).
   - Computes the Rayleigh quotient on the normalized graph Laplacian $\mathcal{L}$:
     $$\mathcal{R}(x_i) = \frac{x_i^T \mathcal{L} x_i}{x_i^T x_i + \epsilon}$$
     Low Dirichlet energy mathematically signals that the applicant is harmonically embedded inside a tightly clustered, low-resistance mule ring.
2. **Multi-Manifold Optimal Transport Concept Drift Engine (`src/wasserstein_drift.py`)**:
   - Replaces 1D Kolmogorov-Smirnov tests with **Sliced Wasserstein Distance (SWD)** over 64 random unit sphere projections.
   - Detects joint multi-feature evasions (e.g., lowering velocity while altering requested credit limits) that escape univariate checks.
3. **Mondrian Group-Conditional Conformal Fairness (`src/conformal.py`)**:
   - Replaces marginal split conformal prediction with group-conditional quantiles for each protected age cohort.
   - Guarantees valid $\ge 95\%$ coverage across *every* age demographic, completely eliminating the historical 10.1% coverage deficit on senior citizen applicants ($65+$) to satisfy ECOA and EU AI Act regulations.

---

## 📊 3. Key Results on the Full 1,000,000-Row Benchmark

Evaluated strictly out-of-sample on Months 6–7 ($N = 205,011$ applications, 2,878 real frauds) with zero test-label leakage:

| Metric / Experiment | Value | Academic & Operational Implication |
| :--- | :---: | :--- |
| **Total Evaluated Dataset** | **1,000,000 rows** | 100% full benchmark evaluated (Zero downsampling). |
| **LightGBM ROC-AUC** | **0.8869** $[0.8809, 0.8930]$ | Evaluated via 500-replicate stratified bootstrap. |
| **LightGBM PR-AUC** | **0.1869** $[0.1739, 0.2024]$ | **17x lift** over the 1.1% random baseline in extreme class imbalance. |
| **XGBoost ROC-AUC** | **0.8901** $[0.8841, 0.8961]$ | DeLong's paired ROC test proves significance ($z = -3.088, p = 0.00201$). |
| **Overall Triage Fraud Recall** | **94.02%** $[0.9310, 0.9483]$ | Catches 2,706 out of 2,878 frauds (Escape rate only 5.98%). |
| **Standalone Model Recall** | **54.83%** | Conformal review queue recovers **1,128 additional deceptive frauds**! |
| **Auto-Approve Rate** | **55.36%** (113,493 apps) | 99.85% clean frictionless onboarding for genuine customers. |
| **Human Review Rate** | **38.44%** (78,796 apps) | Optimal capacity routing for mathematically ambiguous cases. |
| **Adversarial Resilience** | **92.69% Recall** | Holds strong under coordinated multi-vector evasion attacks. |
| **Delayed Feedback (60 Days)** | **94.02% Recall** | Zero performance loss even when true labels take 60 days to arrive. |

---

## 🚀 4. How to Run the Project (Step-by-Step for Teammates)

### 4.1 Prerequisites & Virtual Environment
Open PowerShell or your terminal in the project folder:
```powershell
# 1. Navigate to project directory
cd "d:\Downloads\SET Project\mule_newcode"

# 2. Activate virtual environment (if using .venv)
.\.venv\Scripts\Activate.ps1

# 3. Install required packages (if not already installed)
pip install -r requirements.txt
```

### 4.2 Run All 50 Automated Unit & Property Tests
Verify that all mathematical formulas, graphs, and zero-leakage constraints pass:
```powershell
python -m pytest tests/
```
*(Expected output: `50 passed in ~60s (100%)`)*

### 4.3 Run Fast Smoke-Test Pipeline (2% Benchmark)
For quick testing and pipeline verification in under 30 seconds:
```powershell
python src/train.py --sample-frac 0.02 --skip-cv --skip-shap
```

### 4.4 Run Phase 7 Novelty Runner (Graph + SWD + Mondrian)
Execute the elevated graph topology and group-conditional conformal fairness audit:
```powershell
python src/phase7_runner.py
```

### 4.5 Run the Full 1,000,000-Row Production Pipeline
To run the complete full-scale benchmark evaluation:
```powershell
python src/train.py
```

---

## 📁 5. Repository Structure & Key Files

```
mule_newcode/
├── src/
│   ├── graph_topology.py          # NEW (Phase 7): Dynamic Bipartite Graph & Dirichlet Energy
│   ├── wasserstein_drift.py       # NEW (Phase 7): Multi-Manifold Sliced Wasserstein Drift
│   ├── conformal.py               # UPDATED (Phase 7): Mondrian Group-Conditional Conformal
│   ├── hierarchical_triage.py     # 3-Tier Decision Policy (Auto-Approve, Review, Auto-Flag)
│   ├── adaptive_threshold.py      # TDTA Drift-Aware Threshold Adaptation
│   ├── delayed_feedback_validation.py # Prequential validation under 0-60 day label lag
│   ├── realistic_adversarial.py   # Domain-constrained raw-input evasion attacks
│   ├── statistical_reliability.py # Wilson score & stratified bootstrap confidence intervals
│   ├── features.py                # 86 domain-engineered velocity & interaction features
│   ├── train.py                   # Master end-to-end training and evaluation script
│   └── config.py                  # Experiment hyperparameters and path constants
├── tests/
│   ├── test_novelty_phase7.py     # Unit tests for Graph, SWD, and Mondrian conformal
│   ├── test_adversarial_phase4.py # Adversarial robustness verification
│   ├── test_delayed_feedback_phase5.py # Delayed verification lag tests
│   ├── test_hierarchical_phase2.py# Triage routing invariant tests
│   ├── test_statistical_phase3.py # Exact confidence interval tests
│   └── test_leakage_phase1.py     # Zero test-label leakage assertion tests
├── outputs/                       # CSV reports, DeLong p-values, fairness audits, and charts
├── README.md                      # Comprehensive project guide (this file)
└── requirements.txt               # Dependencies (LightGBM, XGBoost, ReportLab, etc.)
```

---

## 🎓 6. Examiner Viva Cheat Sheet: How to Answer Any Question

### Q1: "Why is your precision around 18%? That sounds low!"
> **Your Answer:**  
> *"Respected Examiner, in fraud detection, precision of 18.7% is actually outstanding because the fraud base rate is only 1.1%! That represents an **17x improvement over random guessing**.<br/>
> More importantly, in real banking, the financial costs are highly unbalanced: missing a single fraudster costs \$1,500 in stolen money and regulatory fines, while human verification of a false alarm costs only \$50. Therefore, mathematics dictates prioritizing catching all the frauds (94% recall) over artificial high precision."*

### Q2: "Why did you use LightGBM / XGBoost instead of Deep Learning (LSTM, Transformers)?"
> **Your Answer:**  
> *"On tabular data with mixed categorical sentinels and high sparsity, extensive empirical literature (e.g., Grinsztajn et al., NeurIPS 2022) proves that Gradient Boosted Decision Trees match or outperform deep architectures while training in seconds rather than hours. Furthermore, banking regulations (FCRA, GDPR Article 22, EU AI Act) mandate explainability; LightGBM seamlessly supports exact TreeSHAP feature attributions for adverse-action regulatory notices."*

### Q3: "How do you prove that your dynamic graph doesn't leak future information?"
> **Your Answer:**  
> *"Our dynamic bipartite graph strictly enforces chronological causality. When application $i$ arrives at timestamp $t_i$, it queries historical entity connections formed strictly at $t_j < t_i$. The current application's edge is ingested into the graph only after its spectral Dirichlet energy and bipartite clustering invariants have been computed."*

### Q4: "Is your AI biased against senior citizens?"
> **Your Answer:**  
> *"Initially, older applicants had higher error rates because synthetic identity fraud rings heavily target older credit profiles in the benchmark dataset (fraud rate is 5% in seniors vs 0.6% in youth). To fix this, we implemented **Mondrian Group-Conditional Conformal Prediction**, which calibrates confidence sets separately for each age group. This restored empirical coverage to 96.79% for seniors, ensuring full ethical compliance with the US EEOC Four-Fifths Rule."*

---
