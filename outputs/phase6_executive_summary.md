# Phase 6 Executive Summary: Major Project Final Experimental Synthesis

**Project**: Mule Account Fraud Detection (SET Major Research Project)  
**Author**: MSc Data Science Candidate  
**Benchmark Suite**: NeurIPS 2022 Bank Account Fraud (`BaseDataset.csv`)  
**Scale of Evaluation**: Full 1,000,000 Records  
**Execution Timestamp**: November 2026  

---

## 1. At a Glance: Key Experimental Findings

| Benchmark Milestone | Smoke-Test (2% Sample) | Full 1M Dataset (Final Result) | Academic / Operational Implication |
| :--- | :---: | :---: | :--- |
| **Total Applications** | 20,000 | **1,000,000** | Full benchmark processed without downsampling. |
| **Out-of-Sample Test Set** | 4,100 (61 frauds) | **205,011 (2,878 frauds)** | Rigorous large-sample statistical power. |
| **LightGBM ROC-AUC** | 0.8318 | **0.8869 [0.8809, 0.8930]** | $+0.0551$ gain from full-scale training. |
| **LightGBM PR-AUC** | 0.1206 | **0.1869 [0.1739, 0.2024]** | Narrow confidence interval ($5\times$ variance reduction). |
| **Model Comparison (DeLong)**| $p = 0.6567$ | **$p = 2.01 \times 10^{-3}$ ($z = -3.09$)** | XGBoost statistically outperforms LightGBM at 99% CI. |
| **Direct Automated Recall** | 44.26% | **54.83%** | Applications directly intercepted by automated threshold. |
| **Hierarchical Triage Recall** | 93.44% | **94.02% [0.9310, 0.9483]** | Conformal safety net captures 1,128 additional frauds. |
| **Fraud Escape Rate** | 6.56% | **5.98% (172 / 2,878 frauds)** | Frictionless pass leakage constrained to $<6\%$. |
| **Human Review Workload** | 57.00% | **38.44%** | Feasible operational queue load at full scale. |
| **Automated Approval Rate** | 37.17% | **55.36%** | Majority of applications approved frictionlessly. |
| **Adversarial Multi-Vector** | Triage Recall = 90.16% | **Triage Recall = 92.69%** | Framework resists coordinated syndicate evasion. |
| **60-Day Delayed Feedback** | Triage Recall = 93.44% | **Triage Recall = 94.02%** | Zero additional fraud escape under verification lag. |

---

## 2. Research Integrity Checklist

- [x] **Zero Test-Label Leakage**: Enforced across all threshold selection, calibration, and policy tuning paths.
- [x] **Chronological Temporal Split**: Train (Months 0–4), Validation (Month 5), Out-of-Sample Test (Months 6–7).
- [x] **Pre-Engineering Adversarial Perturbations**: All derived velocity and behavioral graph features recomputed.
- [x] **Non-Asymptotic Uncertainty Quantification**: Exact Wilson Score intervals for binomial proportions; bootstrap for ranking metrics.
- [x] **Full 1M Dataset Execution**: Completely trained and evaluated on 1,000,000 records in 24.6 minutes.
- [x] **Automated Regression Suite**: 47 / 47 tests passed in 75.06 seconds.

---

## 3. Executive Sign-Off

The experimental evaluation for Phase 6 is complete, validated, and reproducible. All artifacts, models, CSV logs, and analytical summaries are finalized in the `outputs/` directory.

