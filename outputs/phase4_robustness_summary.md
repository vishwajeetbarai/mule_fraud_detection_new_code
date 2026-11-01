# Phase 4 Adversarial Robustness & Operational Resilience Report

**Project**: Mule Account Fraud Detection (SET Major Research Project)  
**Dataset**: NeurIPS 2022 Bank Account Fraud Benchmark (`BaseDataset.csv`)  
**Scope**: Realistic Constrained Perturbations on Raw Input Data  
**Status**: Executed, Validated, and Statistically Audited  

---

## 1. Executive Summary

Phase 4 validates whether the trained fraud detection framework and Hierarchical Triage Policy remain reliable under realistic, domain-constrained adversarial manipulation of raw input features. 

Unlike conventional machine learning attacks that add unconstrained $L_p$-norm noise directly to normalized feature embeddings, Phase 4 perturbs **raw applicant and client records prior to preprocessing and feature engineering**. This guarantees that:
1. All derived interaction features (e.g., velocity surge acceleration, behavioral graph hub centrality, credit leverage strain, and identity incoherence) are recalculated from the ground up.
2. Attackers are strictly constrained to attributes within applicant or client control (e.g., form submissions, browser OS header, pacing velocity), while verified external data (credit bureau scores, core banking tenure, biological age) remain immutable.
3. Records violating physical sanity checks (e.g., address tenure exceeding applicant lifespan) are formally rejected by the `DomainValidator`.

---

## 2. Experimental Results: Clean vs Perturbed Performance

Evaluated on the out-of-sample chronological test partition (Months 6 & 7, $N=4,100$, 61 true frauds):

| Scenario | Rejection Rate (%) | ROC-AUC ($\Delta$) | PR-AUC ($\Delta$) | Standalone Recall | Realized Cost/App ($\Delta$) | Triage Recall [95% Wilson CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | 0.32% | 0.8403 (0.000) | 0.1256 (0.000) | 0.4426 | $14.10 ($0.00) | **0.9344 [0.8432, 0.9742]** |
| **S1: App Camouflage** | 0.39% | 0.8054 (-0.0349) | 0.0927 (-0.0328) | 0.1803 (-0.262) | $19.06 (+$4.96) | **0.9180 [0.8221, 0.9645]** |
| **S2: Velocity Throttle**| 0.27% | 0.8437 (+0.0034) | 0.1430 (+0.0175) | 0.4754 (+0.033) | $13.48 (-$0.62) | **0.9344 [0.8432, 0.9742]** |
| **S3: Device Cloaking** | 0.32% | 0.8024 (-0.0379) | 0.1390 (+0.0134) | 0.4754 (+0.033) | $14.34 (+$0.24) | **0.9836 [0.9128, 0.9971]** |
| **S4: Sensor Occlusion** | 0.32% | 0.8173 (-0.0230) | 0.1091 (-0.0165) | 0.4590 (+0.016) | $14.17 (+$0.07) | **0.9344 [0.8432, 0.9742]** |
| **S5: Multi-Vector** | 0.34% | **0.7729 (-0.0674)**| **0.0819 (-0.0437)**| **0.2459 (-0.197)** | **$18.00 (+$3.89)** | **0.9016 [0.8016, 0.9541]** |

---

## 3. Operational Action Distribution & Safety-Net Analysis

The central hypothesis of the Hierarchical Triage Policy is that even when adversarial evasion suppresses the model score below the fixed automated threshold $\tau_{\text{cost}}$, the **conformal prediction set and drift guard will classify the application as uncertain, routing it to `HUMAN_REVIEW` rather than allowing it to be `AUTO_APPROVED`**.

| Scenario | % Auto-Approve | % Human Review | % Auto-Flag | Frauds Flagged (Auto) | Frauds Referred (Review) | Frauds Slipped (Approve) | Direct Recall | Triage Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | 37.2% | 56.9% | 5.8% | 27 | 30 | 4 | 44.3% | **93.4%** |
| **S1: App Camouflage** | 44.0% | 53.5% | 2.5% | 11 | 45 | 5 | 18.0% | **91.8%** |
| **S2: Velocity Throttle**| 37.6% | 56.1% | 6.3% | 29 | 28 | 4 | 47.5% | **93.4%** |
| **S3: Device Cloaking** | 21.2% | 69.7% | 9.1% | 29 | 31 | 1 | 47.5% | **98.4%** |
| **S4: Sensor Occlusion** | 23.6% | 69.1% | 7.3% | 28 | 29 | 4 | 45.9% | **93.4%** |
| **S5: Multi-Vector** | 36.8% | 59.3% | 3.9% | 15 | 40 | 6 | 24.6% | **90.2%** |

### Critical Scientific Observations:
1. **The Vulnerability of Standalone Thresholds**:
   In **Scenario 1 (Application Camouflage)**, tuning self-reported form fields successfully cuts the standalone automated detection recall from **44.3% down to 18.0%** ($\Delta = -26.2\%$). If the bank relied exclusively on a static binary decision rule, over 80% of fraudsters would evade detection.
2. **The Conformal Safety-Net in Action**:
   Under Scenario 1, while direct `AUTO_FLAG` catches only 11 frauds, the conformal predictor recognizes the incongruence between verified credit bureau metrics and applicant form fields, categorizing prediction sets as ambiguous $\{0, 1\}$. As a result, **45 fraud cases are routed to `HUMAN_REVIEW`**, preserving overall fraud capture at **91.8%** ($[0.8221, 0.9645]$).
3. **Worst-Case Degradation**:
   **Scenario 5 (Multi-Vector Syndicate)** caused the most severe stress, reducing ROC-AUC by $-0.0674$ (from 0.8403 to 0.7729) and PR-AUC by $-0.0437$ (from 0.1256 to 0.0819). Average simulated loss rose by $+3.89$ per applicant. Even under this coordinated attack, 55 of 61 frauds (90.2%) were intercepted (15 auto-flagged + 40 routed to review), with only 6 slipping through to automated approval.

---

## 4. Domain Validation & Rejection Accounting

The `DomainValidator` successfully intercepted records with invalid physical or chronological states:
- Total test records evaluated per scenario: $4,100$.
- Average records rejected: 11 to 16 records (~0.3% of test set).
- Primary rejection cause: In raw BAF records, a small fraction of baseline rows had pre-existing invalidities (e.g. `current_address_months_count` exceeding applicant lifespan $(age+10)\times 12$).
- No synthetic modifications created out-of-vocabulary categoricals or negative non-sentinel numbers.

---

## 5. Cost Analysis (Simulated Operational Bounds)

Using the standardized simulated cost parameters ($C_{\text{FN}} = \$1500$, $C_{\text{FP}} = \$30$, $C_{\text{TP}} = \$10$):
- **Clean Baseline**: $\$14.10$ / application.
- **S1 (Application Camouflage)**: Increased to $\$19.06$ (+$4.96$) due to false negatives in standalone score.
- **S2 (Velocity Throttle)**: Remained stable at $\$13.48$ (-$0.62$).
- **S3 & S4 (Device & Sensor)**: Remained essentially neutral at $\$14.34$ and $\$14.17$.
- **S5 (Multi-Vector Syndicate)**: Increased to $\$18.00$ (+$3.89$).

