# Phase 6 Full Dataset Profile: NeurIPS 2022 Bank Account Fraud (BAF)

**Benchmark**: NeurIPS 2022 Bank Account Fraud Suite (`BaseDataset.csv`)  
**Scale**: Full 1,000,000 Records  
**Verification Date**: November 2026  
**Status**: 100% Processed without Downsampling  

---

## 1. Overall Dataset Characteristics

- **Total Row Count**: Exactly **1,000,000 applications**.
- **Number of Raw Columns**: 32 (31 features + 1 target `fraud_bool`).
- **Engineered Dimensionality**: 86 numerical features (including behavioral graph metrics, velocity surge accelerations, credit leverage strain, and rolling baseline drift z-scores).
- **Target Variable**: `fraud_bool` $\in \{0, 1\}$.
- **Overall Class Imbalance**:
  - Legitimate (Class 0): **988,971 records (98.897%)**
  - Fraud (Class 1): **11,029 records (1.103%)**
  - Overall Fraud Prevalence: **1.1029%** (~1 fraud per 90 applications).
- **Duplicate Records**: 0 duplicate rows detected.
- **Missingness Pattern**:
  - Missingness is encoded using domain sentinels (e.g. `-1` in `prev_address_months_count`, `current_address_months_count`, `bank_months_count`, `session_length_in_minutes`, `device_distinct_emails_8w`, and negative values in `intended_balcon_amount`).
  - Explicit boolean missing indicators (`_missing`) are extracted for all sentinel columns.

---

## 2. Chronological Partitioning & Target Distribution

The project strictly adheres to the official benchmark temporal split protocol:
- **Months 0–4 (Training)**: Historical batch used for feature fitting and gradient-boosted tree training.
- **Month 5 (Validation)**: Carved out strictly for early stopping, model selection, non-conformity calibration, and cost-optimal threshold determination. Test data is never touched during model selection.
- **Months 6–7 (Out-of-Sample Test)**: Held out strictly for final out-of-sample evaluation.

| Partition | Months Included | Total Applications | Legitimate Count | Fraud Count | Fraud Prevalence (%) | Cumulative Share (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training** | `[0, 1, 2, 3, 4]` | **675,666** | 668,926 | 6,740 | 0.9975% | 67.57% |
| **Validation** | `[5]` | **119,323** | 117,912 | 1,411 | 1.1825% | 11.93% |
| **Out-of-Sample Test**| `[6, 7]` | **205,011** | 202,133 | 2,878 | 1.4038% | 20.50% |
| **Total** | `[0 - 7]` | **1,000,000** | **988,971** | **11,029** | **1.1029%** | **100.00%** |

---

## 3. Month-by-Month Evolution & Temporal Concept Drift

Analyzing the dataset month-by-month demonstrates clear non-stationary distribution shift across time:

| Month Index | Total Applications | Fraud Cases | Fraud Rate (%) | Role in Experimental Protocol |
| :---: | :---: | :---: | :---: | :--- |
| **Month 0** | 132,440 | 1,500 | 1.1326% | Model Training |
| **Month 1** | 127,620 | 1,198 | 0.9387% | Model Training |
| **Month 2** | 136,979 | 1,198 | 0.8746% | Model Training |
| **Month 3** | 150,936 | 1,392 | 0.9222% | Model Training |
| **Month 4** | 127,691 | 1,452 | 1.1371% | Model Training |
| **Month 5** | 119,323 | 1,411 | 1.1825% | **Validation & Calibration (Zero-Leakage Boundary)** |
| **Month 6** | 108,168 | 1,450 | 1.3405% | **Out-of-Sample Test (Period 1)** |
| **Month 7** | 96,843 | 1,428 | 1.4746% | **Out-of-Sample Test (Period 2)** |

### Critical Empirical Insight:
- **Rising Fraud Prevalence**: From Month 2 (0.87%) to Month 7 (1.47%), the underlying fraud prevalence increases by **+68.6%**.
- **Significant Behavioral Drift**: As measured by the Population Stability Index (PSI) between validation (Month 5) and test (Months 6–7), `velocity_6h` exhibits severe drift (**PSI = 1.0886**, KS Stat = 0.4411, $p = 0.0$), and `credit_risk_score` exhibits significant drift (**PSI = 0.2612**, KS Stat = 0.1791, $p = 0.0$).
- This empirical shift establishes the core motivation for Temporal Drift-Aware Threshold Adaptation (TDTA) and the Hierarchical Triage Policy.

