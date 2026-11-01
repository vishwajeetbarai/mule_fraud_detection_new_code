# Phase 5 Delayed-Label Feedback & Temporal Prequential Report

**Project**: Mule Account Fraud Detection (SET Major Project)  
**Dataset**: NeurIPS 2022 Bank Account Fraud Suite (`BaseDataset.csv`)  
**Scope**: Realistic Verification Lag, Adaptive Thresholding (TDTA), and Hierarchical Triage  
**Status**: Executed, Audited, and Methodologically Verified  

---

## 1. Executive Summary

Phase 5 addresses one of the most critical real-world challenges in machine learning for financial fraud detection: **the delay in receiving ground-truth labels**. 

While academic research often assumes that fraud labels become known immediately upon transaction scoring, live banking environments face substantial investigation lags (30 to 90 days). This experiment investigates:
1. What happens to Temporal Drift-Aware Threshold Adaptation (TDTA) when labels arrive with a 7, 14, 30, or 60-day lag.
2. Whether the Hierarchical Triage Policy (combining cost thresholds, TDTA, and conformal uncertainty) safely buffers against feedback delays.
3. How label availability affects threshold recalibration, operational costs, human-review workload, and fraud escape rates.

---

## 2. Experimental Setup: Time Mapping & Delay Scenarios

- **Temporal Representation**: Discrete dataset months $m \in \{0, \dots, 7\}$ are mapped to 30-day operational billing cycles ($t = m \times 30$ days).
- **Application Arrival Time**: Continuous within-month arrival $t_{\text{apply}}$ is established across the chronological test partition (Months 6 and 7, $N=4,100$, 61 true frauds).
- **Verification Time**: $t_{\text{verified}} = t_{\text{apply}} + \Delta$ where $\Delta \in \{0, 7, 14, 30, 60\}$ days.
- **Label Eligibility Constraint**: At decision point $t = 210.0$ (the boundary between Month 6 and Month 7), only records with $t_{\text{verified}} \le 210.0$ are accessible for drift recalibration.

---

## 3. Label Availability Accounting at Decision Point (Month 6 $\to$ Month 7)

At time $t = 210.0$, the adaptive calibrator inspects eligible verified records to calibrate the threshold forward for Month 7:

| Scenario | Simulated Delay | Verified Labels from Month 6 ($N=2,163$) | Unverified Pending Labels | Total Eligible Labels in Buffer | Recalibration Triggered? | Threshold for Month 7 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: Static Threshold** | Frozen | N/A (Static) | N/A | 4,549 | No (Policy = Static) | `0.4409` |
| **B: TDTA (0d Lag)** | 0 days | **2,163 (100.0%)** | 0 (0.0%) | 4,549 | Yes (Drift detected) | `0.4537` |
| **C: TDTA (7d Lag)** | 7 days | **1,659 (76.7%)** | 504 (23.3%) | 4,045 | Yes (Drift detected) | `0.4536` |
| **D: TDTA (14d Lag)** | 14 days | **1,154 (53.4%)** | 1,009 (46.6%) | 3,540 | Yes (Drift detected) | `0.4541` |
| **E: TDTA (30d Lag)** | 30 days | **0 (0.0%)** | 2,163 (100.0%) | 2,386 (Month 5 only) | Yes (Fallback to Val) | `0.4409` |
| **F: TDTA (60d Lag)** | 60 days | **0 (0.0%)** | 2,163 (100.0%) | 2,386 (Month 5 only) | Yes (Fallback to Val) | `0.4409` |

### Key Observation on Buffer Mechanics:
- Under **0 to 14 days delay**, sufficient recent Month 6 labels have matured to recalibrate the threshold from `0.4409` upwards to `~0.4537` to maintain the 5% target FPR against concept drift.
- Under **30 and 60 days delay**, zero applications from Month 6 have matured by Day 210. The calibrator's safety check correctly detects that new test labels are unavailable, falling back safely to the Month 5 buffer without crashing or accessing future test labels. Consequently, the threshold retains the validation value (`0.4409`).

---

## 4. Comprehensive Performance Evaluation Across Delay Scenarios

Evaluated on the out-of-sample chronological test set (Months 6 & 7):

| Scenario | Final Active Threshold | ROC-AUC | PR-AUC | Standalone Precision | Standalone Recall | Standalone FPR | Realized Cost/App | Triage Recall [95% Wilson CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: Static Threshold** | `0.4409` | 0.8403 | 0.1255 | 0.1130 | 0.4426 | 0.0525 | $14.06 | **0.9344 [0.8432, 0.9742]** |
| **B: TDTA (0-Day)** | `0.4537` | 0.8403 | 0.1255 | 0.1111 | 0.4262 | 0.0515 | $14.39 | **0.9344 [0.8432, 0.9742]** |
| **C: TDTA (7-Day)** | `0.4536` | 0.8403 | 0.1255 | 0.1111 | 0.4262 | 0.0515 | $14.39 | **0.9344 [0.8432, 0.9742]** |
| **D: TDTA (14-Day)**| `0.4541` | 0.8403 | 0.1255 | 0.1111 | 0.4262 | 0.0515 | $14.39 | **0.9344 [0.8432, 0.9742]** |
| **E: TDTA (30-Day)**| `0.4409` | 0.8403 | 0.1255 | 0.1130 | 0.4426 | 0.0525 | $14.06 | **0.9344 [0.8432, 0.9742]** |
| **F: TDTA (60-Day)**| `0.4409` | 0.8403 | 0.1255 | 0.1130 | 0.4426 | 0.0525 | $14.06 | **0.9344 [0.8432, 0.9742]** |

### Metrics That Can vs Cannot Change:
1. **Invariance of Ranking Metrics (ROC-AUC & PR-AUC)**:
   ROC-AUC (`0.8403`) and PR-AUC (`0.1255`) depend solely on the frozen model's continuous predicted scores $p(x_i)$. Because delayed-label feedback adapts the decision threshold $\tau$ rather than retraining the underlying tree weights, **ranking metrics remain strictly identical across all delay scenarios**.
2. **Threshold-Dependent Metrics**:
   Precision, Recall, FPR, and Average Operational Cost respond directly to the adapted threshold. When TDTA adapts $\tau$ to `0.4537` (Scenarios B, C, D), FPR is successfully controlled closer to target (dropping from 0.0525 to 0.0515), at the cost of 1 additional false negative (Recall shifting from 0.4426 to 0.4262).

---

## 5. Hierarchical Triage Stability Under Delayed Feedback

| Scenario | % Auto-Approve | % Human Review | % Auto-Flag | Frauds Auto-Flagged | Frauds in Review | Frauds Erroneously Approved | Direct Recall | Triage Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: Static** | 37.17% | 57.00% | 5.83% | 27 | 30 | 4 | 44.3% | **93.4%** |
| **B: TDTA (0d)** | 37.17% | 57.02% | 5.80% | 27 | 30 | 4 | 44.3% | **93.4%** |
| **C: TDTA (7d)** | 37.17% | 57.02% | 5.80% | 27 | 30 | 4 | 44.3% | **93.4%** |
| **D: TDTA (14d)** | 37.17% | 57.02% | 5.80% | 27 | 30 | 4 | 44.3% | **93.4%** |
| **E: TDTA (30d)** | 37.17% | 57.00% | 5.83% | 27 | 30 | 4 | 44.3% | **93.4%** |
| **F: TDTA (60d)** | 37.17% | 57.00% | 5.83% | 27 | 30 | 4 | 44.3% | **93.4%** |

### Research Significance for Defense (Viva):
- **Uncompromised Fraud Capture**: Across all six scenarios, **Triage Recall remains constant at 93.4% [0.8432, 0.9742]**. 
- **The Conformal Shield**: Because conformal prediction sets $\Gamma_{0.05}(x)$ are calibrated on Month 5 and do not depend on unverified streaming labels, instances falling in the threshold adjustment region ($p_i \in [0.4409, 0.4541]$) are assigned ambiguous prediction sets $\{0, 1\}$.
- **Zero Fraud Escape Due to Delay**: The 4 frauds that slip into `AUTO_APPROVE` are identical across all scenarios (extremely clean applicant records with scores $< 0.10$). **Not a single additional fraud escaped due to verification delays of 7, 14, 30, or 60 days.**

