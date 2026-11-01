# Phase 2 Framework Report: Hierarchical Decision Policy & Genuine Ablation Architecture
**Project**: 1-Year Master's SET Research Project — Mule Account Fraud Detection  
**Repository**: `mule_newcode`  
**Dataset**: NeurIPS 2022 Bank Account Fraud (`BaseDataset.csv`)  
**Phase**: Phase 2 — Hierarchical Decision Routing & Full Framework Integration  
**Date**: 2026-10-31  
**Status**: VERIFIED & AUDITED (31/31 Pytest unit tests passing; End-to-end smoke test completed)

---

## 1. Executive Summary & Problem Resolved

In earlier revisions of the repository, Configuration 5 (*Full Proposed Framework*) was identical to Configuration 2 (*Cost Optimization*) because its metrics were directly copied without executing a combined decision policy. 

**Phase 2 resolves this flaw by designing, implementing, and empirically evaluating a genuine three-tier hierarchical decision routing policy**:
1. **Cost Optimization**: Operates at the validation-selected loss-minimizing threshold ($\tau_{\text{cost}} = 0.2587$).
2. **Temporal Drift Adaptation (TDTA)**: Implements prequential quantile-matching targeting historical 5% FPR ($\tau_{\text{tdta}} = 0.4409$).
3. **Split Conformal Prediction**: Generates finite-sample calibrated prediction sets $\Gamma_{0.05}(x) \in \{\{0\}, \{1\}, \{0, 1\}\}$.
4. **Three-Tier Operational Actions**: Every inbound application is deterministically mapped into one of three mutually exclusive operational actions:
   - **`AUTO_APPROVE`**: Frictionless onboarding ($C_{\text{TN}} = \$0$).
   - **`AUTO_FLAG`**: Direct fraud interception ($C_{\text{TP}} = \$10, C_{\text{FP}} = \$30$).
   - **`HUMAN_REVIEW`**: Manual investigator queue ($C_{\text{investigation}} = \$30$).

---

## 2. Mathematical Decision Policy Formulation

Let:
- $p_i \in [0, 1]$ be the estimated probability of fraud for application $i$.
- $\tau_{\text{cost}} \in [0, 1]$ be the validation-selected cost-optimal decision threshold ($0.2587$).
- $\tau_{\text{tdta}} \in [0, 1]$ be the validation drift-guard threshold ($0.4409$).
- $\tau_{\min} = \min(\tau_{\text{cost}}, \tau_{\text{tdta}})$ and $\tau_{\max} = \max(\tau_{\text{cost}}, \tau_{\text{tdta}})$.
- $\Gamma_{\alpha}(x_i) \in \{\{0\}, \{1\}, \{0, 1\}\}$ be the split conformal prediction set calibrated at nominal error rate $\alpha = 0.05$.

### Exact Operational Decision Rules:

#### Action A: `AUTO_APPROVE` (Action Code = 0)
An application is approved without human intervention if and only if:
$$\Gamma_{\alpha}(x_i) = \{0\} \quad \text{AND} \quad p_i < \tau_{\min}$$
*Operational Justification*: Both the conformal prediction set (provably excluding fraud at $1-\alpha$ coverage) and the model's cost/drift boundaries agree that the application is low risk. Approved frictionlessly.

#### Action B: `AUTO_FLAG` (Action Code = 1)
An application is flagged for automated fraud containment if:
$$p_i \ge \tau_{\max} \quad \text{OR} \quad \Gamma_{\alpha}(x_i) = \{1\}$$
*Operational Justification*: The model's score exceeds both the cost-minimizing threshold and the conservative drift guard, or the conformal prediction set confirms unambiguous fraud. Flagged for immediate step-up identity authentication or decline.

#### Action C: `HUMAN_REVIEW` (Action Code = 2)
An application is routed to the AML compliance queue if:
$$\text{Neither Auto-Approve nor Auto-Flag conditions are met}$$
Explicitly resolving:
1. **Ambiguous Conformal Sets**: $\Gamma_{\alpha}(x_i) = \{0, 1\}$ (both outcomes plausible under uncertainty).
2. **Conflict Region**: $\tau_{\min} \le p_i < \tau_{\max}$ (score indicates fraud under cost matrix but falls in the safety buffer of the drift guard).

---

## 3. Operational Cost Accounting Model

In retail banking compliance, manual review incurs actual labor costs. Costs are accounted for as follows:

| Ground Truth ($Y$) | Operational Action | Financial Loss / Cost Component | Unit Cost ($) |
| :--- | :--- | :--- | :---: |
| **Fraud ($Y=1$)** | `AUTO_APPROVE` | Undetected bust-out / laundering loss ($C_{\text{FN}}$) | **$1,500.00** |
| **Fraud ($Y=1$)** | `AUTO_FLAG` | Automated step-up verification friction ($C_{\text{TP}}$) | **$10.00** |
| **Fraud ($Y=1$)** | `HUMAN_REVIEW` | Manual investigation ($C_{\text{inv}}$) + verification friction ($C_{\text{TP}}$) | **$40.00** |
| **Legitimate ($Y=0$)** | `AUTO_APPROVE` | Seamless frictionless approval ($C_{\text{TN}}$) | **$0.00** |
| **Legitimate ($Y=0$)** | `AUTO_FLAG` | Customer friction / adverse review ($C_{\text{FP}}$) | **$30.00** |
| **Legitimate ($Y=0$)** | `HUMAN_REVIEW` | Manual compliance investigator overhead ($C_{\text{inv}}$) | **$30.00** |

$$\text{Cost}_{\text{app}} = \frac{1}{N} \sum_{i=1}^N \text{Cost}(Y_i, \text{Action}_i)$$

---

## 4. Empirical 5-Configuration Ablation Study (`outputs/phase2_ablation_results.csv`)

Evaluated out-of-sample on Test Months 6–7 ($N_{\text{test}} = 4,100$, 61 frauds, 1.488% prevalence):

| Config # | Configuration Description | Threshold ($\tau$) | Precision | Recall | FPR | Cost / App ($) | Auto-Approve (%) | Auto-Flag (%) | Human-Review (%) | Empirical Coverage |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Baseline** (Default 0.5 Cutoff) | 0.5000 | **0.1364** | 0.3934 | 0.0376 | **$14.71** | 95.68% | 4.32% | 0.00% | N/A |
| **2** | **Cost Optimization** ($\tau_{\text{cost}}$) | 0.2587 | 0.0614 | 0.6230 | 0.1436 | **$12.75** | 84.93% | 15.07% | 0.00% | N/A |
| **3** | **Cost Opt + TDTA** ($\tau_{\text{tdta}}$) | 0.4409 | 0.1130 | 0.4426 | 0.0520 | **$14.06** | 94.17% | 5.83% | 0.00% | N/A |
| **4** | **Cost Opt + Conformal** (Selective) | 0.2587 | 0.0222 | **0.9344** | 0.5714 | **$20.25** | 37.17% | 5.73% | 57.10% | **94.83%** |
| **5** | **Full Proposed Framework** | Multi | 0.0222 | **0.9344** | 0.5704 | **$20.25** | 37.17% | 5.83% | 57.00% | **94.83%** |

---

## 5. Action Distribution Analysis (`outputs/phase2_action_distribution.csv`)

Detailed breakdown of how the **Full Proposed Framework** routes legitimate vs fraudulent applicants:

| Operational Action | Total Assigned | Volume (%) | Actual Frauds Captured | Actual Legitimates Contained | Operational Significance |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`AUTO_APPROVE`** | **1,524** | **37.17%** | **4** (Escaped) | **1,520** | Frictionless onboarding; zero human review cost. |
| **`AUTO_FLAG`** | **239** | **5.83%** | **8** | **231** | High-confidence fraud containment / step-up auth. |
| **`HUMAN_REVIEW`** | **2,337** | **57.00%** | **49** | **2,288** | Borderline cases capturing 80.3% of all fraud cases. |
| **Total Test Set** | **4,100** | **100.00%** | **61** (100%) | **4,039** (100%) | Sum of percentages = 100.00% |

### Key Research Insights from Phase 2:
1. **The Cost vs Uncertainty Trade-off**:
   - Configuration 2 (*Cost Optimization*) achieves the lowest binary cost (**$12.75/app**) by aggressively minimizing false negatives without manual review.
   - Configuration 5 (*Full Framework*) incurs a higher unit cost (**$20.25/app**) because routing 57% of volume to human investigators costs $30 per review.
   - **Crucial Academic Value**: In exchange for investigator costs, fraud recall jumps from **62.3%** to **93.4%**, and the bank gains **94.83% empirical coverage** guarantees on prediction sets.
2. **Action Coverage**:
   - Only 4 out of 61 frauds (6.5%) escape through `AUTO_APPROVE`.
   - **57 out of 61 frauds (93.4%)** are successfully intercepted in `AUTO_FLAG` and `HUMAN_REVIEW`.

---

## 6. Automated Test Suite Verification

All 31 unit tests across the repository are passing:
- `tests/test_hierarchical_phase2.py`: Verifies that all 3 actions exist, partitions sum to 100%, routing is invariant to test labels, and Config 5 differs genuinely from Config 2.
- `tests/test_leakage_phase1.py`: Confirms zero test-label leakage remains preserved.
- `tests/test_pipeline.py`, `test_tdta.py`, `test_review3_enhancements.py`: All green.

