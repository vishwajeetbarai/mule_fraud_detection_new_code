# Phase 6 Final Validation Summary & Viva Defense Guide

**Project**: Mule Account Fraud Detection (SET Major Project)  
**Deliverable**: Final Validation Summary & Viva Examination Guide  
**Scale**: Full 1,000,000 Dataset Run  
**Status**: APPROVED  

---

## 1. Quick Reference: Core Metrics on Full 1,000,000 Records

- **Total Test Applications**: $205,011$ (Months 6 and 7).
- **Total Test Frauds**: $2,878$ (Prevalence: $1.4038\%$).
- **Primary Model Selected**: LightGBM (Best iteration: 240, train time: 25s).
- **Test ROC-AUC**: `0.8869` [0.8809, 0.8930] (Stratified Bootstrap, 500 reps).
- **Test PR-AUC**: `0.1869` [0.1739, 0.2024] (Stratified Bootstrap, 500 reps).
- **TPR @ 1% FPR**: `0.2450` [0.2272, 0.2594] | **TPR @ 5% FPR**: `0.5309` [0.5120, 0.5501].
- **Standalone Precision / Recall**: Precision = $12.40\%$, Recall = $54.83\%$ (at $\tau_{\text{tdta}} = 0.7569$).
- **Hierarchical Triage Fraud Capture**: **94.02%** ($[0.9310, 0.9483]$).
- **Fraud Escape Rate**: **5.98%** ($172$ out of $2,878$ frauds).
- **Automated Approval**: **55.36%** ($113,493$ applications).
- **Human Review**: **38.44%** ($78,796$ applications).
- **Direct Automated Flag**: **6.21%** ($12,722$ applications).

---

## 2. Answers to Essential Viva Questions

### Q1: What is the primary novelty of this project over conventional fraud classifiers?
> *"Most literature applies a single binary threshold to classifier probabilities, forcing an impossible trade-off between massive customer friction (high false alarms) and catastrophic financial losses (escaped fraud). Our novelty lies in a three-tier hierarchical triage policy that combines: (1) validation-selected cost-optimal thresholds, (2) temporal drift-aware threshold adaptation (TDTA) to contain false alarms under non-stationary shifts, and (3) split conformal prediction uncertainty sets. Rather than treating uncertainty as a nuisance, we route ambiguous predictions ($\Gamma_{0.05} = \{0, 1\}$) to human review. On the full 1,000,000 dataset, this lifts overall fraud capture from 54.83% to 94.02% while allowing 55.36% of applicants to pass frictionlessly."*

### Q2: Did you prevent data leakage during threshold selection?
> *"Yes, absolutely. Thresholds were selected exclusively on Month 5 validation data ($\tau_{\text{cost}} = 0.5373$, $\tau_{\text{tdta}} = 0.7569$) and frozen. The out-of-sample test set (Months 6 and 7, containing 205,011 records) was scored strictly with frozen parameters. Test labels were never touched during model training, feature selection, or policy calibration."*

### Q3: How did the model behave under delayed fraud verification?
> *"In live banking, fraud labels are discovered with 30–90 days delay. When we evaluated our framework under 7, 14, 30, and 60 days of feedback delay, the buffer gating logic prevented lookahead leakage. Under 30 and 60 days lag, where Month 6 labels had not matured, the system safely fell back to the Month 5 buffer. Overall triage fraud capture remained completely steady at 94.02% across all delay scenarios, proving that conformal uncertainty provides defense-in-depth even when feedback is delayed."*

### Q4: Why did older applicants experience lower approval rates in your fairness audit?
> *"In our audit, applicants aged 35–50 and 50–65 had approval rates below the 80% Four-Fifths threshold relative to applicants under 25. However, this is driven by the synthetic dataset's underlying distributions: ground-truth fraud prevalence rises from 0.64% in the <25 group to 5.00% in the 65+ group. When looking at true fraudsters, equal opportunity was preserved: over 95% of true fraud cases in older cohorts were intercepted by the triage framework."*

---

## 3. Final Sign-Off

- **Pytest Suite**: 47 passed in 75.06s.
- **Dataset**: Full 1,000,000 rows executed and verified.
- **Phase 6 Status**: **APPROVED**.

