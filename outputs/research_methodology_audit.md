# Research-Methodology & Implementation Defensibility Audit: Mule Account Fraud Detection
**Project**: 1-Year MSc Data Science Capstone Research Project (VIT Major Project Review 3)  
**Repository**: `mule_newcode`  
**Dataset**: NeurIPS 2022 Bank Account Fraud Benchmark (`BaseDataset.csv`, 1,000,000 rows, Months 0–7)  
**Audit Scope**: Strict research validity, mathematical correctness, leakage vectors, and experimental defensibility.  
**Auditor**: Senior Machine Learning & Financial Systems Research Architect  
**Audit Status**: COMPLETED (Codebase inspected without modifications)

---

## Executive Summary & Research Assessment

An academic research thesis is evaluated by whether its empirical findings reflect true statistical phenomena under real-world operational constraints, rather than an uncoordinated assembly of algorithms. While `mule_newcode` possesses strong foundations (temporal splitting, expanding window cross-validation, and DeLong's statistical significance tests), this deep audit reveals **critical research discrepancies, severe test-label leakage in the newly hooked ablation/cost modules, and conceptual conflations between uncertainty calibration and operational triage** that would be immediately penalized by an expert academic review panel.

### Summary of Audit Findings by Severity

| Severity Level | Count | Primary Impact Areas |
| :--- | :---: | :--- |
| **CRITICAL** | **3** | Direct test-label leakage during cost threshold optimization; Degenerate/Copy-pasted ablation benchmark (Full Framework == Cost Opt); Violations of exchangeability in claimed finite-sample conformal guarantees under temporal drift. |
| **HIGH** | **4** | Conformal prediction disconnected from decision/cost pipeline; Conflation of nominal target FPR with realized out-of-distribution FPR in TDTA; Target/feature leakage in graph feature proxies across time; Adversarial evasion perturbation unboundedness and impossible record generation. |
| **MEDIUM** | **4** | Approximated normal CI instead of exact Wilson score interval for rare demographic slices; Inconsistent cost matrix definitions across scripts ($C_{\text{FN}}=\$1500$ vs $\$3000$ vs $\$5000$); Periodic retraining missing verification label lag in CV loops; Misleading reporting of 2% sample results as full benchmark metrics. |
| **LOW** | **2** | Hardcoded sentinel value handling; Redundant metric logging across disparate CSV tables. |

---

## Detailed Investigation Across 13 Audit Questions

### 1. Temporal Split Consistency Across Ablation Configurations
- **Files & Functions**: `src/train.py:main()`, `src/ablation.py:run_ablation_study()`, `src/config.py`.
- **Exact Implementation Behaviour**:
  In `src/train.py`, data is partitioned into:
  - Training: Months 0–4 ($N \approx 677,000$)
  - Validation: Month 5 ($N \approx 119,000$)
  - Test: Months 6–7 ($N \approx 204,000$)
  When `src/ablation.py:run_ablation_study` is invoked from `train.py:525`, it receives:
  `y_true = y_te`, `y_proba = probas[best]`, `y_val = y_va`, and `proba_val = models[best].predict_proba(X_va)[:, 1]`.
  All 5 ablation configurations evaluate the identical test array (`y_te`, `probas[best]`).
- **Academic Assessment**: **CORRECT**.
- **Why It Matters**: Cross-configuration comparability requires that the underlying evaluation slice and model weights remain completely invariant across configurations. This is satisfied.
- **Recommended Correction**: None for split consistency.

---

### 2. Decision Threshold Selection & Test-Label Leakage
- **Files & Functions**: 
  - `src/ablation.py:run_ablation_study()` (Lines 66–75)
  - `src/review3_enhancements.py:evaluate_multi_scenario_cost()` (Lines 9–37)
  - `src/train.py:main()` (Lines 378–392, Lines 398–399)
  - `src/evaluation.py:optimize_cost_threshold()` (Lines 62–81)
- **Exact Implementation Behaviour**:
  In `src/ablation.py`, line 67:
  ```python
  c_res = ev.optimize_cost_threshold(y_true, y_proba, config.COST_FN, config.COST_FP, config.COST_TP)
  thr_cost = c_res["optimal_threshold"]
  ```
  Here, `y_true` passed into `optimize_cost_threshold` is `y_te` (**the final test ground truth labels**).
  Similarly, in `src/train.py`:
  ```python
  c_opt = ev.optimize_cost_threshold(y_te, p, cost_fn=config.COST_FN, cost_fp=config.COST_FP)
  df_scenarios = evaluate_multi_scenario_cost(y_te, probas[best])
  ```
  And in `src/review3_enhancements.py`:
  ```python
  def evaluate_multi_scenario_cost(y_true, proba):
      # Finds threshold by minimizing cost directly on y_true (test labels)
  ```
- **Academic Assessment**: **CRITICAL ERROR (Direct Test-Label Leakage / Cheating)**.
- **Why It Matters**:
  Optimizing decision thresholds directly on test-period labels violates the fundamental rule of machine learning evaluation. In an operational bank, future test labels do not exist when setting decision thresholds. Selecting $\tau^*$ on test labels inflates test-set performance, understates real-world financial loss, and renders the cost savings claim invalid in a peer-reviewed or thesis defence setting.
- **Recommended Correction**:
  Thresholds must be calibrated strictly on the **Validation set** (`y_va`, `proba_val` from Month 5). That frozen threshold must then be applied forward out-of-sample to `y_te`. Cost curves on the test set should be reported at the validation-selected threshold, not at the post-hoc test-minimized threshold.

---

### 3. Identity of "Full Proposed Framework" vs "Baseline + Cost Optimization"
- **Files & Functions**: `src/ablation.py:run_ablation_study()` (Lines 140–154)
- **Exact Implementation Behaviour**:
  Lines 143–154 state:
  ```python
  # 5. Full Proposed Framework (Cost-Aware + Conformal Triage + Drift Safeguards)
  results.append({
      "configuration": "5. Full Proposed Framework",
      "components_active": "Cost-Optimal + TDTA Drift Guard + Conformal Triage",
      "threshold": round(thr_cost, 4),
      "precision": round(prec_c, 4),
      "recall": round(rec_c, 4),
      "fpr": round(fpr_c, 4),
      "operational_cost_per_app": round(cost_c, 2),
      "human_review_workload": round(float(np.mean(pred_cost)), 4),
      "uncertainty_guarantee": f"{cov['empirical_coverage']*100:.1f}% Empirical Coverage",
      "primary_benefit": "Simultaneous financial loss minimization & rigorous uncertainty-bounded triage"
  })
  ```
- **Academic Assessment**: **CRITICAL FLAW (Ablation Degeneracy / Illusory Integration)**.
- **Why It Matters**:
  The code literally copy-pastes the metrics from configuration 2 (`thr_cost`, `prec_c`, `rec_c`, `fpr_c`, `cost_c`, `pred_cost`) and attaches the label "Full Proposed Framework". TDTA threshold adaptation and conformal triage are not actually executing or modifying the decisions in configuration 5. An academic examiner will immediately see that row 5 and row 2 are identical and conclude that the "unified framework" is illusory.
- **Recommended Correction**:
  Define a genuine hierarchical decision policy:
  1. High confidence fraud ($p \ge \tau_{\text{cost\_optimal}}$ or conformal singleton $\{1\}$): Auto-Decline ($C_{\text{TP}}$ friction).
  2. Provably legitimate ($p < \tau_{\text{safe}}$ and conformal singleton $\{0\}$): Auto-Approve ($C_{\text{TN}} = 0$).
  3. Non-singleton conformal set $\{0, 1\}$ or drift-flagged region: Route to Human Investigation ($C_{\text{investigate}}$).
  Evaluate this combined decision rule explicitly to report distinct, empirically grounded metrics.

---

### 4. Integration of Conformal Prediction in Final Decisions
- **Files & Functions**: `src/train.py:504–510`, `src/conformal.py:predict_sets()`, `src/ablation.py:113–138`.
- **Exact Implementation Behaviour**:
  In `src/train.py`, conformal prediction is executed as an isolated reporting block:
  ```python
  cp = ConformalFraudPredictor(alpha=0.05)
  cp.calibrate(y_va, models[best].predict_proba(X_va)[:, 1])
  cp_res = cp.evaluate_coverage(y_te, probas[best])
  ```
  The predictions `cp_res` are dumped to `outputs/conformal_prediction_summary.csv` and never passed to the cost-optimization model, the final decision threshold, or the regulatory audit card.
- **Academic Assessment**: **HIGH SEVERITY (Cosmetic / Disconnected Component)**.
- **Why It Matters**:
  Claiming an "uncertainty-guided fraud decision architecture" when conformal sets are merely calculated in parallel and discarded creates an indefensible paper claim.
- **Recommended Correction**:
  Formally integrate the conformal output into the bank triage queue:
  - Total operational cost must explicitly include the investigation cost ($C_{\text{investigation}} = \$30$) for all applications assigned $\{0, 1\}$ (Human Review).
  - Compare this triage cost against the standard binary decision threshold without conformal prediction.

---

### 5. Mathematical Correctness of Conformal Coverage Calculation
- **Files & Functions**: `src/conformal.py:ConformalFraudPredictor` (Lines 12–26, Lines 54–70).
- **Exact Implementation Behaviour**:
  ```python
  # Calibration
  s_0 = p[y == 0]
  q_level_0 = np.ceil((n_0 + 1) * (1.0 - self.alpha)) / n_0
  self.q_neg_ = float(np.quantile(s_0, min(1.0, q_level_0)))
  s_1 = 1.0 - p[y == 1]
  q_level_1 = np.ceil((n_1 + 1) * (1.0 - self.alpha)) / n_1
  self.q_pos_ = float(np.quantile(s_1, min(1.0, q_level_1)))

  # Inclusion rule
  include_0 = p <= self.q_neg_
  include_1 = (1.0 - p) <= self.q_pos_

  # Coverage check
  if yi == 0 and 'Legitimate' in set_str: covered[i] = True
  elif yi == 1 and 'Fraud' in set_str: covered[i] = True
  ```
- **Academic Assessment**: **QUESTIONABLE / PARTIALLY FLAWED FORMULATION**.
- **Why It Matters**:
  The formulation uses non-conformity scores $S_i(0) = p_i$ and $S_i(1) = 1 - p_i$ and calibrates separate quantiles $q_{\text{neg}}$ and $q_{\text{pos}}$ marginally at $1 - \alpha$.
  However, in Mondrian (class-conditional) conformal prediction:
  - If both classes calibrate at $1 - \alpha$, the overall marginal coverage is guaranteed to be at least $1 - \alpha$, but the two-tailed inclusion creates massive overlapping prediction sets $\{0, 1\}$ ($57.1\%$ in your run).
  - More importantly, when $n_1$ is small (e.g., in a 2% sample with 26 frauds in Month 5), $(n_1 + 1)(1 - \alpha) / n_1$ exceeds 1.0, forcing `min(1.0, q_level_1)` to 1.0, which clips $q_{\text{pos}}$ to `np.quantile(s_1, 1.0) = max(s_1)`.
- **Recommended Correction**:
  Adopt standard split conformal prediction using standard non-conformity measure $S_i = 1 - \hat{P}(Y_i = y_i \mid X_i)$ or calibrate a single conformal threshold $\hat{q}$ at $(1 - \alpha)(1 + 1/n)$, ensuring strict adherence to Vovk et al. (2005).

---

### 6. Validity of Finite-Sample Coverage Assumptions Under Temporal Drift
- **Files & Functions**: `src/conformal.py`, `src/train.py:504–510`.
- **Exact Implementation Behaviour**:
  Calibration is performed on Month 5:
  `cp.calibrate(y_va, models[best].predict_proba(X_va)[:, 1])`
  Testing is performed on Months 6 & 7:
  `cp.evaluate_coverage(y_te, probas[best])`
  In `train.py:504`, the log prints:
  `[Conformal Prediction: Finite-Sample Coverage Guarantees]`
- **Academic Assessment**: **INCORRECT CLAIM / THEORETICAL VIOLATION**.
- **Why It Matters**:
  **Conformal prediction finite-sample validity relies strictly on the Exchangeability (or i.i.d.) assumption** between calibration and test data ($Z_1, \dots, Z_n, Z_{n+1} \sim P$).
  In this dataset, your own drift detector proves that `velocity_6h` ($\text{PSI} = 1.1424$, $p=0.000$) and `credit_risk_score` ($\text{PSI}=0.2710$) undergo **statistically significant temporal concept drift** between Month 5 and Months 6–7.
  Under distribution drift ($P_{\text{calib}} \ne P_{\text{test}}$), standard conformal prediction **loses its finite-sample coverage guarantee**. Claiming a "guarantee" on drifted test data is a known theoretical fallacy that reviewers frequently penalize.
- **Recommended Correction**:
  Frame conformal prediction as **empirical risk-bounded triage under temporal shift**, cite Tibshirani et al. (2019) (*Conformal Prediction Under Covariate Shift*), and evaluate whether empirical coverage degrades month-by-month (Month 6 vs Month 7).

---

### 7. Information Horizon in TDTA (Temporal Drift-Aware Threshold Adaptation)
- **Files & Functions**: `src/tdta_experiment.py:run_sequential_policy()` (Lines 128–250).
- **Exact Implementation Behaviour**:
  - For month $m$:
    - Decisions are made using `active_thr_for_m = current_threshold` established *prior* to observing test period $m$.
    - Drift on month $m$ is measured using unlabeled features $X_m$ and model probabilities $p_m$.
    - The adaptation decision for month $m+1$ restricts historical labels to $mon \le m - \text{label\_delay}$.
    - Updated threshold is applied forward to $m+1$, never retroactively to $m$.
- **Academic Assessment**: **CORRECT & METHODOLOGICALLY RIGOROUS**.
- **Why It Matters**: The prequential protocol in `src/tdta_experiment.py` maintains an air gap against future labels.
- **Recommended Correction**: Preserve this design, but ensure that `src/train.py` and `src/ablation.py` call this exact policy rather than a simplified mock.

---

### 8. Actual False Positive Rate (FPR) Control vs Alert Volume / Quantile Matching
- **Files & Functions**: `src/adaptive_threshold.py:calibrate()` (Lines 145–150).
- **Exact Implementation Behaviour**:
  ```python
  neg_scores = calib_p[calib_y == 0]
  new_thr = float(np.quantile(neg_scores, 1.0 - self.target_fpr, method="linear"))
  ```
- **Academic Assessment**: **QUESTIONABLE / PARTIALLY CONFLATED**.
- **Why It Matters**:
  The calibrator computes the $(1 - \text{target\_fpr})$ quantile strictly over **historical negative samples** (`calib_y == 0`).
  This guarantees that on the *historical calibration window*, empirical FPR is $\le 5\%$.
  However, when this threshold is applied to future test month $m$:
  1. If feature drift or score calibration drift occurs, the scores of negative instances shift.
  2. The realized test FPR will deviate from $5\%$ (in your 2% smoke test run, realized FPR on test was $5.2\%$).
  In prose, this must be distinguished from unconditional score quantile matching (which fixes alert volume $\hat{P}(\hat{Y}=1) = \alpha$, regardless of true labels).
- **Recommended Correction**:
  Explicitly document in the thesis that TDTA controls the *historical calibration FPR constraint*, while out-of-sample realized test FPR remains subject to generalization error under non-stationary covariate shift.

---

### 9. Periodic Retraining & Verification Lag Integrity
- **Files & Functions**: 
  - `src/delayed_feedback.py:get_training_slice_under_delay()`
  - `src/delayed_feedback_experiment.py:run_delayed_benchmark()`
  - `src/train.py:temporal_cv()` (Lines 267–335)
- **Exact Implementation Behaviour**:
  In `src/delayed_feedback_experiment.py`, verification delay is modeled cleanly:
  ```python
  train = get_training_slice_under_delay(df, current_month=6, delay_months=lag)
  ```
  However, in `src/train.py:temporal_cv()` (the primary 5-fold cross-validation loop):
  ```python
  for test_m in config.CV_TEST_MONTHS:
      tr_idx = df[df[config.TIME_COL] < test_m].index
      te_idx = df[df[config.TIME_COL] == test_m].index
  ```
- **Academic Assessment**: **HIGH SEVERITY (Idealized Retraining in Primary CV Loop)**.
- **Why It Matters**:
  In `train.py:temporal_cv`, when testing on month $m$, training is performed on all applications with `month < m`.
  This assumes that fraud labels from month $m-1$ are $100\%$ verified on day 1 of month $m$ (Zero Verification Lag, $L=0$).
  In banking, confirmed fraud or chargebacks take 30 to 90 days ($L \in [1, 2, 3]$) to mature. Retraining on month $m-1$ labels without delay is an idealized assumption that overstates retraining responsiveness.
- **Recommended Correction**:
  Parameterize `temporal_cv` with `label_delay_months=1` so that training on month $m$ uses data up to $m - 1 - \text{delay}$. Compare $L=0$ vs $L=1$ in the CV summary table.

---

### 10. Target, Preprocessing, and Temporal Leakage Across Engineered Features
- **Files & Functions**: `src/features.py:FeaturePreprocessor` (Lines 44–138).
- **Exact Implementation Behaviour**:
  The 86 features were audited for leakage vectors:
  - **Categorical Encoders**: Categories are strictly fitted on `TRAIN_MONTHS` (`fit()` in Line 141). Out-of-vocabulary categories in test are mapped to zeros. (Zero leakage).
  - **Temporal Baseline Drift Z-Scores**: Historical mean and standard deviation for `velocity_6h`, `proposed_credit_limit`, and `credit_risk_score` are frozen during `fit()` on training data (Lines 147–150). (Zero leakage).
  - **Graph Degree Proxies**:
    Features like `device_sharing_density = device_deg * dob_deg` and `graph_hub_degree_proxy` use row-level columns already provided in the raw dataset (`device_distinct_emails_8w`, `date_of_birth_distinct_emails_4w`, `bank_branch_count_8w`, `zip_count_4w`).
- **Academic Assessment**: **QUESTIONABLE RESEARCH CLAIM (Proxy vs True Dynamic Graph)**.
- **Why It Matters**:
  There is **no data leakage** in the pipeline because the features are row-level transformations of pre-aggregated window features provided by the BAF authors.
  However, there is a **conceptual/academic reporting issue**:
  The thesis cannot claim to have "constructed a live dynamic graph across 1,000,000 nodes". These features are algebraic combinations of 8-week entity counters already aggregated in the BAF schema. Presenting them as a "newly constructed multiplex bipartite graph" risks severe pushback from graph ML examiners.
- **Recommended Correction**:
  Accurately designate these features as **"Ego-Network Bipartite Degree Proxies and Higher-Order Collision Interactions"** derived from the benchmark's historical entity counters.

---

### 11. Mathematical Validity of Cost Calculations
- **Files & Functions**:
  - `src/config.py:85–88`
  - `src/evaluation.py:compute_cost_matrix()` (Lines 50–59)
  - `src/review3_enhancements.py:evaluate_multi_scenario_cost()` (Lines 10–26)
  - `src/ablation.py:run_ablation_study()` (Line 51, 98, 124)
- **Exact Implementation Behaviour**:
  In `config.py` and `evaluation.py`:
  $$\text{Cost} = \frac{\text{FN} \times C_{\text{FN}} + \text{FP} \times C_{\text{FP}} + \text{TP} \times C_{\text{TP}} + \text{TN} \times C_{\text{TN}}}{N}$$
  Where:
  - $C_{\text{FN}} = \$1,500$ (Loss exposure)
  - $C_{\text{FP}} = \$30$ (Manual compliance investigation)
  - $C_{\text{TP}} = \$10$ (Automated step-up KYC verification)
  - $C_{\text{TN}} = \$0$ (Frictionless auto-approval)
  However, in `src/review3_enhancements.py:evaluate_multi_scenario_cost()`:
  $$\text{Cost}_{\text{multi}} = \frac{\text{FN} \times C_{\text{FN}} + \text{FP} \times C_{\text{FP}}}{N}$$
  Here, $C_{\text{TP}}$ is omitted entirely ($C_{\text{TP}} = 0$).
- **Academic Assessment**: **HIGH SEVERITY (Inconsistent Formula Across Modules)**.
- **Why It Matters**:
  Comparing the numbers in `cost_optimization_results.csv` ($C_{\text{TP}}=\$10$) against `cost_scenarios_multi.csv` ($C_{\text{TP}}=\$0$) creates numerical inconsistency. If automated step-up friction costs \$10, omitting it in the multi-scenario module distorts the relative cost rankings when TP is large.
- **Recommended Correction**:
  Standardize `evaluate_multi_scenario_cost()` to include $C_{\text{TP}} = \$10$ consistently across all cost modules.

---

### 12. Realism of Adversarial Evasion Perturbations
- **Files & Functions**: `src/evaluation.py:evaluate_adversarial_robustness()` (Lines 249–299).
- **Exact Implementation Behaviour**:
  ```python
  X_perturbed = X_te.copy()
  for col in target_cols:
      sigma = float(X_te[col].std()) if X_te[col].std() > 0 else 1.0
      noise = np.random.normal(0, eps * sigma, size=len(X_te))
      # Fraudsters dampen suspicion (attenuate velocity/distinct count)
      X_perturbed[col] = (X_perturbed[col] * (1.0 - eps) + noise).clip(lower=0)
  ```
- **Academic Assessment**: **HIGH SEVERITY (Generates Statistically Impossible Records)**.
- **Why It Matters**:
  1. **Discrete Variables Made Continuous**: `device_distinct_emails_8w` is an integer count ($0, 1, 2, \dots$). Adding Gaussian noise creates non-integer continuous values (e.g., $1.432$ distinct emails), which a production schema validator or tree split would detect as anomalous.
  2. **Inter-Feature Consistency Broken**:
     If `velocity_6h` is perturbed downwards, it can become larger than `velocity_24h` or smaller than $0$, violating logical constraints ($v_{6h} \le v_{24h} \le v_{4w}$).
  3. **Engineered Feature Decoupling**:
     `X_te` contains both raw features (`velocity_6h`) and engineered derivatives (`velocity_ratio_6h_24h`, `velocity_spike_index`). Perturbing `X_te` directly perturbs the raw column **without re-computing the engineered features**, creating internally inconsistent input vectors.
- **Recommended Correction**:
  Perturb raw tabular features in a copy of the dataframe `df_te`, apply feature constraints ($v_{6h} \le v_{24h}$, integer rounding on counts), and pass the perturbed dataframe through `preprocessor.transform(df_perturbed)` so all 86 features re-derive consistently.

---

### 13. Statistical Rigor of Fairness Metrics & Wilson Confidence Intervals
- **Files & Functions**: `src/review3_enhancements.py:evaluate_enhanced_fairness()` (Lines 61–95).
- **Exact Implementation Behaviour**:
  In Lines 80–84:
  ```python
  approval_rate = float(np.mean(pred_g == 0))
  p_est = approval_rate
  se = np.sqrt(max(0.0, p_est * (1.0 - p_est)) / n_g)
  ci_low = max(0.0, p_est - 1.96 * se)
  ci_high = min(1.0, p_est + 1.96 * se)
  ```
- **Academic Assessment**: **HIGH SEVERITY (Misnamed Normal Approximation, Not Wilson Score Interval)**.
- **Why It Matters**:
  The code claims to implement **Wilson score confidence intervals**, but the formula implemented is the **Wald (Normal) approximation interval**:
  $$\hat{p} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$
  For small sample sizes ($N=31$ for `Age 65+`) or extreme proportions ($\hat{p} \approx 0.993$ for `Age <25`), the Wald interval suffers from severe **zero-coverage and overshoot pathologies** (Agresti & Coull, 1998). An academic examiner in Data Science or Statistics will immediately catch that the formula is Wald, not Wilson.
- **Recommended Correction**:
  Implement the true **Wilson Score Interval**:
  $$CI_{\text{Wilson}} = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

---

## Complete Audit Summary Table

| Issue # | Area | Implementation | Status | Severity | Recommended Fix |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | Temporal Split | `train.py`, `ablation.py` | Correct | Valid | Retain current chronological split |
| **2** | Threshold Selection | `ablation.py:67`, `review3_enhancements.py:26` | Test labels used to tune thresholds | **CRITICAL** | Calibrate threshold on Val (Mo 5); test out-of-sample |
| **3** | Ablation Study | `ablation.py:143–154` | Full Framework is a clone of Cost-Opt | **CRITICAL** | Formulate true hierarchical policy combining cost + conformal |
| **4** | Conformal Integration | `train.py:504–510` | Run as detached report, ignored in decisions | **HIGH** | Use conformal triage sets $\{0, 1\}$ directly in queue cost model |
| **5** | Conformal Math | `conformal.py:17–25` | Margin clipping issues when $n_1$ is small | **MEDIUM** | Use standard non-conformity score $S_i = 1 - \hat{p}(y_i)$ |
| **6** | Coverage Guarantees | `train.py:504`, `README.md` | Claiming finite-sample guarantees under drift | **HIGH** | Frame as empirical risk bounds under non-exchangeable shift |
| **7** | TDTA Information Horizon | `tdta_experiment.py:128–250` | Unlabeled features used for drift; labels lagged | Correct | Maintain prequential protocol |
| **8** | Target FPR Control | `adaptive_threshold.py:145` | Quantile matching on historical negatives | Questionable | Acknowledge out-of-sample realization error under drift |
| **9** | Periodic Retraining | `train.py:temporal_cv:267` | Uses $L=0$ label lag (unrealistic in banking) | **HIGH** | Add $L=1$ month verification lag to temporal CV loop |
| **10** | Graph Feature Leakage | `features.py:70–107` | Row-level entity counter products | Questionable | Re-label as higher-order ego-network collision proxies |
| **11** | Cost Consistency | `review3_enhancements.py:25` | Omits $C_{\text{TP}}=\$10$ present in `evaluation.py` | **MEDIUM** | Unify cost formula ($C_{\text{FN}}, C_{\text{FP}}, C_{\text{TP}}, C_{\text{TN}}$) across all files |
| **12** | Adversarial Realism | `evaluation.py:279–284` | Perturbs raw matrix directly; creates invalid records | **HIGH** | Perturb raw dataframe with bounds; re-transform with preprocessor |
| **13** | Fairness CIs | `review3_enhancements.py:82` | Wald interval implemented, claimed as Wilson | **HIGH** | Implement exact Wilson Score formula |

---

## Prioritized Implementation Roadmap (Phase-by-Phase Plan)

To transition this repository from an MSc 8/10 to a defensible 9.5/10 for Review 3, execute these prioritized corrections:

```
Phase 1: Leakage Elimination (Highest Academic Risk)
  ├── 1. Fix threshold optimization in ablation.py & review3_enhancements.py to calibrate strictly on Month 5
  └── 2. Apply calibrated threshold forward to Months 6–7 out-of-sample

Phase 2: True Hierarchical Ablation Architecture
  ├── 1. Implement authentic Full Framework decision logic:
  │      • High confidence fraud -> Auto Decline ($C_{\text{TP}}$)
  │      • Conformal singleton {0} -> Auto Approve ($C_{\text{TN}}$)
  │      • Ambiguous set {0, 1} or drift trigger -> Human Review ($C_{\text{FP}}$)
  └── 2. Re-run ablation.py to produce non-identical, defensible rows

Phase 3: Statistical Rigor & Theory Compliance
  ├── 1. Replace Wald confidence intervals with exact Wilson Score intervals in evaluate_enhanced_fairness()
  ├── 2. Re-frame conformal prediction claims from "Finite-Sample Guarantees" to "Covariate-Shift Evaluated Uncertainty Triage"
  └── 3. Standardize cost definitions ($C_{\text{TP}}=\$10$) across all modules

Phase 4: Realistic Adversarial Robustness
  ├── 1. Enforce physical constraints: integer rounding on count features, $v_{6h} \le v_{24h}$
  └── 2. Pass perturbed raw dataframes through FeaturePreprocessor.transform()
```

