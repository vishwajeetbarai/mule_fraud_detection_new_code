# Phase 5 Delayed-Label Feedback Methodology Specification

**Project**: Mule Account Fraud Detection (SET Major Project)  
**Dataset**: NeurIPS 2022 Bank Account Fraud (BAF) Benchmark Suite (`BaseDataset.csv`)  
**Scope**: Realistic Delayed Fraud Verification, Prequential TDTA & Hierarchical Triage Stability  

---

## 1. Research Motivation & The Delayed-Feedback Dilemma

In real-world retail banking and digital account opening:
1. **Decision Time**: The automated risk scoring model and triage engine must make a decision at application time ($t = t_{\text{apply}}$), routing applicants to `AUTO_APPROVE`, `HUMAN_REVIEW`, or `AUTO_FLAG`.
2. **Label Realization Lag**: The ground-truth fraud label ($y_i \in \{0, 1\}$) is **never immediately available**. Fraud is a hidden, adversarial activity discovered through:
   - Customer dispute / unauthorized chargeback filing (typically 30–90 days).
   - Mule account laundering discovery during AML monitoring or law enforcement subpoena (60–180 days).
   - First-party default / bust-out fraud investigation (30–60 days).
3. **Prequential Integrity Requirement**:
   - At time $t$, an adaptive model or threshold policy can **only observe fraud labels whose verification delay has elapsed** ($t_{\text{verified}} = t_{\text{apply}} + \delta \le t$).
   - Any evaluation that updates thresholds or recalibrates uncertainty sets using labels from current or unverified applications suffers from **catastrophic temporal lookahead leakage**.

---

## 2. Mapping Dataset Time Units to Simulated Operational Delays

The NeurIPS 2022 BAF benchmark represents time via discrete integer months $m \in \{0, 1, 2, 3, 4, 5, 6, 7\}$. 
- **Training Period**: Months 0–4 ($N \approx 670,000$ full; $13,514$ in 2% sample).
- **Initial Validation Period**: Month 5 ($N \approx 120,000$ full; $2,386$ in 2% sample).
- **Out-of-Sample Test Period**: Months 6 and 7 ($N \approx 205,000$ full; $4,100$ in 2% sample).

To bridge discrete month steps with industry-standard day-based operational verification timelines:
- Each discrete month $m$ models a **30-day operational billing/monitoring cycle**.
- For each application $i$ arriving in month $m_i \in \{6, 7\}$, continuous intra-month application arrival time is established by normalizing its arrival rank within the month:
  $$t_{\text{apply}}(i) = m_i \times 30.0 + \left(\frac{\text{rank}(i)}{N_m}\right) \times 30.0 \quad \text{days}$$
- Given a simulated feedback delay $\Delta \in \{0, 7, 14, 30, 60\}$ days, the simulated ground-truth verification timestamp is:
  $$t_{\text{verified}}(i) = t_{\text{apply}}(i) + \Delta$$

### Formal Information Eligibility Rule:
At decision time $t$, application $i$'s ground-truth label $y_i$ is **strictly inaccessible** unless:
$$t_{\text{verified}}(i) \le t \iff t_{\text{apply}}(i) + \Delta \le t$$

If $t < t_{\text{verified}}(i)$, the label status is strictly `UNVERIFIED_PENDING`.

---

## 3. Evaluated Delay Scenarios & Experimental Matrix

We evaluate six explicit decision-making configurations across the out-of-sample test horizon (Months 6 and 7):

| Scenario Code | Policy Name | Verification Delay ($\Delta$) | Description & Eligibility Boundary |
| :--- | :--- | :---: | :--- |
| **Scenario A** | **Static Validation Threshold** | N/A (Frozen) | Uses frozen threshold $\tau_{\text{val}}$ selected on Month 5. Zero adaptation regardless of drift or delay. |
| **Scenario B** | **TDTA (Zero-Day Delay)** | $\Delta = 0$ days | Theoretical instant feedback baseline: labels from Month $m$ become immediately available at the boundary to adapt for Month $m+1$. |
| **Scenario C** | **TDTA (7-Day Delay)** | $\Delta = 7$ days | Fast alert feedback: applications arriving in the first 23 days of Month 6 are verified before Month 7 begins; final 7 days remain unverified. |
| **Scenario D** | **TDTA (14-Day Delay)** | $\Delta = 14$ days | Bi-weekly feedback: applications from the first 16 days of Month 6 are verified before Month 7 begins; final 14 days remain unverified. |
| **Scenario E** | **TDTA (30-Day Delay)** | $\Delta = 30$ days | Standard 1-month lag: Month 6 labels are completely unverified when Month 7 begins. Threshold for Month 7 must rely on Month 5 validation buffer. |
| **Scenario F** | **TDTA (60-Day Delay)** | $\Delta = 60$ days | Extended 2-month investigation lag: Month 5 and 6 labels remain unverified. Model relies strictly on Month 4 training buffer. |

---

## 4. Preservation of Conformal Uncertainty & Triage Integrity

1. **Frozen Conformal Calibration**:
   Conformal prediction non-conformity quantiles ($q_{\text{neg}}, q_{\text{pos}}$ at $1 - \alpha = 0.95$) are calibrated strictly on Month 5 validation data. **Conformal prediction sets are NEVER recalibrated on unverified or future test applications**, preventing label leakage.
2. **Unlabeled Drift Detection Allowed**:
   The Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests for feature drift on current application features $X_t$ and model output scores $p(X_t)$ do **not** require labels. Drift detection executes continuously on unlabeled scoring batches.
3. **Threshold Calibration Buffer Guard**:
   When drift triggers threshold adaptation, the calibrator queries the buffer of verified labels $\mathcal{D}_{\text{eligible}} = \{(x_j, y_j) \mid t_{\text{verified}}(j) \le t\}$.
   - If $|\mathcal{D}_{\text{eligible}}| < N_{\text{min}}$ (or negative sample count $< N_{\text{neg}}$), the calibrator automatically executes a safe fallback: retaining the existing threshold rather than updating on an unrepresentative or empty buffer.

