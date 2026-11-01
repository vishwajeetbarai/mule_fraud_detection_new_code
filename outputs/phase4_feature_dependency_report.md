# Phase 4 Feature Dependency & Domain Constraint Specification

**Project**: Mule Account Fraud Detection (SET Major Project)  
**Dataset**: NeurIPS 2022 Bank Account Fraud (BAF) Suite (`BaseDataset.csv`)  
**Scope**: Raw Input Features, Domain Validity Bounds, Derived Feature Mappings & Adversarial Mutability  

---

## 1. Overview of Pipeline Architecture & Perturbation Point

In scientific adversarial evaluation, perturbations applied *after* feature transformations violate domain integrity because derived behavioral, graph, and ratio features fail to reflect the modified underlying states.

Phase 4 enforces the strict sequence:
$$\text{Raw Record } X_{\text{raw}} \xrightarrow{\text{Constrained Perturbation}} \tilde{X}_{\text{raw}} \xrightarrow{\text{Domain Validation / Rejection}} \tilde{X}_{\text{valid}} \xrightarrow{\text{FeaturePreprocessor}} \tilde{X}_{\text{engineered}} \xrightarrow{\text{Frozen Model + Triage}} \hat{Y}$$

---

## 2. Raw Input Features: Types, Sentinels, Bounds & Mutability

| Feature Name | Type | Sentinel / Null | Valid Operational Range | Attacker Mutability | Rationale / Physical Constraint |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `income` | Float | None | $[0.1, 0.9]$ (quantized/binned) | **Mutable (Applicant)** | Self-reported on credit/account application. |
| `name_email_similarity` | Float | None | $[0.0, 1.0]$ | **Mutable (Applicant)** | Governed by email address chosen vs applicant name. |
| `prev_address_months_count` | Int | `-1` | $\{-1\} \cup [0, 480]$ | **Mutable (Applicant)** | Self-reported residency history (integer count). |
| `current_address_months_count` | Int | `-1` | $\{-1\} \cup [0, 600]$ | **Mutable (Applicant)** | Self-reported residency history (integer count). |
| `customer_age` | Int | None | $[10, 90]$ (binned to decades) | **Immutable** | Verified by government identity (KYC/DoB check). |
| `days_since_request` | Float | None | $[0.0, 100.0]$ | **Immutable (System)** | Internal bank audit timestamp difference. |
| `intended_balcon_amount` | Float | $<0$ | $\mathbb{R}_{<0} \cup [0.0, 200.0]$ | **Mutable (Applicant)** | Initial balance transfer requested. |
| `payment_type` | Categorical | None | `{'AA', 'AB', 'AC', 'AD', 'AE'}` | **Mutable (Applicant)** | Chosen payment method mechanism. |
| `zip_count_4w` | Int | None | $[0, 10000]$ | **Immutable (External)** | Regional application density aggregated by postal service. |
| `velocity_6h` | Float | None | $[0.0, 25000.0]$ | **Mutable (Behavioral/Throttling)** | Fraudster can slow rate or inject pacing delays. |
| `velocity_24h` | Float | None | $[0.0, 15000.0]$ | **Mutable (Behavioral/Throttling)** | Throttling burst pacing; must maintain $velocity \ge 0$. |
| `velocity_4w` | Float | None | $[0.0, 10000.0]$ | **Immutable (Network)** | Long-term 4-week window; not easily erased immediately. |
| `bank_branch_count_8w` | Int | None | $[0, 5000]$ | **Immutable (Bank)** | Institutional branch counter across region. |
| `date_of_birth_distinct_emails_4w` | Int | None | $[0, 50]$ | **Immutable (Bank/Identity)** | Velocity of synthetic identities sharing DoB in bureau. |
| `employment_status` | Categorical | None | `{'CA', 'CB', 'CC', 'CD', 'CE', 'CF', 'CG'}` | **Mutable (Applicant)** | Self-reported employment code on application form. |
| `credit_risk_score` | Int | None | $[-200, 500]$ | **Immutable (Bureau)** | External credit bureau score (Experian/Equifax/CIBIL). |
| `email_is_free` | Binary | None | $\{0, 1\}$ | **Mutable (Applicant)** | Choice of email provider domain (Gmail/Yahoo vs corporate). |
| `housing_status` | Categorical | None | `{'BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG'}` | **Mutable (Applicant)** | Self-reported residential status (rent, own, mortgaged). |
| `phone_home_valid` | Binary | None | $\{0, 1\}$ | **Mutable (Applicant)** | Providing valid format / telecommunications lookup flag. |
| `phone_mobile_valid` | Binary | None | $\{0, 1\}$ | **Mutable (Applicant)** | Providing active mobile MSISDN verification. |
| `bank_months_count` | Int | `-1` | $\{-1\} \cup [0, 48]$ | **Immutable (Core Bank)** | Existing account tenure with financial institution. |
| `has_other_cards` | Binary | None | $\{0, 1\}$ | **Mutable (Applicant)** | Self-declared credit card ownership. |
| `proposed_credit_limit` | Float | None | $[200.0, 3000.0]$ | **Mutable (Applicant)** | Desired card limit or overdraft limit requested. |
| `foreign_request` | Binary | None | $\{0, 1\}$ | **Mutable (Behavioral)** | Proxy / VPN routing concealment vs direct IP location. |
| `source` | Categorical | None | `{'INTERNET', 'TELEAPP'}` | **Mutable (Channel)** | Application submission portal. |
| `session_length_in_minutes` | Float | `-1` | $\{-1\} \cup [0.1, 120.0]$ | **Mutable (Behavioral)** | Application form dwell time and interaction speed. |
| `device_os` | Categorical | None | `{'linux', 'macintosh', 'other', 'windows', 'x11'}` | **Mutable (Spoofing)** | Browser User-Agent header emulation. |
| `keep_alive_session` | Binary | None | $\{0, 1\}$ | **Mutable (Client)** | HTTP session persistence flag. |
| `device_distinct_emails_8w` | Int | `-1` | $\{-1\} \cup [0, 10]$ | **Partially Mutable (Device)**| Virtual machine reset, canvas spoofing, cookie clearing. |
| `device_fraud_count` | Int | None | Constant 0 | **Immutable** | Constant baseline in Base dataset. |
| `month` | Int | None | $\{0, 1, ..., 7\}$ | **Immutable (Chronological)**| Chronological partition index; NEVER perturbed. |

---

## 3. Derived Feature Dependencies & Consistency Invariants

When raw features are perturbed, the following mathematical invariants and dependencies are re-computed via `FeaturePreprocessor._engineer()`:

### A. Velocity & Acceleration Ratios
- `velocity_ratio_6h_24h` = $\frac{\text{velocity\_6h}}{\text{velocity\_24h} + \epsilon}$
- `velocity_surge_intensity` = $\frac{\text{velocity\_6h} / 6}{\text{velocity\_24h} / 24 + \epsilon}$
- `velocity_burst_acceleration` = $\frac{\text{velocity\_6h}}{6} - \frac{\text{velocity\_24h}}{24}$
- `velocity_spike_index` = $\frac{(\text{velocity\_6h})^{1.5}}{\sqrt{\text{velocity\_4w} + 1.0} + \epsilon}$
- *Invariance Constraint*: Velocity counts cannot be negative. If an attacker attempts to reduce burst detection by throttling `velocity_6h`, the surge acceleration and spike index must dynamically adjust downwards in exact mathematical synchrony.

### B. Behavioral Graph Projections
- `device_sharing_density` = $\text{device\_distinct\_emails\_8w} \times \text{date\_of\_birth\_distinct\_emails\_4w}$
- `device_recycling_speed` = $\frac{\text{device\_distinct\_emails\_8w}}{\text{session\_length\_in\_minutes} + \epsilon}$
- `graph_hub_degree_proxy` = $\sum \log(1 + \text{degrees})$
- `device_to_address_collision_ratio` = $\frac{\text{device\_distinct\_emails\_8w}}{\text{current\_address\_months\_count} + 1}$
- *Invariance Constraint*: If `device_distinct_emails_8w` is cloaked from 2 to 1, device recycling speed and hub centrality must decrease simultaneously.

### C. Financial Exposure & Incoherence
- `credit_limit_to_income` = $\frac{\text{proposed\_credit\_limit}}{\text{income} + \epsilon}$
- `name_email_free_mismatch` = $(1 - \text{name\_email\_similarity}) \times \text{email\_is_free}$
- `channel_risk_index` = $(1 - \text{phone\_home\_valid}) + (1 - \text{phone\_mobile\_valid}) + \text{foreign\_request}$
- *Invariance Constraint*: Artificially claiming lower credit limits directly reduces the credit-to-income leverage ratio; creating a high-similarity email address directly cancels the free-mismatch penalty.

### D. Residential & Banking Tenure
- `total_address_months` = $\text{current\_address\_months\_count} + \text{prev\_address\_months\_count}$
- `address_stability_ratio` = $\frac{\text{current\_address\_months\_count}}{\text{total\_address\_months} + 1}$
- `bank_tenure_to_age_ratio` = $\frac{\text{bank\_months\_count}}{\text{customer\_age} \times 12 + \epsilon}$
- *Invariance Constraint*: Total address months must equal the sum of current and previous address months; tenure cannot exceed applicant biological lifespan.

---

## 4. Rejection Criteria for Domain-Invalid Records

Any perturbed record violating the following domain rules is flagged as an invalid state and rejected by the validator:
1. **Negative Value Violations**: Non-sentinel features (e.g. `income`, `proposed_credit_limit`, `session_length_in_minutes`) taking values $< 0$.
2. **Probability / Bounded Violations**: `name_email_similarity` $\notin [0.0, 1.0]$ or `income` $> 1.0$.
3. **Invalid Sentinels**: Categorical or count features having arbitrary negative sentinels other than defined $\{-1\}$.
4. **Illegal Categoricals**: Substituted categories outside the admissible benchmark vocabulary (e.g., unknown OS, unknown employment category).
5. **Age / Tenure Impossibilities**: Current address tenure exceeding biological age in months ($\text{current\_address\_months} > \text{customer\_age} \times 12$).
6. **Immutable Tampering**: Any mutation targeting verified bureau scores (`credit_risk_score`), bank history (`bank_months_count`), regional stats (`zip_count_4w`, `bank_branch_count_8w`), or time index (`month`).

