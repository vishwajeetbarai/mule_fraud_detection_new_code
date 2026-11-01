# Phase 4 Realistic Threat Models & Adversarial Scenarios

**Project**: Mule Account Fraud Detection (SET Major Research Project)  
**Dataset**: NeurIPS 2022 Bank Account Fraud (BAF) Benchmark (`BaseDataset.csv`)  
**Adversarial Scope**: Constrained, Domain-Admissible Evasion on Raw Inputs  

---

## 1. Threat Modeling Philosophy & Research Boundaries

In academic literature on fraud detection, threat models are frequently criticized for granting attackers unrealistic capabilities (e.g., modifying internal credit bureau databases or changing chronological timestamps). 

In real-world bank account opening fraud:
- **Attacker Knowledge**: Attackers do NOT possess white-box access to the bank's internal gradient boosted decision trees, conformal non-conformity calibration quantiles, or exact decision thresholds.
- **Attacker Capabilities**: Attackers can only control:
  1. The information they enter on the application form (income, requested credit limit, contact information, residential history).
  2. The technical parameters of their client device (User-Agent OS string, session dwell time, VPN/proxy egress).
  3. The pacing and velocity of their automated scripts (throttling burst submissions).
- **Attacker Non-Capabilities**: Attackers CANNOT modify:
  1. External credit bureau scores (`credit_risk_score`).
  2. Verified birth dates or government ID registry age (`customer_age`).
  3. Internal core banking tenure (`bank_months_count`).
  4. Regional postal application volume (`zip_count_4w`, `bank_branch_count_8w`).
  5. Cross-entity graph collisions tracked across the banking network (`date_of_birth_distinct_emails_4w`).

---

## 2. Specification of Evaluated Realistic Threat Models

We define 5 realistic, domain-constrained threat models plus a composite multi-vector scenario:

### Scenario 1: Identity & Application Camouflage (Applicant-Provided Evasion)
- **Target Vector**: Applicant-controlled form fields.
- **Attacker Action**:
  - Modifies email address generation so that `name_email_similarity` increases towards 1.0 (matching applicant name).
  - Uses a paid/custom domain rather than a free disposable provider (`email_is_free` toggled from 1 to 0).
  - Provides residential history that appears stable (`current_address_months_count` increased by +12 to +36 months).
  - Normalizes requested credit limit (`proposed_credit_limit` reduced towards median $500).
- **Perturbation Budget**:
  - $\Delta \text{name\_email\_similarity} \in [0.10, 0.30]$ (clamped to 1.0).
  - $\text{email\_is\_free} \to 0$.
  - $\text{proposed\_credit\_limit} \leftarrow \max(200.0, \text{proposed\_credit\_limit} \times 0.8)$.
- **Domain Invariants**: $0 \le \text{name\_email\_similarity} \le 1.0$; credit limit $\ge 200$; current address months $\le \text{customer\_age} \times 12$.

### Scenario 2: Velocity Throttling & Burst Concealment (Behavioral Pacing)
- **Target Vector**: Application velocity and burst pacing.
- **Attacker Action**: Fraud syndicates use timers, rate limiters, or delay queues between consecutive submissions to reduce short-term surge spikes.
- **Attacker Modification**:
  - `velocity_6h` throttled down by $15\%$ to $30\%$.
  - `session_length_in_minutes` increased by $+2.0$ to $+5.0$ minutes (simulating realistic human typing/reading delays).
- **Perturbation Budget**:
  - $\text{velocity\_6h} \leftarrow \text{velocity\_6h} \times (1 - \epsilon)$ with $\epsilon \in \{0.15, 0.30\}$.
  - $\text{session\_length\_in\_minutes} \leftarrow \text{session\_length\_in\_minutes} + 3.0$.
- **Domain Invariants**: Non-negative velocity; recalculation of all velocity acceleration, burst, and recycling speed features in `FeaturePreprocessor`.

### Scenario 3: Device Cloaking & User-Agent Masquerade (Client Spoofing)
- **Target Vector**: Client technical footprint.
- **Attacker Action**: Mule rings switch browser user-agents, rotate browser fingerprints, or clear cache/cookies between submissions.
- **Attacker Modification**:
  - `device_distinct_emails_8w` reduced from 2 to 1 (simulating browser profile isolation or fresh VM instances).
  - `device_os` substituted to standard desktop environment (`'windows'` or `'macintosh'`).
  - `keep_alive_session` set to 1.
- **Perturbation Budget**: Discrete valid substitution within benchmark category levels `['windows', 'macintosh', 'linux', 'other', 'x11']`.
- **Domain Invariants**: Categorical levels strictly contained in training vocabulary; distinct email count $\ge 0$.

### Scenario 4: Missingness Exploitation & Sensor Occlusion
- **Target Vector**: Optional form fields and privacy shielding.
- **Attacker Action**: Fraudsters withhold voluntary fields or conceal phone numbers to prevent telecommunication carrier risk checks.
- **Attacker Modification**:
  - Set `prev_address_months_count` to sentinel `-1` (missing).
  - Set `intended_balcon_amount` to sentinel `< 0` (no initial balance transfer specified).
  - Set `phone_home_valid` to 0.
- **Perturbation Budget**: Discrete sentinel replacement.
- **Domain Invariants**: Missingness sentinel indicators (`_missing` flags) are recalculated dynamically by `_clean_missing()`.

### Scenario 5: Sophisticated Multi-Vector Syndicate Evasion (Composite Attack)
- **Target Vector**: Simultaneous application form tuning, rate throttling, and device profile cloaking.
- **Attacker Action**: Combines Scenarios 1, 2, and 3:
  - Form camouflage: $\text{name\_email\_similarity} \leftarrow \min(1.0, \text{name\_email\_similarity} + 0.20)$, $\text{email\_is\_free} \leftarrow 0$.
  - Behavioral throttling: $\text{velocity\_6h} \leftarrow \text{velocity\_6h} \times 0.80$, $\text{session\_length} \leftarrow \text{session\_length} + 3.0$.
  - Device cloaking: $\text{device\_distinct\_emails\_8w} \leftarrow 1$ (if $>1$), $\text{device\_os} \leftarrow \text{'windows'}$.
- **Attacker Capability Rationale**: Represents an organized fraud syndicate executing a coordinated evasion playbook.

---

## 3. Threat Model Matrix & Summary

| Threat Scenario | Attacker Type | Target Features | Immutable Protected Features | Target Goal | Expected System Effect |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **S1: App Camouflage** | Opportunistic Applicant | `name_email_similarity`, `email_is_free`, `proposed_credit_limit`, `current_address_months` | `credit_risk_score`, `customer_age`, `bank_months` | Reduce form suspicion | Decreases credit leverage & incoherence penalties |
| **S2: Velocity Throttle** | Scripted Bot Operator | `velocity_6h`, `session_length_in_minutes` | `velocity_4w`, `bank_branch_count_8w` | Bypass rate-limiting | Dampens burst acceleration and surge index |
| **S3: Device Cloaking** | Emulation / Multi-Accounter | `device_distinct_emails_8w`, `device_os`, `keep_alive_session` | Graph co-occurrence in bureau | Conceal device reuse | Reduces graph hub degree and clique density |
| **S4: Sensor Occlusion** | Privacy / Evasion | `prev_address_months_count`, `intended_balcon_amount`, `phone_home_valid` | Verified ID, Core Bureau | Hide adverse history | Triggers sentinel missingness indicators |
| **S5: Multi-Vector** | Organized Mule Syndicate | S1 + S2 + S3 joint application | All external bank & bureau records | Maximize evasion | Broadest performance stress on triage |

