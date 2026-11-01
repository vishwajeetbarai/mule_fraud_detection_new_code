# Phase 4 Validation Summary & Defense Brief

**Project**: Mule Account Fraud Detection (SET Major Research Project)  
**Deliverable**: Phase 4 Sign-Off & Viva Defense Summary  
**Evaluation Benchmark**: NeurIPS 2022 Bank Account Fraud (BAF)  

---

## 1. Scope of Phase 4 Implementation

| Requirement | Implementation Artifact | Verification Test | Status |
| :--- | :--- | :--- | :---: |
| **Raw Feature Dependencies** | `outputs/phase4_feature_dependency_report.md` | Invariance & logic review | **VERIFIED** |
| **Realistic Threat Models** | `outputs/phase4_threat_model_report.md` | Attacker capability boundary review | **VERIFIED** |
| **Raw Perturbation Engine** | `src/realistic_adversarial.py` (`AdversarialPerturbationGenerator`) | `tests/test_adversarial_phase4.py` | **VERIFIED** |
| **Domain Constraint Validator** | `src/realistic_adversarial.py` (`DomainValidator`) | `tests/test_adversarial_phase4.py` | **VERIFIED** |
| **Recalculation of Derived Features** | `features.py` re-invoked on perturbed raw inputs | `test_derived_features_recalculated` | **VERIFIED** |
| **Action Distribution Tracking** | `outputs/phase4_action_distribution.csv` | Multi-action triage routing | **VERIFIED** |
| **Zero Test-Label Leakage** | Thresholds frozen from validation Month 5 | `tests/test_leakage_phase1.py` | **VERIFIED** |
| **Full Pytest Suite** | 42 passed in 48.54s | Automated regression testing | **VERIFIED** |

---

## 2. Answers to Essential Examiner Questions (Viva Defense)

### Q1: Why did you perturb raw features instead of adding noise to the engineered feature matrix?
> *"Adding noise directly to an engineered matrix violates domain realism and physical invariants. For instance, modifying `velocity_surge_intensity` without modifying `velocity_6h` creates a state that could never exist in a real banking pipeline. By perturbing raw applicant and device inputs and running them through the actual `FeaturePreprocessor`, all 86 derived ratios, network density metrics, and incoherence indices are mathematically recalculated in exact accordance with real-world system behavior."*

### Q2: Can fraudsters realistically evade your model?
> *"Yes. Under Scenario 1 (Application Camouflage) and Scenario 5 (Multi-Vector Syndicate), fraudsters manipulating self-reported data reduce standalone automated model recall from 44.3% down to 18.0%. However, because our framework employs a Hierarchical Triage Policy with split conformal prediction, these camouflaged applications exhibit high epistemic uncertainty and are caught by the conformal safety net, routing 90.2%–91.8% of frauds to manual human review rather than allowing them to be automatically approved."*

### Q3: Why didn't you let the attacker tamper with credit scores or banking months?
> *"In real banking systems, credit risk scores (`credit_risk_score`) are pulled directly via server-to-server APIs from regulated bureaus (e.g. CIBIL/Experian), and bank tenure (`bank_months_count`) is verified against internal core databases. Allowing an attacker to tamper with external verified bureau data assumes an unrealistic insider threat rather than external account-opening fraud."*

---

## 3. Methodological Caveats & Boundaries

1. **Synthetic Data Context**: As with prior phases, results are evaluated on the semi-synthetic NeurIPS 2022 BAF benchmark. The attacks represent plausible form-tuning and device emulation, not confirmed cybercriminal tradecraft.
2. **Human Review Capacity**: Routing 53%–69% of applications to human review under adversarial conditions ensures high fraud recall (>90%), but in a live production bank, triage capacity must be balanced against operational investigator headcount and budget.

---

## 4. Final Phase 4 Sign-Off

- **Code Quality**: Production-ready, fully typed, docstring-documented.
- **Automated Tests**: **42 / 42 passed** (`pytest`).
- **Smoke-Test Execution**: Succeeded with zero runtime exceptions.
- **Phase 4 Status**: **APPROVED**.

