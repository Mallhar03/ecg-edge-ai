# Clinical Classification & AAMI EC57 Standard

In clinical electrocardiography, raw beat annotations must be organized into clinically actionable diagnostic categories. This project adheres strictly to the **Association for the Advancement of Medical Instrumentation (AAMI) EC57 standard**.

---

## 1. Five-Class Mapping Schema

All MIT-BIH Arrhythmia Database annotation symbols are explicitly mapped to 5 standard classes:

| Class Name | AAMI Class | MIT-BIH Symbols Included | Description & Clinical Severity |
|---|---|---|---|
| **Normal** | **N** | `N`, `.` | Normal sinus rhythm and sinus bradycardia/tachycardia. Baseline healthy rhythm. |
| **Supraventricular** | **S** | `A`, `a`, `J`, `S`, `e`, `j` | Atrial and nodal ectopic beats (Premature Atrial Contractions, Junctional beats). High risk of atrial fibrillation. |
| **Ventricular** | **V** | `V`, `F` | Premature Ventricular Contractions (PVCs) and Ventricular Fusion beats. Life-threatening; high risk of Ventricular Tachycardia and Fibrillation. |
| **Bundle Branch Block** | **N / BBB** | `L`, `R` | Left and Right Bundle Branch Block. Intraventricular conduction delay and structural heart disease. |
| **Paced / Other** | **Q** | `!`, `/`, `f`, `Q` | Pacemaker rhythms, paced fusion beats, and unclassifiable waveforms. |

---

## 2. Preventing Common Pitfalls & Data Leakage

### Inter-Patient vs. Intra-Patient Splitting
- **The Problem**: Intra-patient splitting randomly shuffles all beats across the training and test sets. Because a patient's individual ECG morphology is consistent, intra-patient models merely memorize patient signatures, resulting in falsely inflated F1 scores (98–99%) that catastrophically fail on unseen patients.
- **The Solution**: We strictly enforce **inter-patient splitting** based on *Chazal et al. (2004)*. Entire patient records are allocated exclusively to either the training set (44 records) or the test set (4 independent patient records: `104`, `107`, `217`, `232`).

### Eliminating Silent Fallbacks
- Prior implementations erroneously fell back to labeling unmapped beats as `Normal`. This caused over **14,900 dangerous beats** (7,130 PVCs and 7,028 paced beats) to contaminate the Normal training distribution.
- In this implementation, every beat is explicitly mapped, and unmapped artifacts produce an all-zero vector and are discarded from training.

### Global Normalization Consistency
- Global mean ($\mu_{\text{train}}$) and standard deviation ($\sigma_{\text{train}}$) are computed **strictly from the training records**.
- These exact scalars are reused for normalizing the validation and test sets to prevent distributional data leakage.
