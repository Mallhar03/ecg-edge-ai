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
- **The Solution**: We strictly enforce **inter-patient splitting** based on the canonical *Chazal et al. (2004)* benchmark:
  - **Excluded Records**: Records `102`, `104`, `107`, and `217` are explicitly excluded as mandated by AAMI EC57 because they contain paced beats.
  - **DS1 (Training & Validation Set - 22 records)**: `101`, `106`, `108`, `109`, `112`, `114`, `115`, `116`, `118`, `119`, `122`, `124`, `201`, `203`, `205`, `207`, `208`, `209`, `215`, `220`, `223`, `230`.
  - **DS2 (Independent Test Set - 22 records)**: `100`, `103`, `105`, `111`, `113`, `117`, `121`, `123`, `200`, `202`, `210`, `212`, `213`, `214`, `219`, `221`, `222`, `228`, `231`, `232`, `233`, `234`.

### Eliminating Silent Fallbacks
- Prior implementations erroneously fell back to labeling unmapped beats as `Normal`. This caused over **14,900 dangerous beats** (7,130 PVCs and 7,028 paced beats) to contaminate the Normal training distribution.
- In this implementation, every beat is explicitly mapped, and unmapped artifacts produce an all-zero vector and are discarded from training.

### Global Normalization Consistency
- Global mean ($\mu_{\text{train}}$) and standard deviation ($\sigma_{\text{train}}$) are computed **strictly from the training records**.
- These exact scalars are reused for normalizing the validation and test sets to prevent distributional data leakage.
