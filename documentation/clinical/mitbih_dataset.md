# MIT-BIH Database & Preprocessing

The **MIT-BIH Arrhythmia Database** (PhysioNet) is the gold standard benchmark for evaluating automated ECG arrhythmia detection algorithms.

---

## 1. Database Characteristics

- **Total Records**: 48 half-hour two-channel ambulatory ECG recordings from 47 subjects.
- **Sampling Frequency ($f_s$)**: 360 Hz.
- **Resolution**: 11-bit resolution over a $\pm 10\text{ mV}$ range.
- **Leads**: Channel 1 is typically a modified limb lead II (MLII); Channel 2 is typically a modified lead $V_1$ (occasionally $V_2, V_4,$ or $V_5$).

---

## 2. Ingestion & Automated Download

The automated downloader fetches recordings directly from PhysioNet via WFDB:

```bash
python software/scripts/download_dataset.py --config software/config/config.yaml
```

This downloads raw `.dat`, `.hea`, and `.atr` files into `software/data/raw/`.

---

## 3. Preprocessing Steps

```mermaid
flowchart LR
    A[Raw ECG] --> B[Bandpass Filter<br>0.5 - 45 Hz]
    B --> C[XQRS Peak Detection]
    C --> D[Window Extraction<br>256 Samples]
    D --> E[Binary Label Matching<br>O(log M)]
    E --> F[Training Tensor]
```

1. **R-Peak Detection (`detect_r_peaks`)**:
   - Uses `wfdb.processing.xqrs_detect` to identify the peak of the QRS complex.
   - Enforces physiological refractory periods corresponding to minimum and maximum heart rate limits ($30 \le \text{HR} \le 300\text{ BPM}$).

2. **Dual-Lead Window Segmentation (`extract_windows`)**:
   - Simultaneously extracts from Channel 1 (MLII) and Channel 2 (V1), producing 2-channel tensors of shape `(2, 256)`.
   - 256 samples at 360 Hz corresponds to $\approx 711.1\text{ ms}$, ensuring complete capture of the preceding P-wave, PR interval, QRS complex, ST segment, and T-wave.

3. **Auxiliary RR Timing Ratios (`compute_rr_features`)**:
   - Pre-RR ratio: $(R_i - R_{i-1}) / \text{median}(RR_{\text{local}})$.
   - Post-RR ratio: $(R_{i+1} - R_i) / \text{median}(RR_{\text{local}})$.
   - Clamped to $[0.2, 3.0]$ to reject extreme artifact outliers while preserving premature ectopy indicators (PACs, PVCs).

4. **Patient-Adaptive Baseline Subtraction (`compute_patient_median_template`)**:
   - Computes the patient's individual median Normal beat morphology.
   - Subtracting idiosyncratic resting morphology suppresses anatomical inter-patient variance, boosting generalization on unseen patient test sets.

5. **Label Assignment (`extract_labels`) & Multi-Hot Encoding**:
   - Uses binary search (`np.searchsorted`) to match each window center with the nearest cardiologist annotation within a 12-sample tolerance window ($33.3\text{ ms}$).
   - Produces a target vector $\mathbf{y} \in \{0, 1\}^5$ for each beat, supporting multi-label cardiac conditions.
