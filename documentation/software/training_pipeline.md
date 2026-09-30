# Software Training Pipeline & Optimization

The software training pipeline is implemented in PyTorch and handles data loading, loss formulation, learning rate scheduling, validation, and post-training threshold optimization.

---

## 1. Loss Formulation

Cardiac datasets exhibit extreme class imbalance (e.g., >75,000 Normal beats vs. ~3,000 Supraventricular beats). Standard Binary Cross Entropy (BCE) causes the model to favor the majority class, sacrificing sensitivity on critical arrhythmias.

### Focal Binary Cross Entropy (`FocalBCELoss`)
Down-weights easy, well-classified examples (Normal beats) and focuses gradient updates on difficult minority classes:
$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
- $\gamma = 2.0$ (focusing parameter)
- $\alpha_t$: per-class morphology weighting tensor loaded from `config.yaml`.

### Morphology-Weighted BCE (`MorphologyWeightedBCELoss`)
Applies asymmetric positive class weighting using `pos_weight` in BCEWithLogitsLoss:
$$\mathcal{L}_c = - \left[ w_c \cdot y_c \log(\sigma(z_c)) + (1 - y_c) \log(1 - \sigma(z_c)) \right]$$

---

## 2. Training Enhancements & Augmentation

1. **Realistic ECG Data Augmentation (`ECGDataset`)**:
   - **Time Jitter**: Random circular shift $\pm 10$ samples ($\approx \pm 27.7\text{ ms}$).
   - **Amplitude Scaling**: Random voltage scaling by $\pm 10\%$ ($0.90 \le \alpha \le 1.10$).
   - **Baseline Gaussian Noise**: Additive Gaussian noise $\mathcal{N}(0, 0.01^2)$ to emulate electrode motion artifacts.

2. **Learning Rate Scheduling**:
   - Cosine Annealing with Warm Restarts (`CosineAnnealingLR`) or validation F1-triggered decay (`ReduceLROnPlateau`).
   - Default initial learning rate: $\eta = 1.0 \times 10^{-3}$, min learning rate: $\eta_{\min} = 1.0 \times 10^{-6}$.

3. **Early Stopping**:
   - Monitors macro-average F1 on the independent patient validation split with patience of 10 epochs.

---

## 3. Per-Class Optimal Threshold Tuning

Rather than applying a fixed binary decision threshold ($t = 0.5$) across all classes, the trainer performs a post-training sweep across validation probabilities:
$$t_c^* = \arg\max_{t \in [0.01, 0.99]} \text{F1}_c(t)$$
These optimal thresholds are saved to `optimal_thresholds.json` and bundled directly inside model checkpoints.
