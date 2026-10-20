# Multi-Scale 1D-CNN Model Architecture

The classifier uses a specialized **Multi-Scale 1D-Convolutional Neural Network (1D-CNN)** designed specifically for 1D biomedical signals and highly constrained embedded deployment.

---

## 1. Network Topology

```
Input: (Batch, Channels=2, Length=256) [Lead II + Lead V1]
  │
  ├──► Branch 1 (k=3): Conv1D(2->32) -> BN -> ReLU -> Conv1D(32->32) -> BN -> ReLU -> SE -> DualPool(16) ──► (Batch, 32, 32)
  │
  ├──► Branch 2 (k=5): Conv1D(2->32) -> BN -> ReLU -> Conv1D(32->32) -> BN -> ReLU -> SE -> DualPool(16) ──► (Batch, 32, 32)
  │
  └──► Branch 3 (k=7): Conv1D(2->32) -> BN -> ReLU -> Conv1D(32->32) -> BN -> ReLU -> SE -> DualPool(16) ──► (Batch, 32, 32)
  │
  ▼
Channel Concatenation: (Batch, 96, 32)
  │
  ▼
Flatten: (Batch, 3072)
  │
  ├── [Auxiliary RR Features: pre_RR, post_RR ratios] ──► (Batch, 2)
  ▼
Concatenation: (Batch, 3074)
  │
  ▼
Linear(3074 -> 64) -> ReLU -> Dropout(0.3)
  │
  ▼
Linear(64 -> 5) -> Raw Logits
```

---

## 2. Core Design Components

### Multi-Scale Parallel Branches
ECG features occur across differing temporal durations:
- **Sharp QRS spikes** (~80–120 ms) benefit from compact receptive fields ($k=3$).
- **Broader P-waves and T-waves** (~120–200 ms) require wider temporal context ($k=5, 7$).
By executing parallel branches with varying kernel sizes, the network simultaneously captures high-frequency edges and low-frequency baseline waveforms without requiring deep, computationally heavy networks.

### Double Conv1D Hierarchy
Each branch utilizes two sequential Conv1D layers with Batch Normalization:
$$\text{Conv1D}(C_{\text{in}}, C_{\text{out}}, k) \to \text{BN} \to \text{ReLU} \to \text{Conv1D}(C_{\text{out}}, C_{\text{out}}, k) \to \text{BN} \to \text{ReLU}$$
This provides non-linear hierarchical feature extraction while adding minimal total parameters.

### Squeeze-and-Excitation (SE) 1D Channel Attention
Learns per-channel calibration weights for each branch:
$$\mathbf{s} = \sigma\left(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \text{AdaptiveAvgPool1D}(\mathbf{x}))\right)$$
With a reduction ratio $r=4$, this dynamically focuses attention on the most clinically discriminative channels.

### Dual Pooling Strategy
Standard CNNs typically employ only average pooling or only max pooling. Here, both are computed and concatenated:
$$\mathbf{y}_{\text{pool}} = \left[ \text{AdaptiveMaxPool1D}(16), \; \text{AdaptiveAvgPool1D}(16) \right] \in \mathbb{R}^{32 \times 32}$$
- **Max-Pooling** retains peak voltage amplitudes (essential for R-peak and PVC magnitude detection).
- **Avg-Pooling** preserves overall wave morphology and area-under-the-curve.

### Auxiliary Timing Feature Fusion
Clinical arrhythmias (such as Premature Ventricular Contractions and Premature Atrial Contractions) are fundamentally characterized by prematurity relative to the underlying sinus cadence:
$$\text{pre\_RR} = \frac{R_i - R_{i-1}}{\text{median}(RR_{\text{local}})}, \quad \text{post\_RR} = \frac{R_{i+1} - R_i}{\text{median}(RR_{\text{local}})}$$
These scalar ratios are concatenated directly into the dense classification head.

---

## 3. Hardware Footprint & Parameter Budget

| Layer / Component | Input Shape | Output Shape | Parameters | INT8 Size (Bytes) |
|---|---|---|---|---|
| Branch 1 ($k=3$) | $(2, 256)$ | $(32, 32)$ | 3,456 | 3,456 |
| Branch 2 ($k=5$) | $(2, 256)$ | $(32, 32)$ | 5,504 | 5,504 |
| Branch 3 ($k=7$) | $(2, 256)$ | $(32, 32)$ | 7,552 | 7,552 |
| SE Attention Blocks (x3) | $(32, 32)$ | $(32, 32)$ | 1,536 | 1,536 |
| Dense Classifier (FC1) | $(3074)$ | $(64)$ | 196,736 | 196,736 |
| Output Layer (FC2) | $(64)$ | $(5)$ | 325 | 325 |
| **Total Architecture** | | | **~215k params** | **< 220 KB (fits entirely in STM32H7 SRAM)** |
