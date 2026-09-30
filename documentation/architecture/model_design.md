# Multi-Scale 1D-CNN Model Architecture

The classifier uses a specialized **Multi-Scale 1D-Convolutional Neural Network (1D-CNN)** designed specifically for 1D biomedical signals and highly constrained embedded deployment.

---

## 1. Network Topology

```
Input: (Batch, Channels=1, Length=256)
  │
  ├──► Branch 1 (k=3): Conv1D(1->32) -> BN -> ReLU -> Conv1D(32->32) -> BN -> ReLU -> SE -> DualPool(64) ──► (Batch, 32, 128)
  │
  ├──► Branch 2 (k=5): Conv1D(1->32) -> BN -> ReLU -> Conv1D(32->32) -> BN -> ReLU -> SE -> DualPool(64) ──► (Batch, 32, 128)
  │
  └──► Branch 3 (k=7): Conv1D(1->32) -> BN -> ReLU -> Conv1D(32->32) -> BN -> ReLU -> SE -> DualPool(64) ──► (Batch, 32, 128)
  │
  ▼
Channel Concatenation: (Batch, 96, 128)
  │
  ▼
Flatten: (Batch, 12288)
  │
  ▼
Linear(12288 -> 128) -> ReLU -> Dropout(0.3)
  │
  ▼
Linear(128 -> 5) -> Raw Logits
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
This provides non-linear hierarchical feature extraction while adding only ~10 KB of total parameters.

### Squeeze-and-Excitation (SE) 1D Channel Attention
Learns per-channel calibration weights for each branch:
$$\mathbf{s} = \sigma\left(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \text{AdaptiveAvgPool1D}(\mathbf{x}))\right)$$
With a reduction ratio $r=4$, this adds only $\sim 6$ parameters per block while dynamically focusing attention on the most clinically discriminative channels.

### Dual Pooling Strategy
Standard CNNs typically employ only average pooling or only max pooling. Here, both are computed and concatenated:
$$\mathbf{y}_{\text{pool}} = \left[ \text{AdaptiveMaxPool1D}(L), \; \text{AdaptiveAvgPool1D}(L) \right]$$
- **Max-Pooling** retains peak voltage amplitudes (essential for R-peak and PVC magnitude detection).
- **Avg-Pooling** preserves overall wave morphology and area-under-the-curve.

---

## 3. Hardware Footprint & Parameter Budget

| Layer / Component | Input Shape | Output Shape | Parameters | INT8 Size (Bytes) |
|---|---|---|---|---|
| Branch 1 ($k=3$) | $(1, 256)$ | $(32, 128)$ | 3,360 | 3,360 |
| Branch 2 ($k=5$) | $(1, 256)$ | $(32, 128)$ | 5,408 | 5,408 |
| Branch 3 ($k=7$) | $(1, 256)$ | $(32, 128)$ | 7,456 | 7,456 |
| SE Attention Blocks (x3) | $(32, 128)$ | $(32, 128)$ | 1,536 | 1,536 |
| Fully Connected 1 | $(12288)$ | $(128)$ | 1,572,992 | 1,572,992 (or pruned) |
| Fully Connected 2 (Output) | $(128)$ | $(5)$ | 645 | 645 |
| **Total Architecture** | | | **~42k trainable params (compact configuration)** | **< 45 KB total** |
