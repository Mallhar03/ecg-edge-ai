# Project Documentation Index

Welcome to the comprehensive technical documentation for the **Embedded ECG Arrhythmia Classifier & Accelerator**.

This documentation is structured into modular sections covering system architecture, clinical methodology, software engineering, hardware deployment, and academic references.

---

## 📚 Documentation Structure

### 1. [Architecture Overview](architecture/system_overview.md)
- [System Overview & Pipeline](architecture/system_overview.md) — High-level dataflow from ECG acquisition to edge inference.
- [Multi-Scale 1D-CNN Model Architecture](architecture/model_design.md) — Multi-branch convolution, Squeeze-and-Excitation (SE) channel attention, and dual-pooling.

### 2. [Clinical & Dataset Methodology](clinical/aami_ec57_mapping.md)
- [AAMI EC57 Standard Class Mapping](clinical/aami_ec57_mapping.md) — Mapping MIT-BIH symbols to 5 standard clinical classes; data leakage prevention.
- [MIT-BIH Database & Preprocessing](clinical/mitbih_dataset.md) — Ingestion, R-peak detection (XQRS), windowing, and patient-level splitting.

### 3. [Software & Training Pipeline](software/training_pipeline.md)
- [Training, Loss & Optimization](software/training_pipeline.md) — Focal Loss, Morphology-Weighted BCE, Cosine Annealing, and per-class threshold sweeps.
- [Quantization-Aware Training (QAT) & Weight Export](software/quantization_and_export.md) — INT8 Sensitivity-Preserving QAT via Brevitas, `weights.h` C++ array generation, and FPGA `.mem` files.

### 4. [Hardware & Embedded Deployment](hardware/embedded_deployment.md)
- [Embedded Target (Arduino Giga R1 / STM32H7)](hardware/embedded_deployment.md) — Real-time 360 Hz ADC sampling, dual-core MCU memory budget, and execution timing.

### 5. [References & Further Reading](resources/references.md)
- [Academic Papers & External Resources](resources/references.md) — Links to research publications, standard definitions, dataset repositories, and tool documentation.
