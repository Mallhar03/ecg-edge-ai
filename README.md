# ECG Arrhythmia Accelerator (Arduino Giga R1 Target)

> **Real-time cardiac arrhythmia detection on Arduino Giga R1 (STM32H747XI MCU) using an INT8-quantized Multi-Scale 1D-CNN trained on the MIT-BIH Arrhythmia Database.**

This repository contains the full end-to-end pipeline for an embedded ECG arrhythmia classifier — from raw PhysioNet signal ingestion to INT8 quantized model training (with clinical sensitivity preservation) through to C++ `weights.h` header array exports for deployment on the Arduino Giga R1 MCU (with original FPGA `.mem` hardware artifacts archived for reference).

---

## Table of Contents

- [Project Overview](#project-overview)
- [Current Hardware Target](#current-hardware-target)
- [Repository Structure](#repository-structure)
- [Software Documentation](#software-documentation)
  - [Architecture](#architecture)
  - [Dataset & Preprocessing](#dataset--preprocessing)
  - [Model Architecture](#model-architecture)
  - [Training Pipeline (Phase 1)](#training-pipeline-phase-1)
  - [Quantization-Aware Training — SP-QAT (Phase 2)](#quantization-aware-training--sp-qat-phase-2)
  - [Hardware Weight Export](#hardware-weight-export)
  - [Configuration Reference](#configuration-reference)
  - [Setup & Installation](#setup--installation)
  - [Running the Pipeline](#running-the-pipeline)
  - [Evaluation & Metrics](#evaluation--metrics)
- [Hardware Documentation](#hardware-documentation)
  - [Current Hardware Target: Arduino Giga R1](#current-hardware-target-arduino-giga-r1)
  - [Archived: Original FPGA Implementation](#archived-original-fpga-implementation)
    - [FPGA Target & Toolchain](#fpga-target--toolchain)
    - [RTL Architecture](#rtl-architecture)
    - [Weight File Format](#weight-file-format)
    - [Weights Manifest](#weights-manifest)
    - [Constraints](#constraints)
    - [Simulation & Testbenches](#simulation--testbenches)
  - [Hardware–Software Interface](#hardwaresoftware-interface)
- [Clinical Design Decisions](#clinical-design-decisions)
- [Development Workflow & Branching](#development-workflow--branching)
- [Contributors](#contributors)

---

## Project Overview

Cardiac arrhythmias are among the leading causes of sudden cardiac death. Real-time, low-power ECG analysis—traditionally performed on resource-limited embedded devices—is a hard problem. This project bridges the gap between a clinically validated deep-learning classifier and an embedded hardware target.

**Key highlights:**

| Aspect | Detail |
|---|---|
| Dataset | MIT-BIH Arrhythmia Database (PhysioNet), 48 records |
| Classes | ST-segment, QT-interval, P-wave anomaly, Bundle Branch Block, Normal (5-class multi-label) |
| Model | Multi-Scale 1D-CNN with parallel branches (kernel sizes 3, 5, 7) |
| Quantization | INT8 Sensitivity-Preserving QAT via Brevitas |
| Max sensitivity drop allowed | **2.0%** per class (vs. FP32 baseline) |
| Current Hardware Target | **Arduino Giga R1** (STM32H747XI MCU) *(Note: original FPGA path archived in `hardware_fpga_archive/`)* |
| Weight handoff format | C++ `weights.h` header arrays (and `.mem` files for archived FPGA path) + `weights_manifest.json` |

---

## Current Hardware Target

> **Active Migration Notice:** The project has migrated its hardware deployment target from FPGA (Verilog/Vivado) to the **Arduino Giga R1 WiFi** (STM32H747XI dual-core ARM Cortex-M7 @ 480 MHz / Cortex-M4 @ 240 MHz).
> - Active firmware, 16-bit ADC sampling @ 360 Hz, diagnostic serial streaming, and C++ inference stubs reside in [`hardware_giga/`](file:///home/mallhar/Downloads/rns-major-project/ecg-fpga-accelerator/hardware_giga/).
> - The original FPGA RTL source code, constraints, testbenches, and Vivado project files have been archived for reference in [`hardware_fpga_archive/`](file:///home/mallhar/Downloads/rns-major-project/ecg-fpga-accelerator/hardware_fpga_archive/).

---

## Repository Structure

```
ecg-fpga-accelerator/
├── software/
│   ├── config/
│   │   └── config.yaml              # Single source of truth for all hyperparameters
│   ├── scripts/
│   │   ├── download_dataset.py      # PhysioNet MIT-BIH downloader
│   │   ├── train.py                 # Phase 1: FP32 training entrypoint
│   │   ├── evaluate.py              # Evaluate model on test set, save metrics JSON
│   │   ├── run_qat.py               # Phase 2: SP-QAT + hardware export entrypoint
│   │   └── validate_mem_files.py    # Verify .mem files match manifest
│   ├── src/
│   │   ├── data/
│   │   │   ├── loader.py            # PhysioNet record loader (wfdb)
│   │   │   ├── preprocessor.py      # Windowing, annotation mapping, inter-patient split
│   │   │   └── dataset.py           # PyTorch Dataset + make_dataloaders()
│   │   ├── models/
│   │   │   ├── multiscale_cnn.py    # FP32 Multi-Scale 1D-CNN
│   │   │   └── quantized_cnn.py     # INT8 Brevitas-quantized version
│   │   ├── training/
│   │   │   ├── loss.py              # MorphologyWeightedBCELoss
│   │   │   ├── metrics.py           # Clinical metrics (sensitivity, specificity, F1)
│   │   │   └── trainer.py           # Training loop, checkpointing, early stopping
│   │   └── quantization/
│   │       ├── sp_qat.py            # Sensitivity-Preserving QAT pipeline
│   │       └── export.py            # INT8 weight extraction to .mem & weights.h C++ files
│   ├── outputs/
│   │   ├── checkpoints/             # Model checkpoints (.pth) — git-ignored
│   │   ├── mem_files/               # Hardware handoff files (.mem + weights.h + manifest) — git-ignored
│   │   ├── logs/                    # Training/QAT logs — git-ignored
│   │   └── plots/                   # test_metrics.json, training curves
│   ├── data/
│   │   ├── raw/                     # Downloaded PhysioNet records — git-ignored
│   │   └── processed/               # X.npy, y.npy after preprocessing — git-ignored
│   ├── tests/                       # pytest test suite
│   └── requirements.txt
├── hardware_giga/                   # Active Arduino Giga R1 firmware & diagnostics
│   ├── README.md                    # Giga hardware documentation
│   ├── firmware/                    # .ino sketch, adc_acquisition, diagnostics, inference stub
│   └── tests/golden_vectors/        # C++ vs Python output comparison test vectors
├── hardware_fpga_archive/           # Legacy FPGA implementation archive (Verilog, XDC, Vivado)
│   ├── README.md                    # Archive reference notice
│   ├── rtl/                         # Verilog/SystemVerilog RTL source files
│   ├── tb/                          # Simulation testbenches
│   ├── constraints/                 # Vivado XDC pin/timing constraints
│   └── vivado/                      # Vivado project files
└── .gitignore
```

> **Note:** `software/data/`, `software/outputs/checkpoints/`, `software/outputs/mem_files/`, and `software/outputs/logs/` are git-ignored as they may be large or machine-generated. Run the pipeline locally to regenerate them.

---

## Software Documentation

### Architecture

The software pipeline is structured in two phases:

```
Phase 1 (FP32 Training)
  ├── Download MIT-BIH records (wfdb / PhysioNet)
  ├── Preprocess: windowing, annotation mapping, inter-patient split
  ├── Train MultiScale1DCNN with MorphologyWeightedBCELoss
  ├── Evaluate on held-out test set → test_metrics.json
  └── Save best_model.pth

Phase 2 (INT8 SP-QAT + Export)
  ├── Load FP32 weights into QuantizedMultiScale1DCNN (Brevitas)
  ├── Calibrate scale factors (100 batches)
  ├── QAT fine-tuning (20 epochs, lr=1e-4)
  ├── Validate: INT8 sensitivity drop ≤ 2% vs FP32 baseline per class
  ├── Save best_qat_model.pth
  └── Export → .mem files + weights_manifest.json + weights.h
```

---

### Dataset & Preprocessing

**Source:** [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) — 48 half-hour ECG recordings, sampled at **360 Hz**, 2 leads.

**Inter-patient split (AAMI standard, Chazal et al. 2004):**
- Records are split at the **patient level** to prevent data leakage.
- **Training records (44):** 100, 101, 103, 105, 106, 108, 109, 111–119, 121–124, 200–215, 219–223, 228, 230, 231, 233, 234
- **Test records (4):** 104, 107, 217, 232
- Validation is carved from training records at a **15% fraction**.

**Annotation mapping (clinical decisions):**

| Class | MIT-BIH Symbols | Description |
|---|---|---|
| `ST_segment` | `S`, `J` | ST-change, J-point elevation (ischaemia marker) |
| `QT_interval` | `f`, `Q`, `q` | Fusion beat, unclassifiable (QT morphology) |
| `P_wave` | `A`, `a`, `e`, `j` | Atrial premature / escape / aberrant (pre-AF) |
| `Bundle_Branch_Block` | `B`, `L`, `R`, `r` | BBB, LBBB, RBBB, R-on-T |
| `Normal` | `N`, `.` | Normal sinus rhythm, paced beat |

**Class index is IMMUTABLE** — the hardware inference interface is built around index order `[0]=ST_segment [1]=QT_interval [2]=P_wave [3]=Bundle_Branch_Block [4]=Normal`.

**Windowing:**
- Window size: **256 samples** (~711 ms at 360 Hz), centred on each annotated beat
- Lead used: **Lead 0** (MLII)

---

### Model Architecture

#### FP32 Baseline — `MultiScale1DCNN`

```
Input: (batch, 1, 256)

MultiScaleLayer × 3
  ├── Parallel Conv1d branches: kernel sizes [3, 5, 7, 9]
  ├── Channel concatenation
  └── BatchNorm1d → ReLU

MaxPool1d(2) after each of the first two layers
AdaptiveAvgPool1d(1) after the third layer

Flatten → Linear(base_filters×4, num_classes)
Output: raw logits (batch, 5) — NO sigmoid (applied at loss/inference time)
```

#### INT8 Quantized — `QuantizedMultiScale1DCNN`

Mirrors the FP32 architecture exactly but replaces:
- `nn.Conv1d` → `brevitas.nn.QuantConv1d` (INT8 weight + activation)
- `nn.Linear` → `brevitas.nn.QuantLinear` (INT8 weight + activation)

```
Input: (batch, 1, 256)

3 parallel QuantConv1d branches (kernel sizes [3, 5, 7], out_channels=32 each)
  └── BatchNorm1d → ReLU → AdaptiveAvgPool1d(64)

Concat (dim=1) → Flatten
  → QLinear(6144, 128) → ReLU → Dropout(0.3)
  → QLinear(128, 5)

Output: raw logits (batch, 5)
```

**Quantization scheme:**
- Weights: `Int8WeightPerTensorFloat` (per-tensor symmetric)
- Activations: `Int8ActPerTensorFloat` (per-tensor symmetric)
- All layers except the final output layer return `QuantTensor`

---

### Training Pipeline (Phase 1)

#### Loss — `MorphologyWeightedBCELoss`

Custom `BCEWithLogitsLoss` with **per-class morphology weights** reflecting clinical severity:

| Class | Weight | Rationale |
|---|---|---|
| ST_segment | 3.0 | Ischaemia marker — false negatives are dangerous |
| QT_interval | 3.5 | Highest risk — linked to sudden cardiac death |
| P_wave | 2.5 | Atrial ectopics — precursor to atrial fibrillation |
| Bundle_Branch_Block | 2.0 | Conduction disease — requires monitoring |
| Normal | 1.0 | Baseline reference |

Weights are loaded from `config.yaml` — never hardcoded in source.

#### Trainer

- **Optimizer:** Adam (lr=0.001, weight_decay=0.0001)
- **Epochs:** up to 50 with **early stopping** (patience=10)
- **Checkpointing:** every 5 epochs + best model by val F1-macro
- **Reproducibility:** global seed 42 applied to PyTorch, NumPy, and Python `random`

**Checkpoint format (`best_model.pth`):**
```python
{
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'epoch': int,
    'val_f1': float,
    'config': dict
}
```

---

### Quantization-Aware Training — SP-QAT (Phase 2)

The `SPQATPipeline` in `src/quantization/sp_qat.py` executes the following steps in order:

#### 1. `load_fp32_weights`
Transfers weights from `best_model.pth` into the quantized model using `strict=False`. Logs matched vs. skipped keys (architecture changes between FP32 and quantized model result in some keys being skipped by design).

#### 2. `calibrate`
Runs **100 calibration batches** through the quantized model in `eval()` mode with `torch.no_grad()` to initialize Brevitas scale factors before training begins.

#### 3. `train_qat`
Fine-tunes the quantized model for **20 epochs** at `lr=1e-4` using the same `MorphologyWeightedBCELoss`. Saves the best checkpoint to `outputs/checkpoints/best_qat_model.pth` by validation F1-macro.

#### 4. `evaluate_and_validate`
Evaluates the INT8 model on the test set and enforces the **clinical sensitivity constraint**:

```
For each class c:
    drop = (fp32_sensitivity[c] - int8_sensitivity[c]) * 100
    if drop > 2.0%:
        → RuntimeError: QAT pipeline is aborted
```

FP32 baseline sensitivities are loaded from `software/outputs/plots/test_metrics.json` (generated by `evaluate.py`). If the file is missing, the following immutable defaults are used:

| Class | FP32 Baseline Sensitivity |
|---|---|
| ST_segment | 0.9993 |
| QT_interval | 0.7540 |
| P_wave | 0.9137 |
| Bundle_Branch_Block | 0.7040 |
| Normal | 0.9860 |

---

### Hardware Weight Export

`src/quantization/export.py` provides export routines for both active (Arduino Giga R1) and archived (FPGA) targets:

1. **C++ Header Export (`export_weights_to_c_header`)**: Generates `weights.h` containing `#ifndef/#define/#endif` include guards and `const int8_t layer_*_weights[]` arrays alongside `shape` and `scale_factor` metadata.
2. **FPGA `.mem` Export (`export_weights_to_mem`)**: Generates 2-digit uppercase hex `.mem` files for Verilog `$readmemh` hardware loading.
3. **Manifest (`weights_manifest.json`)**: Contains layer shape, total value count, scale factor, and filenames.

---

### Configuration Reference

All hyperparameters live in `software/config/config.yaml`. Key sections:

```yaml
data:
  sample_rate: 360          # MIT-BIH native sample rate (Hz)
  window_size: 256          # Samples per ECG window (~711 ms)
  lead_index: 0             # MLII lead
  val_split_fraction: 0.15  # Fraction of train records for validation

model:
  branch_out_channels: 32   # Output channels per parallel branch
  kernel_sizes: [3, 5, 7]   # Parallel branch kernel sizes
  pool_output_size: 64      # AdaptiveAvgPool output size per branch
  fc_hidden_size: 128       # FC hidden layer size
  dropout_rate: 0.3
  num_classes: 5
  class_weights: [3.0, 3.5, 2.5, 2.0, 1.0]  # Clinical severity weights

training:
  epochs: 50
  batch_size: 64
  learning_rate: 0.001
  optimizer: adam
  weight_decay: 0.0001
  checkpoint_every_n_epochs: 5
  early_stopping_patience: 10

qat:
  epochs: 20
  lr: 0.0001
  calibration_batches: 100
  weight_bits: 8
  activation_bits: 8
  max_sensitivity_drop_pct: 2.0   # Hard clinical constraint

export:
  output_dir: software/outputs/mem_files

inference:
  threshold: 0.5
  window_size: 256
```

> **Warning:** The `class_names` list order in `config.yaml` is immutable. Do not reorder. Hardware inference relies on index positions `[0..4]` matching `[ST_segment, QT_interval, P_wave, Bundle_Branch_Block, Normal]`.

---

### Setup & Installation

**Prerequisites:**
- Python 3.10+
- A virtual environment (strongly recommended)

```bash
# 1. Clone the repository
git clone git@github.com:Mallhar03/ecg-fpga-accelerator.git
cd ecg-fpga-accelerator

# 2. Create and activate virtual environment
python3 -m venv ecg-fpga-venv
source ecg-fpga-venv/bin/activate   # Linux/macOS
# .\ecg-fpga-venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r software/requirements.txt
```

**Key dependencies:**

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.1.2+cpu | Model training and inference |
| `brevitas` | 0.10.2 | Quantization-Aware Training |
| `wfdb` | 4.1.2 | PhysioNet record I/O |
| `numpy` | 1.26.4 | Numerical computation |
| `scikit-learn` | 1.4.2 | Evaluation utilities |
| `PyYAML` | 6.0.1 | Config file parsing |
| `tqdm` | 4.66.4 | Progress bars |

---

### Running the Pipeline

> All commands must be run from the **repository root** (`ecg-fpga-accelerator/`).

#### Step 0: Activate environment

```bash
source ../ecg-fpga-venv/bin/activate
```

#### Step 1: Download the MIT-BIH dataset

```bash
# Download all records defined in config.yaml (train + test)
python software/scripts/download_dataset.py \
    --config software/config/config.yaml \
    --output software/data/raw/ \
    --all

# Or download specific records:
python software/scripts/download_dataset.py \
    --output software/data/raw/ \
    --records 100 101 104 107
```

Requires free registration at [physionet.org](https://physionet.org).

#### Step 2: Preprocess the data

Preprocessing is invoked automatically inside `make_dataloaders()` when training or evaluation scripts are first run. The processed `X.npy` and `y.npy` are saved to `software/data/processed/`.

#### Step 3: Train the FP32 model (Phase 1)

```bash
python software/scripts/train.py \
    --config software/config/config.yaml
```

Optional — resume from a checkpoint:
```bash
python software/scripts/train.py \
    --config software/config/config.yaml \
    --resume software/outputs/checkpoints/epoch_15.pth
```

Output: `software/outputs/checkpoints/best_model.pth`

#### Step 4: Evaluate the FP32 model

```bash
python software/scripts/evaluate.py \
    --config software/config/config.yaml \
    --checkpoint software/outputs/checkpoints/best_model.pth
```

This prints a clinical metrics table and writes `software/outputs/plots/test_metrics.json` — required as the FP32 baseline for Phase 2.

**Example output:**
```
Class                     Sensitivity  Specificity  F1
-----------------------------------------------------------------
ST_segment                0.9993       0.9987       0.9990
QT_interval               0.7540       0.9912       0.8102
P_wave                    0.9137       0.9845       0.9477
Bundle_Branch_Block       0.7040       0.9991       0.8249
Normal                    0.9860       0.9802       0.9831
-----------------------------------------------------------------
Macro Average             0.8714       0.9907       0.9130
```

#### Step 5: Run SP-QAT and export weights (Phase 2)

```bash
python software/scripts/run_qat.py \
    --config software/config/config.yaml \
    --fp32-checkpoint software/outputs/checkpoints/best_model.pth
```

This pipeline:
1. Loads FP32 weights, calibrates scale factors
2. Runs 20 epochs of QAT fine-tuning
3. Validates INT8 sensitivity (fails loudly if any class drops >2%)
4. Exports `weights.h` (C++), `.mem` files (Verilog), and `weights_manifest.json`

Output: `software/outputs/mem_files/`

#### Step 6: Validate hardware weight files

```bash
python software/scripts/validate_mem_files.py \
    --dir software/outputs/mem_files/
```

Expected output: `All .mem files valid.`

#### Running Tests

```bash
pytest software/tests/ -v
```

---

### Evaluation & Metrics

The `compute_clinical_metrics()` function in `src/training/metrics.py` computes **per-class** and **macro-average** metrics:

| Metric | Formula | Clinical relevance |
|---|---|---|
| **Sensitivity** | TP / (TP + FN) | Probability of correctly detecting a condition |
| **Specificity** | TN / (TN + FP) | Probability of correctly ruling out a condition |
| **F1 Score** | 2·P·R / (P+R) | Harmonic mean of precision and recall |

Metrics are computed at a decision **threshold of 0.5** (configurable in `config.yaml → inference.threshold`).

Multi-label classification: a single ECG window can belong to multiple classes simultaneously.

---

## Hardware Documentation

### Current Hardware Target: Arduino Giga R1

Active hardware development targets the **Arduino Giga R1 WiFi** powered by the **STM32H747XI** dual-core microcontroller (ARM Cortex-M7 @ 480 MHz main core + Cortex-M4 @ 240 MHz).

#### Firmware Structure (`hardware_giga/`)
- `firmware/ecg_giga_main.ino`: Top-level Arduino sketch initializing hardware subsystems and driving the main loop.
- `firmware/adc_acquisition.h/.cpp`: 16-bit SAR ADC acquisition engine configured at an explicit **360 Hz sample rate** (`SAMPLE_RATE_HZ = 360`) matching the native sampling rate of the PhysioNet MIT-BIH dataset. Uses `Arduino_AdvancedAnalog` with 32-sample DMA buffers (`DMABuffer<Sample>`) and internal queue release (`current_buf.release()`).
- `firmware/signal_diagnostics.h/.cpp`: Real-time diagnostic logging over Serial in CSV format (`timestamp,raw_value`) for external plotting and signal saturation/clipping analysis.
- `firmware/inference.h/.cpp`: Documented C++ stub for INT8 Multi-Scale 1D-CNN inference (to be fully integrated post-ADC front-end signal quality validation).

#### C++ Weight Export (`weights.h`)
Weights from Brevitas INT8 QAT are exported into C++ arrays using `export_weights_to_c_header()` in `software/src/quantization/export.py`. The resulting `weights.h` header contains `#ifndef/#define/#endif` include guards and `const int8_t layer_*_weights[]` arrays alongside original shape and `scale_factor` metadata.

#### Current Status
The firmware is in the **ADC signal acquisition & quality diagnostics stage**, streaming raw 16-bit 360 Hz samples over USB Serial to diagnose and resolve front-end analog signal distortion before enabling online INT8 CNN classification.

---

### Archived: Original FPGA Implementation

> **Archive Notice:** This section describes the original FPGA-based hardware path, now archived in [`hardware_fpga_archive/`](file:///home/mallhar/Downloads/rns-major-project/ecg-fpga-accelerator/hardware_fpga_archive/). Active development targets the Arduino Giga R1 — see the Giga R1 section above.

#### FPGA Target & Toolchain

- **Target FPGA:** Xilinx/AMD series (Vivado project files in `hardware_fpga_archive/vivado/`)
- **Synthesis Tool:** Vivado Design Suite
- **HDL:** Verilog / SystemVerilog
- **Constraints:** XDC format (`hardware_fpga_archive/constraints/`)

#### RTL Architecture

The hardware accelerator implements the `QuantizedMultiScale1DCNN` inference graph in RTL, consuming INT8 weights pre-loaded from `.mem` files via `$readmemh`.

**Top-level data flow:**

```
ADC / ECG Input (256×INT16 samples)
        ↓
  [Input Buffer / Window Register]
        ↓
  [Branch 0]  [Branch 1]  [Branch 2]
  Conv1D k=3  Conv1D k=5  Conv1D k=7    ← INT8 MACs
     ↓            ↓            ↓
  BN+ReLU      BN+ReLU      BN+ReLU
     ↓            ↓            ↓
  AvgPool64   AvgPool64    AvgPool64
        ↓
  [Concatenation: 3×32×64 = 6144 channels]
        ↓
  [FC Layer 1: 6144 → 128, INT8]
        ↓  ReLU
  [FC Layer 2: 128 → 5, INT8]
        ↓
  [5-bit Output Register: class logits]
        ↓
  [Threshold comparator (0.5 × scale)]
        ↓
  CLASSIFICATION OUTPUT [5 bits, one-hot or multi-hot]
```

All multiply-accumulate (MAC) operations use **INT8 arithmetic**. Scale factors from `weights_manifest.json` are used by RTL for dequantization at the output stage.

#### Weight File Format

`.mem` files follow Verilog `$readmemh` convention:
- **One weight value per line**
- **2-digit uppercase hexadecimal** (e.g., `7F`, `FF`, `00`, `80`)
- **Two's complement** representation (e.g., `-1` → `FF`, `-128` → `80`)
- **Row-major (C) order** — the flattening matches PyTorch's default weight layout

**Example:** `layer_branches_0_0_weights.mem` (first 4 lines)
```
3D
F2
0A
C1
```

To instantiate in RTL:
```verilog
reg signed [7:0] branch0_weights [0:95]; // 32 × 1 × 3 = 96 values
initial $readmemh("layer_branches_0_0_weights.mem", branch0_weights);
```

#### Weights Manifest

`weights_manifest.json` provides metadata for each layer:

```json
{
  "branches.0.0": {
    "shape": [32, 1, 3],
    "num_values": 96,
    "scale_factor": 0.004298259504139423,
    "mem_file": "layer_branches_0_0_weights.mem"
  },
  "branches.1.0": {
    "shape": [32, 1, 5],
    "num_values": 160,
    "scale_factor": 0.003570995293557644,
    "mem_file": "layer_branches_1_0_weights.mem"
  },
  "branches.2.0": {
    "shape": [32, 1, 7],
    "num_values": 224,
    "scale_factor": 0.003073744475841522,
    "mem_file": "layer_branches_2_0_weights.mem"
  },
  "fc1": {
    "shape": [128, 6144],
    "num_values": 786432,
    "scale_factor": 0.001872877823188901,
    "mem_file": "layer_fc1_weights.mem"
  },
  "fc2": {
    "shape": [5, 128],
    "num_values": 640,
    "scale_factor": 0.0031097440514713526,
    "mem_file": "layer_fc2_weights.mem"
  }
}
```

#### Constraints

Timing and pin constraints for the archived FPGA implementation are in `hardware_fpga_archive/constraints/`. These include:
- Clock period definitions
- I/O pin assignments for ADC interface
- Output logic for classification flag registers

#### Simulation & Testbenches

Testbench files in `hardware_fpga_archive/tb/` provide:
- Weight loading verification (checks `$readmemh` output against manifest)
- Datapath functional simulation: stimulates with known ECG windows and checks output flags
- Timing simulation post-synthesis

To run simulation with Vivado:
```tcl
# In Vivado Tcl console:
open_project hardware_fpga_archive/vivado/<project>.xpr
launch_simulation
```

---

### Hardware–Software Interface

The software pipeline and hardware targets share a strict interface contract:

| Contract | Value | Application |
|---|---|---|
| Input window size | 256 samples (~711 ms @ 360 Hz) | Both Giga R1 & Archived FPGA |
| Input data type | INT8 / 16-bit ADC | Both Giga R1 & Archived FPGA |
| Weight data type | `const int8_t` arrays in `weights.h` (Arduino Giga R1) / `.mem` two's complement hex (Archived FPGA) | Target-dependent |
| Dequantization | `output = int8_value × scale_factor` | Both Giga R1 & Archived FPGA |
| Class output order | `[0]=ST [1]=QT [2]=P_wave [3]=BBB [4]=Normal` | Both Giga R1 & Archived FPGA |
| Inference threshold | 0.5 (applied post-dequantization) | Both Giga R1 & Archived FPGA |

The class output bit-vector / array maps directly to clinical diagnosis flags:
- **Bit 0 (ST_segment):** ST-elevation / depression detected
- **Bit 1 (QT_interval):** QT morphology abnormality detected
- **Bit 2 (P_wave):** Atrial ectopic beat detected
- **Bit 3 (Bundle_Branch_Block):** Bundle branch block detected
- **Bit 4 (Normal):** Normal sinus rhythm

---

## Clinical Design Decisions

Several design decisions were made specifically to protect clinical correctness:

1. **Inter-patient split** — train/test records are separated at the patient level following the AAMI EC57 standard (Chazal et al. 2004). Beat-level splitting would leak patient-specific morphologies into the test set.

2. **Morphology-weighted loss** — QT-interval (sudden death risk, weight=3.5) and ST-segment (ischaemia, weight=3.0) receive the highest training penalties for false negatives. This directly penalises missing dangerous conditions more than misclassifying normal beats.

3. **Sensitivity-preserving QAT** — The 2% hard cap on sensitivity drop per class ensures quantization does not compromise detection capability for rare but high-risk arrhythmias. The pipeline raises a `RuntimeError` and refuses to export hardware weights if this constraint is violated.

4. **Immutable class ordering** — The 5-class index is frozen and documented across `config.yaml`, `metrics.py`, the `QuantizedMultiScale1DCNN` output layer, and any hardware inference implementation (RTL in the archived FPGA path, or C++ on the current Arduino Giga R1 target). Any reordering would silently break clinical interpretation of hardware outputs.

5. **Per-window normalization fallback** — If training statistics are unavailable (e.g., during synthetic dry-runs), the dataset falls back to per-window z-score normalization. This is clearly logged as a warning.

---

## Development Workflow & Branching

| Branch | Purpose |
|---|---|
| `main` | Stable, tested code only |
| `feature/phase-1` | Phase 1: Data pipeline, FP32 model, training infrastructure |
| `feature/phase-2` | Phase 2: SP-QAT pipeline, quantized model, hardware export |

**Commit convention:**
```
[Phase N] Author: Short description. Closes #issue.
```

The upstream repository is tracked at: https://github.com/Sudarshan-S-V/ecg-fpga-accelerator

---

## Contributors

| Name | GitHub | Contributions |
|---|---|---|
| Mallhar | [@Mallhar03](https://github.com/Mallhar03) | Project structure, `config.yaml`, data loader, QAT pipeline, hardware export, Giga R1 migration |
| Abhishek | [@TheBeast2208](https://github.com/TheBeast2208) | ECG preprocessor, PyTorch dataset, full test suite (Phase 1A) |
| Sudarshan | — | Repository upstream owner (Phase 1B: loss functions, additional tests) |

---

*For questions about hardware integration, refer to `hardware_giga/` directory and the [Hardware Documentation](#hardware-documentation) section. For software pipeline issues, open a GitHub issue.*
