# External References & Further Reading

This document curates foundational academic papers, medical standards, dataset portals, and software tools relevant to this project.

---

## 1. Clinical Standards & Datasets

- **PhysioNet MIT-BIH Arrhythmia Database**:
  - Gold standard 48-record ambulatory ECG database.
  - URL: [https://physionet.org/content/mitdb/1.0.0/](https://physionet.org/content/mitdb/1.0.0/)
  - Paper: Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Engineering in Medicine and Biology Magazine, 2001.

- **AAMI EC57 Standard**:
  - *Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms.* Association for the Advancement of Medical Instrumentation, 1998/2012.
  - Defines the standard 5-class beat taxonomy (`N`, `S`, `V`, `F`, `Q`) and reporting conventions.

- **Inter-Patient Evaluation Methodology**:
  - Paper: de Chazal P, O'Dwyer M, Reilly RB. *Automatic classification of heartbeats using ECG morphology and heartbeat interval features.* IEEE Transactions on Biomedical Engineering, 2004.
  - Link: [https://doi.org/10.1109/TBME.2004.827359](https://doi.org/10.1109/TBME.2004.827359)

---

## 2. Deep Learning & Signal Processing

- **Multi-Scale Convolutional Networks**:
  - Paper: Wang Z, Yan W, Oates T. *Time series classification from scratch with deep neural networks: A strong baseline.* IJCNN, 2017.
  - Link: [https://arxiv.org/abs/1611.06455](https://arxiv.org/abs/1611.06455)

- **Squeeze-and-Excitation Networks**:
  - Paper: Hu J, Shen L, Sun G. *Squeeze-and-Excitation Networks.* CVPR, 2018.
  - Link: [https://arxiv.org/abs/1709.01507](https://arxiv.org/abs/1709.01507)

- **Focal Loss for Dense Object / Imbalanced Classification**:
  - Paper: Lin TY, Goyal P, Girshick R, He K, Dollár P. *Focal Loss for Dense Object Detection.* ICCV, 2017.
  - Link: [https://arxiv.org/abs/1708.02002](https://arxiv.org/abs/1708.02002)

---

## 3. Quantization & Embedded Tools

- **Brevitas Quantization Library**:
  - Xilinx PyTorch library for neural network quantization.
  - Repository: [https://github.com/Xilinx/brevitas](https://github.com/Xilinx/brevitas)
  - Documentation: [https://xilinx.github.io/brevitas/](https://xilinx.github.io/brevitas/)

- **WFDB Python Package**:
  - Python library for reading and processing PhysioNet waveforms.
  - Documentation: [https://wfdb.readthedocs.io/](https://wfdb.readthedocs.io/)

- **CMSIS-NN for ARM Cortex-M**:
  - Optimized neural network kernels for ARM Cortex-M processors.
  - URL: [https://github.com/ARM-software/CMSIS-NN](https://github.com/ARM-software/CMSIS-NN)

- **Arduino Giga R1 WiFi Documentation**:
  - Technical specs and pinout for STM32H747XI MCU.
  - URL: [https://docs.arduino.cc/hardware/giga-r1-wifi/](https://docs.arduino.cc/hardware/giga-r1-wifi/)
