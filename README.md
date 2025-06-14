# Keyword Spotting on STM32 L4

**Course:** Machine Learning on Microcontrollers (ETH Zürich, FS25)  

## 🎯 Project Overview

This project implements a **Keyword Spotting** (KWS) system that can classify simple spoken words using a low-power **STM32 L4** microcontroller. The model is designed to run **entirely on-device**, making it suitable for **always-on** applications in embedded systems without relying on cloud-based inference.

## 🧠 Motivation

- Enable **natural and intuitive user interaction** with embedded systems via voice.
- Design a **lightweight, efficient** model for **on-edge inference**.
- Achieve real-time performance within the **limited memory and compute** constraints of a microcontroller.

## 🗃️ Dataset

- **Google Speech Commands v0.02**
  - 35 command words, ~2100–3800 samples each.
  - **8 selected words** for this project.
  - Original sampling rate: 16 kHz → **Downsampled to 8 kHz**.

To download the dataset: 
```bash
mkdir dataset/
cd dataset/
wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar -xvf speech_commands_v0.02.tar.gz
```

## 🔧 Tools & Platform

| Stage             | Tools Used                                |
|------------------|--------------------------------------------|
| Model Design      | PyTorch                                    |
| Quantization      | ONNX (8-bit weights + activations)         |
| Deployment        | STM32CubeIDE + X-Cube-AI                   |
| Target MCU        | **STM32 L475**                             |

## 🧪 Model Architecture

- Preprocessing: **MFCC** feature extraction
- Model: Compact **CNN**
  - Layers:
    - `Conv2D + BatchNorm → ReLU → Conv2D → BatchNorm → ReLU → Pool`
    - Repeated with increasing channels
    - `GlobalAvgPool → Dense → ReLU → Dense`
- Training:
  - **SGD + momentum + L2 regularization**
  - **Cosine learning rate scheduler**
  - **100 epochs**, batch size 128

## 📦 Quantization & Optimization

To meet the memory and performance constraints of the MCU:
- **Post-training linear quantization** using ONNX
- Calibration on validation set
- Results:
  - **Flash**: 280.61 KiB → 101.57 KiB (**-65%**)
  - **RAM**: 279.30 KiB → 78.20 KiB (**-72%**)
  - **Accuracy** 93% → 93% (**-0%**)

## 📈 Evaluation

- **On-device inference** was tested and benchmarked.
- Multiple model variants were explored for **accuracy/latency trade-offs**.
