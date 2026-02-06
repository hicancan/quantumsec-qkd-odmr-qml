# Quantum Measurement–Driven Security Lab (QKD + ODMR + QML)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Quantum Backends](https://img.shields.io/badge/Quantum-PennyLane%20%7C%20Qiskit%20%7C%20Cirq%20%7C%20CUDA--Q-purple)](https://pennylane.ai/)

**QuantumSec-QKD-ODMR-QML** is a research-grade, **pure Python** framework that unifies **Information Security × Quantum Measurement × AI/QML** into a reproducible pipeline. It demonstrates how quantum measurement statistics act as "security sensors" in two distinct domains: identifying eavesdroppers in Quantum Key Distribution (QKD) and detecting anomalies in Quantum Sensing (NV-ODMR).

[**English**](#english) | [**中文 (Chinese)**](#chinese)

---

<a name="english"></a>
## 🇬🇧 English Description

### 🌟 Key Features

*   **Unified Security Pipeline (Task A & B)**:
    *   **Task A (QKD / BB84)**: Simulates the BB84 protocol to generate Quantum Bit Error Rate (QBER). We map QBER to **Secret Key Rate (SKR)** and use a Quantum Support Vector Machine (QSVM) to detect eavesdropping attacks (acting as an Intrusion Detection System).
    *   **Task B (NV-ODMR)**: Simulates NV Center ODMR spectra. We extract spectral features and utilize quantum kernels to detect magnetic field anomalies or sensor integrity issues.
*   **Multi-Backend Support**: Implements *identical quantum kernel feature maps* across **PennyLane**, **Qiskit Aer**, **Cirq**, and **CUDA-Q**, with automated cross-validation to ensure numerical equivalence.
*   **Scientific Rigor**: Built-in calculation of information-theoretic bounds (entropy, fidelity) and rigorous validation of physical monotonicity.

### 🚀 Quick Start

#### 1. Installation
This repository follows a standard `src/` layout. We recommend using a virtual environment (Conda or venv).

```bash
# Create and activate environment
conda create -n quantum-sec python=3.9
conda activate quantum-sec

# Install dependencies
pip install -r requirements.txt
```

> **Note**: If you already have a quantum stack installed (PennyLane, Qiskit, etc.), you can skip full installation. CUDA-Q is optional if you lack a compatible environment.

#### 2. Usage (Reproducible Scripts)
We provide a unified CLI to run experiments without modifying `PYTHONPATH` manually.

```bash
# Recommended: Run full experimental suite (BB84 + ODMR + Kernel Benchmarks)
./scripts/run_all.sh

# OR run individual modules via CLI
# 1. Run BB84 Simulation & Analysis
python -m qmsl.cli bb84 --backend numpy --seed 0 --out results

# 2. Run ODMR Simulation & Anomaly Detection
python -m qmsl.cli odmr --seed 0 --out results

# 3. Benchmark Quantum Kernels across Backends
python -m qmsl.cli kernel-bench --seed 0 --out results
```

**Outputs**:
- `results/metrics.json`: JSON report containing AUC, Accuracy, and SKR metrics.
- `docs/figures/*.png`: Generated plots (ROC curves, Confusion Matrices, Spectra).

### 📚 Repository Structure

- `src/qmsl/datasets/`: Data generators for BB84 (QBER stats) and ODMR (Lorentzian spectra).
- `src/qmsl/kernels/`: Quantum kernel implementations (PennyLane, Qiskit, Cirq, CUDA-Q).
- `src/qmsl/models/`: Classical baselines (SVM) and QSVM integration.
- `src/qmsl/eval/`: Evaluation metrics (AUC, Signal-to-Noise) and plotting tools.
- `docs/`: Supplementary documentation and figures.

### 📖 Citation
If you use this code in your research, please cite:
```bibtex
@misc{purohit2025qml,
  title = {Quantum Machine Learning for Quantum Key Distribution and Sensing},
  author = {Purohit, A. and Vyas, V.},
  year = {2025},
  note = {See refs/ for full bibliography}
}
```

---

<a name="chinese"></a>
## 🇨🇳 中文介绍 (Chinese Description)

### 🌟 核心特性

*   **统一安全流水线 (任务 A & B)**:
    *   **任务 A (QKD / BB84)**: 模拟 BB84 协议生成量子比特误码率 (QBER)。我们将 QBER 映射到信息论中的 **安全密钥率 (SKR)**，并使用量子支持向量机 (QSVM) 构建类似入侵检测系统 (IDS) 的防御机制。
    *   **任务 B (NV-ODMR)**: 模拟 NV 色心 ODMR 光谱。提取光谱特征并利用量子核函数检测磁场异常或传感器完整性问题。
*   **多后端支持**: 在 **PennyLane**, **Qiskit Aer**, **Cirq**, 和 **CUDA-Q** 上实现了*完全一致的量子核特征映射*，并通过了严格的数值交叉验证。
*   **科学严谨性**: 内置香农熵、保真度等物理量的计算，确保模拟结果符合物理学理论边界。

### 🚀 快速开始

#### 1. 安装环境
本项目采用标准的 python `src/` 包结构。建议使用 Conda 或 venv 管理环境。

```bash
# 创建并激活环境
conda create -n quantum-sec python=3.9
conda activate quantum-sec

# 安装依赖
pip install -r requirements.txt
```

> **注意**: 如果您已安装常用的量子计算库，可跳过完整安装。CUDA-Q 为可选依赖。

#### 2. 运行实验
可以通过 CLI 轻松复现所有实验结果。

```bash
# 方案 A: 一键运行全套实验 (推荐)
./scripts/run_all.sh

# 方案 B: 单独运行模块
# 1. BB84 模拟与分析
python -m qmsl.cli bb84 --backend numpy --seed 0 --out results

# 2. ODMR 模拟与异常检测
python -m qmsl.cli odmr --seed 0 --out results

# 3. 跨后端量子核基准测试
python -m qmsl.cli kernel-bench --seed 0 --out results
```

**输出产物**:
- `results/metrics.json`: 包含 AUC、准确率 (Accuracy) 和安全密钥率等关键指标。
- `docs/figures/*.png`: 自动生成的 ROC 曲线、混淆矩阵和光谱分布图。

### 📂 项目结构

- `src/qmsl/datasets/`: BB84 (QBER 统计) 与 ODMR (洛伦兹光谱) 数据生成器。
- `src/qmsl/kernels/`: 适配多种后端的量子核函数实现 (PennyLane, Qiskit, Cirq, CUDA-Q)。
- `src/qmsl/models/`: 经典机器学习基线 (SVM) 与 QSVM 实现。
- `src/qmsl/eval/`: 评估指标 (AUC, SNR) 与绘图工具。
- `docs/`: 项目文档与图表。

> 📚 **深度科普**: [当信息安全遇见量子测量——基于 Purohit & Vyas (2025) 综述的 QML-IDS 复现探索](docs/wechat_article.md)

### 📜 许可证 (License)
本项目基于 [MIT License](LICENSE) 开源。
