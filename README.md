# quantumsec-qkd-odmr-qml

**[中文版 README](README_zh.md)**

A research-grade, **pure-Python** repository that unifies **Information Security × Quantum Measurement × AI/QML** in one reproducible pipeline.

This project demonstrates how quantum measurement statistics can serve as a "security sensor" in two distinct domains: identifying eavesdroppers in Quantum Key Distribution (QKD) and detecting anomalies in Quantum Sensing (NV-ODMR).

## Key Features

- **Task A (QKD / BB84):** Simulates the BB84 protocol to generate quantum bit error rates (QBER). We map QBER to information-theoretic **Secret Key Rate (SKR)** and use Quantum SVMs (QSVM) to detect eavesdropping attacks (IDS-style).
- **Task B (NV-ODMR):** Simulates NV-center ODMR spectroscopy. We extract spectral features to detect magnetic field anomalies or sensor integrity issues using quantum kernels.
- **Hard-core Reproducibility:** The *same quantum kernel feature map* is implemented and cross-validated across **PennyLane**, **Qiskit Aer**, **Cirq**, and **CUDA-Q**.
- **Automated Reporting:** Generates metric JSONs, figures, and an auto-filled WeChat-style Markdown article.

> **Deliverables:** CLI tools, rigorous pytest suite, bibliography, and ready-to-publish analysis.

## Quick start

### 1) Install environment
This repo is structured as a `src/` package. It is recommended to create and activate a virtual environment (Conda or venv) first.

```bash
# Example (Conda):
# conda create -n my-quantum-env python=3.9
# conda activate my-quantum-env

pip install -U pip
pip install -r requirements.txt
```

> **Note:** If you already have a quantum stack installed (PennyLane, Qiskit, Cirq, CUDA-Q, QuTiP, SymPy), you can skip the full requirements install. CUDA-Q is optional if you don't have a compatible GPU/environment.

### 2) Install package (Editable mode)

```bash
pip install -e .
```

### 3) Run the full pipeline (One command)

```bash
python -m qmsl.cli run-all --seed 0 --out results
```

**Outputs:**
- `results/metrics.json`: All computed metrics.
- `docs/figures/*.png`: Generated ROC curves, confusion matrices, and data plots.
- `docs/wechat_article.md`: A summary article filled with the actual metrics from your run.

### 4) Run individual modules

```bash
# Run BB84 Simulation & Analysis
python -m qmsl.cli bb84 --backend numpy --seed 0 --out results

# Run ODMR Simulation & Anomaly Detection
python -m qmsl.cli odmr --seed 0 --out results

# Benchmark Quantum Kernels across Backends
python -m qmsl.cli kernel-bench --seed 0 --out results

# Check Environment & Versioning
python -m qmsl.cli env-check
```

### 5) Run Verification Tests

```bash
pytest -q
```
*Tests cover kernel equivalence (ensuring all backends produce identical kernel matrices) and physical monotonicity.*

## Repository Layout

- `src/qmsl/datasets/`: Data generators for BB84 (QBER stats) and ODMR (Lorentzian spectra).
- `src/qmsl/kernels/`: Analytic quantum kernels and backend-specific implementations (Cirq, Qiskit, PennyLane, CUDA-Q).
- `src/qmsl/models/`: Classical baselines (SVM, Thresholding) and QSVM implementations.
- `src/qmsl/eval/`: Metric calculations (AUC, Accuracy) and plotting utilities.
- `docs/`: Documentation and generated figures.
- `refs/`: BibTeX references.
- `tests/`: Automated tests for numerical correctness.

## Academic Note: Logic & Formulas

The project connects physical measurements to security metrics. For the BB84 protocol, we estimate the **Secret Key Rate (SKR)** using the standard asymptotic limit.

The secure key fraction is approximated by:

$$
\text{key\_fraction} \approx \max(0, 1 - 2 H_2(Q))
$$

Where $Q$ is the Quantum Bit Error Rate (QBER), and $H_2$ is the binary entropy function:

$$
H_2(p) = -p\log_2 p -(1-p)\log_2(1-p)
$$

The final rate is:

$$
\text{SKR} = \text{sift\_rate} \cdot \text{key\_fraction}
$$

When $Q$ exceeds ~11%, $H_2(Q)$ is large enough that the key fraction drops to zero, indicating no secure key can be distilled.

## License
MIT (for coursework/demo purposes).
