# quantumsec-qkd-odmr-qml

一个科研级、**纯 Python** 的代码仓库，将 **信息安全 × 量子测量 × AI/QML** 统一在一个可复现的流水线中。

本项目展示了量子测量统计数据如何在两个不同的领域作为“安全传感器”发挥作用：识别量子密钥分发 (QKD) 中的窃听者，以及检测量子传感 (NV-ODMR) 中的异常情况。

## 主要特性

- **任务 A (QKD / BB84):** 模拟 BB84 协议以生成量子比特误码率 (QBER)。我们将 QBER 映射到信息论中的 **安全密钥率 (SKR)**，并使用量子支持向量机 (QSVM) 来检测窃听攻击 (类似入侵检测系统 IDS)。
- **任务 B (NV-ODMR):** 模拟 NV 色心 ODMR 光谱。我们提取光谱特征，并利用量子核函数来检测磁场异常或传感器完整性问题。
- **可复现性:** *相同的量子核特征映射* 在 **PennyLane**, **Qiskit Aer**, **Cirq**, 和 **CUDA-Q** 上均有实现并通过了交叉验证。

## 快速开始

### 1) 安装环境
本仓库采用 `src/` 包结构。建议先创建并激活一个虚拟环境 (Conda 或 venv)。

```bash
# 示例 (Conda):
# conda create -n my-quantum-env python=3.9
# conda activate my-quantum-env

pip install -U pip
pip install -r requirements.txt
```

> **注意:** 如果您已经安装了量子相关软件栈 (PennyLane, Qiskit, Cirq, CUDA-Q, QuTiP, SymPy)，可以跳过完整的依赖安装。如果您没有兼容的 GPU 或环境，CUDA-Q 是可选的。

### 2) 显式环境运行

为了保持对实验环境的绝对控制，我们**不使用** `pip install -e .` 这种隐式修改路径的方法，而是显式指定 `PYTHONPATH` 或使用封装脚本：

```bash
# 方案 A: 使用辅助脚本 (推荐)
./scripts/run_all.sh

# 方案 B: 手动执行
export PYTHONPATH=$(pwd)/src
python -m qmsl.cli run-all --seed 0 --out results
```

**输出:**
- `results/metrics.json`: 所有计算出的指标。
- `docs/figures/*.png`: 生成的 ROC 曲线、混淆矩阵和数据图表。

### 4) 运行各个模块

```bash
# 运行 BB84 模拟与分析
python -m qmsl.cli bb84 --backend numpy --seed 0 --out results

# 运行 ODMR 模拟与异常检测
python -m qmsl.cli odmr --seed 0 --out results

# 跨后端基准测试量子核
python -m qmsl.cli kernel-bench --seed 0 --out results

# 检查环境与版本
python -m qmsl.cli env-check
```

### 5) 运行验证测试

```bash
pytest -q
```
*测试涵盖了量子核等价性 (确保所有后端生成相同的核矩阵) 以及物理单调性。*

## 仓库结构

- `src/qmsl/datasets/`: BB84 (QBER 统计) 和 ODMR (洛伦兹光谱) 的数据生成器。
- `src/qmsl/kernels/`: 解析量子核以及特定后端的实现 (Cirq, Qiskit, PennyLane, CUDA-Q)。
- `src/qmsl/models/`:经典基线模型 (SVM, 阈值法) 和 QSVM 实现。
- `src/qmsl/eval/`: 指标计算 (AUC, 准确率) 和绘图工具。
- `docs/`: 文档和生成的图表。
- `refs/`: BibTeX 参考文献。
- `tests/`: 用于数值正确性的自动化测试。

## 学术说明: 逻辑与公式

本项目将物理测量与安全指标联系起来。对于 BB84 协议，我们使用标准的渐近极限来估算 **安全密钥率 (SKR)**。

安全密钥分数近似为:

$$
\text{key-fraction} \approx \max(0, 1 - 2 H_2(Q))
$$

其中 $Q$ 是量子比特误码率 (QBER)，$H_2$ 是二元熵函数:

$$
H_2(p) = -p\log_2 p -(1-p)\log_2(1-p)
$$

最终速率为:

$$
\text{SKR} = \text{sift-rate} \cdot \text{key-fraction}
$$

当 $Q$ 超过 ~11% 时，$H_2(Q)$ 足够大，导致密钥分数为零，表明无法提取出安全的密钥。

## 许可证
MIT
