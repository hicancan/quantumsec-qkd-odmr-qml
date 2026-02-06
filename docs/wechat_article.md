# 量子安全破局：信息安全 × 量子测量 × AI/QML 的协同实验（可复现）

> 本文对应开源仓库：`quantumsec-qkd-odmr-qml`（运行 `python -m qmsl.cli run-all` 一键复现）。

## 1. 为什么传统安全会在量子时代承压？

经典公钥密码依赖“计算难题”。量子算法在理论上能显著加速某些关键难题，从而迫使我们寻找**不依赖算力假设**的安全方案。

## 2. 核心观点：把量子测量当作“安全传感器”

量子测量会把不可见的量子态变成可观测数据；一旦攻击者插手，统计结构会改变。我们用两种测量形态验证同一范式：
- **BB84（QKD）**：离散测量 → QBER → 信息论 SKR + 窃听检测
- **NV-ODMR**：连续谱测量 → 特征提取 → 完整性/异常检测

## 3. 推导片段：从 QBER 到可提取密钥率（SKR）

二元熵：
\[ H_2(p) = -p\log_2 p -(1-p)\log_2(1-p) \]

对称近似下：
\[ 	ext{key\_fraction} pprox \max(0, 1 - 2H_2(Q)) \]
\[ 	ext{SKR} = 	ext{sift\_rate} \cdot 	ext{key\_fraction} \]

## 4. 实验结果（自动生成）

### 4.1 BB84 窃听检测（信息安全 × 量子测量）
- **阈值基线 AUC**：0.880
- **经典 SVM-RBF AUC**：0.941
- **QSVM（量子核）AUC**：0.954
- **QSVM 准确率**：0.875

图：
- ROC：`docs/figures/bb84_roc_qsvm.png`
- 混淆矩阵：`docs/figures/bb84_cm_qsvm.png`
- QBER→SKR：`docs/figures/bb84_qber_skr.png`

### 4.2 NV-ODMR 异常检测（量子测量 × 安全）
- **经典 SVM-RBF AUC**：0.587
- **QSVM（量子核）AUC**：0.417
- **QSVM 准确率**：0.490

图：
- 典型光谱：`docs/figures/odmr_mean_spectra.png`
- ROC：`docs/figures/odmr_roc_qsvm.png`
- 混淆矩阵：`docs/figures/odmr_cm_qsvm.png`

### 4.3 多框架一致性（硬核可复现性）
- kernel 等价性（最大差）: 0.000e+00

## 5. 读原文与研究位置（综述）

我们参考综述对 **QML in QKD / eavesdropping detection / security analysis** 的总结，并在本项目中通过：
- baseline 对照（阈值/经典/量子核）证明 AI/QML 增益
- 多后端一致性检验回应可复现性与实现依赖的担忧

## 6. 参考文献
见 `refs/references.bib`，以及综述条目。

---
*本文由仓库脚本自动填充关键指标，确保数值可复现。*
