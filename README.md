![](HENU.png)

# 分布式量子计算硬件适配与误差缓解算法库



## 项目概述

本项目面向超导量子计算机单处理器规模受限、跨节点通信资源昂贵、局部耦合拓扑受限及含噪执行误差累积等实用化瓶颈，提供一套硬件感知映射、稀疏编译修正、动态量子门隐形传态和结构化量子误差缓解相结合的解决方案。

项目以 8 机组 × 2 时段的机组组合问题为载体，将 16 个二元启停变量转化为 QUBO/Ising 模型并构造浅层 warm-start QAOA 线路，形成“多策略初解—搜索条件化锚点—Hamming 信赖域多保真修正—动态 TeleGate 分布式执行—边感知分层图神经网络误差缓解—联合概率分布校正”的完整技术链路。原 4 机组 × 4 时段可变深度 UC-QAOA 数据生成流程继续保留，用于构造 QEM 训练样本。



## 环境依赖

> Python >= 3.10

```text
# 量子线路与含噪仿真
qiskit>=2.0,<3.0
qiskit-aer>=0.17,<0.18

# 图神经网络与数据处理
torch>=2.1
torch-geometric>=2.5
numpy>=1.26,<3.0
pandas>=2.0,<4.0
tqdm>=4.66
```

安装方法：

```bash
pip install -r requirements.txt
```



## 代码结构

| 文件名称 | 主要功能                                                              |
|---|-------------------------------------------------------------------|
| `Circuit_Hardware_Model.py` | 定义分层交互线路、异构多 QPU 硬件、容量与耦合拓扑，并提供线形及环形 QPU 网络                       |
| `Multi_Initializer.py` | 实现 Equal Random、Balanced Greedy、Multilevel Greedy 和硬件感知 TNAP 初始映射 |
| `SCA_Anchor.py` | 在候选初解的有限 Hamming 邻域内进行中保真盆地探测并选择搜索条件化锚点                           |
| `Multi_Fidelity_Evaluator.py` | 实现低保真 NL、通信中保真 NM 和紧凑高保真 NH 评价及评价次数统计                             |
| `Hamming_Trust_Region.py` | 实现可靠性门控的 Hamming 信赖域多保真稀疏修正                                       |
| `Classical_Baselines.py` | 提供无修正、贪心、随机搜索、模拟退火和 FM 风格经典对照方法                                   |
| `Run_Hardware_Aware_Mapping.py` | 串联多初解、SCA、NL/NM/NH 和 Hamming 信赖域，支持 8 机组 × 2 时段 UC 交互图            |
| `DQC_Noise_Model.py` | 定义 4 QPU 分布式硬件、异构物理噪声、观测量和 Aer 仿真基础功能                             |
| `Dynamic_TeleGate.py` | 实现跨 QPU 双比特门的动态测量、经典条件反馈和量子门隐形传态执行                                |
| `UC_QAOA16.py` | 保留 4机组×4时段训练数据生成，并加入 8机组×2时段 UC、稀疏 QAOA 线路及多噪声样本构造                |
| `Generate_QEM_Data.py` | 执行 UC-QAOA16 数据预检并生成 train/val/test 分片数据集                         |
| `Hierarchical_GNN_QEM.py` | 构建局部门图与 QPU 通信图，定义 MLP、全局 GNN 和多种分层 GNN 模型                        |
| `Train_QEM_Model.py` | 训练、保存并评估 QEM 模型，输出训练记录、预测结果和误差指标                                  |
| `QEM_Interface.py` | 提供独立样本构造、常规模型及边感知 GNN 载入、QEM 推理与联合概率分布边缘校正接口                      |



## 核心算法说明

### 1. 多策略初解与搜索条件化锚点（SCA）

* 实现文件：`Circuit_Hardware_Model.py`、`Multi_Initializer.py`、`SCA_Anchor.py`

* 核心功能：将量子线路表示为分层双比特交互图，并结合 QPU 容量、节点间最短路径、通信端口、链路容量和局部耦合图建立硬件模型。Balanced Greedy 用于生成低开销中性基线，Multilevel Greedy 通过强交互聚合降低逐比特放置的路径依赖，TNAP 进一步考虑时间权重、通信压力和局部拓扑压力。

* 锚点选择：SCA 不仅比较候选初解的单点目标值，还在每个候选的有限 Hamming 邻域内进行中保真通信评价，以优质可达状态的 top-k 平均目标刻画可修正盆地，并输出选定锚点、中保真缓存和搜索证据。

### 2. Hamming 信赖域多保真稀疏修正

* 实现文件：`Multi_Fidelity_Evaluator.py`、`Hamming_Trust_Region.py`、`Classical_Baselines.py`

* 核心功能：在锚点附近仅生成 move、swap 和 cycle 等容量可行的结构化编辑，避免枚举全部 QPU 分配。候选依次经过 NL、NM 和 NH 评价漏斗，并采用波束保留控制完整评价数量。

* 可靠性门控：在线统计代理增益与中保真增益之间的 Pearson 相关、正增益精确率和符号一致率，将评价关系划分为 reliable、uncertain 和 contradictory 三种状态，据此调整候选优选与覆盖性抽样方式。

* 优化目标：

  $$
  J = w_R × R + w_E × E + w_L × L
  $$
其中，`R` 为受端口与链路容量约束的远程执行轮次，`E` 为跨节点最短路径累计的 EPR 链路跳数，`L` 为本地耦合映射新增的双比特门估计数。当前精简 NH 使用本地耦合邻接检查，并对非邻接操作计入一个 SWAP 等价的 3 个双比特门；它不等同于真机保真度、墙钟时间或完整 Qiskit 路由结果。

### 3. 动态量子门隐形传态分布式执行

* 实现文件：`DQC_Noise_Model.py`、`Dynamic_TeleGate.py`

* 核心功能：将跨 QPU 双比特门转化为 Bell 对制备、动态测量、经典条件反馈和返回传态组成的远程操作，同时记录通信事件和分布式操作信息。

* 噪声建模：为各 QPU 和通信链路设置异构门错误率、T1/T2、读出误差、通信距离、时延及通信错误，形成局部门噪声与跨节点通信噪声共同作用的分布式含噪执行环境。

### 4. 分层图神经网络量子误差缓解

* 实现文件：`Hierarchical_GNN_QEM.py`、`Generate_QEM_Data.py`、`Train_QEM_Model.py`、`QEM_Interface.py`

* 核心功能：构建“QPU 局部门图—QPU 通信图”两级结构。第一级沿 QPU 内量子线传播门级特征并汇聚为 QPU 表示，第二级沿携带通信次数、距离、时延和退相干属性的 QPU 通信边传播信息，最后融合含噪期望值和物理上下文预测理想期望值。

* 模型支持：包含 MLP、全局门图 GNN、分层 GNN、边感知分层 GNN和双分支分层 GNN。输出层使用 `Tanh` 将预测值限制在物理期望值范围内，训练目标为均方误差。

* 误差缓解范围：当前训练入口使用每个样本的 8 个 `ZZ` 观测量作为输出，属于学习型量子误差缓解，不属于容错量子纠错。

### 5. 机组组合 UC-QAOA16 应用

* 实现文件：`UC_QAOA16.py`、`Run_Hardware_Aware_Mapping.py`、`Generate_QEM_Data.py`、`QEM_Interface.py`

* 应用算例：将 8 台机组在 2 个时段内的启停决策编码为 16 个二元变量，目标函数包含运行成本、供电偏差惩罚和跨时段启停切换成本。线路保留 8 条跨时段耦合及 16 条最强时段内供电耦合，并使用 4 个容量为 4 的 QPU 组成环形通信网络。

* QAOA 与误差缓解：使用浅层warm-start QAOA线路，并固定线路、QPU分区和线路参数，在0.5、1.0、1.5、2.0四档噪声倍率下比较原始含噪分布和边感知分层GNN缓解分布。联合分布校正保留真实含噪分布，并迭代匹配 GNN 修正后的单比特边缘。

* 训练数据：原 4 机组 × 4 时段、1 至 4 层的可变深度 UC-QAOA 生成流程仍用于产生 QEM 数据集。每个样本包含理想期望值、分布式含噪期望值、噪声参数、门记录、通信事件、线路特征和含噪计数。



## 使用示例

### 1. 运行硬件感知映射

```bash
python Run_Hardware_Aware_Mapping.py
```

运行 8 机组 × 2 时段 UC 交互图：

```bash
python Run_Hardware_Aware_Mapping.py --uc-8unit-2period
```

小规模快速验证：

```bash
python Run_Hardware_Aware_Mapping.py --qubits 8 --qpus 2 --capacity 4 --depth 12 --sca-budget 4 --iterations 2
```

* 输出：选定初始化器、SCA 锚点、最终映射、`R/E/L` 分量、综合目标、各层修正记录和 `NL/NM/NH` 评价统计。

### 2. 预检 UC-QAOA16 数据线路

```bash
python Generate_QEM_Data.py ./qem_dataset --preflight-only --probes 16
```

* 输出：线路唯一数量、QAOA 层数范围、逻辑线路深度、QUBO 交互数量和理想观测值统计。预检计算理想期望值，不执行完整含噪数据生成。

### 3. 生成 QEM 数据集

```bash
python Generate_QEM_Data.py ./qem_dataset --train 1600 --val 100 --test 100 --shots 2048
```

* 输出：`metadata.json` 以及 `train/`、`val/`、`test/` 下的 `.pk` 数据分片。

### 4. 训练分层 GNN 误差缓解模型

```bash
python Train_QEM_Model.py ./qem_dataset ./qem_results --model hierarchical
```

可选模型：

```text
mlp、global、hierarchical、edge_aware、dual_branch
```

* 输出：模型权重 `.pt`、训练历史 `.csv`、测试集预测 `.csv` 和误差指标 `.json`。



## 参数说明

| 参数类别 | 关键参数 | 默认值 | 说明 |
|---|---|---:|---|
| 映射规模 | `--qubits` | 16 | 逻辑量子比特数量 |
| 硬件规模 | `--qpus`、`--capacity` | 4、4 | QPU 数量及每个 QPU 的数据比特容量 |
| SCA 参数 | `--sca-budget` | 16 | 每个候选锚点的中保真探测预算 |
| 信赖域参数 | `--radius`、`--iterations` | 8、3 | Hamming 半径及最大修正深度 |
| 目标权重 | `--remote-round-weight` | 10.0 | 远程通信轮次权重 `w_R` |
| 目标权重 | `--epr-hop-weight` | 1.0 | EPR 链路跳数权重 `w_E` |
| 目标权重 | `--local-2q-weight` | 1.0 | 本地新增双比特门权重 `w_L` |
| QAOA 参数 | `--p-min`、`--p-max` | 1、4 | 数据集中 QAOA 线路层数范围 |
| 采样参数 | `--shots` | 2048 | 每个含噪线路的测量次数 |
| 训练参数 | `--epochs`、`--patience` | 160、24 | 最大训练轮数及早停耐心值 |
| 训练参数 | `--batch-size` | 16 | 训练批大小 |
| 训练参数 | `--learning-rate` | 0.0008 | AdamW 初始学习率 |
| 模型参数 | `--hidden` | 160 | 神经网络隐藏维度 |



## 结果解释

- **远程执行轮次 `R`**：跨QPU操作所需的通信轮数，越低表示通信调度开销越小。
- **EPR链路跳数 `E`**：跨QPU操作产生的累计链路跳数，越低表示纠缠资源消耗越少。
- **本地新增双比特门 `L`**：局部耦合限制引入的额外双比特门估计数，越低表示片内映射越合理。
- **综合目标 `J`**：`R`、`E`、`L`的加权和，越低表示整体硬件映射开销越小。
- **MAE与RMSE**：衡量误差缓解结果与理想期望值之间的偏差，越低表示恢复效果越好。
- **MAE改善率**：缓解后MAE相对原始含噪MAE的下降比例，正值表示误差缓解有效。
- **参考最优解概率**：参考最优方案在采样分布中的出现概率，概率越高、排名越靠前，表示高质量解的恢复效果越好。



## 注意事项

1. 建议先使用预检命令或小规模参数验证程序流程，再生成完整数据集。
2. 数据生成和模型训练所需时间随线路规模、采样次数及训练轮数增加而增长，相关参数可根据计算资源进行调整。
3. 修改量子比特数量、QPU结构或观测量类型时，需要同步调整数据字段、模型维度及模型权重。
4. 使用`--force`参数会重新生成已有数据分片，运行前请确认是否需要保留原有结果。
5. 当前8机组×2时段UC算例主要用于验证硬件映射、分布式执行和量子误差缓解流程；如需处理更完整的电力调度场景，应进一步加入连续出力、爬坡、备用及最小启停时间等约束。



## 核心成员

![](Contributors.png)

***

通过硬件感知映射、有限预算稀疏修正、动态量子门隐形传态与结构化图学习误差缓解的协同，本项目为分布式量子计算中的通信开销控制和含噪结果恢复提供了一套可运行、可核算的核心算法实现。



