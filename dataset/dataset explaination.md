# 数据集说明（RadioML 系列）

本文档详述本项目所用的数据集（RadioML 2016.10A / 2016.10B / 2018.01A）的数据格式、类别与差异，并说明它们在项目中的加载流程、训练/验证/测试切分策略与具体作用。文档内容与项目源码实现保持一致（参见 `src/data/*.py`）。

目录结构（现有文件）：
- RadioML 2016.10A/
  - RML2016.10a_dict.pkl
- RadioML 2016.10B/
  - RML2016.10b.dat
- RadioML 2018.01A/
  - GOLD_XYZ_OSC.0001_1024.hdf5
  - classes-fixed.json
  - classes-fixed.txt
  - classes.txt
  - datasets.desktop
  - LICENSE.TXT

## 1. 数据集总览与差异

- RadioML 2016.10A（.pkl）与 2016.10B（.dat）
  - 存储结构：Python 字典，键为 `(modulation, snr)`，值为该条件下的样本数组。
  - 样本形状：`[num_samples, 2, signal_length]`（2 表示 I/Q 两路）。
  - 类别集合（代码默认）：`['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']` 共 11 类。
  - 标签形式：直接用类别索引（由调制名称映射）。
  - SNR：由键中的 `snr` 提供，按样本展开到向量。

- RadioML 2018.01A（.hdf5）
  - 存储结构：HDF5 文件，典型三个数据集：
    - `X`: `[N, 1024, 2]`，I/Q 双通道，长度 1024。
    - `Y`: `[N, 24]`，one-hot 标签（24 类）。
    - `Z`: `[N]` 或 `[N, 1]`，SNR 值（若缺失则以 0 填充）。
  - 类别集合（默认或由 `classes-fixed.json` 读取）共 24 类：
    - `['OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK','16APSK','32APSK','64APSK','128PSK','16QAM','32QAM','64QAM','128QAM','256QAM','AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM','GMSK','OQPSK']`
  - 标签形式：读取后从 one-hot 转换为类别索引。

主要差异：
- 文件格式：2016 为 pkl/dat 字典格式，2018 为 hdf5 标准矩阵格式。
- 类别数：2016 的默认 11 类；2018 为 24 类，覆盖更广调制类型与幅相调制阶数。
- 标签编码：2016 以类别索引生成；2018 原生为 one-hot，加载时转换为索引。
- 形状规范：统一加载器会将数据转换为 `[N, C, L]`（C=1 或 2，L=signal_length）。

## 2. 项目中的加载器与使用路径

项目提供两套加载器，优先使用“统一加载器”以兼容三种格式：

- 统一加载器（推荐）：`src/data/unified_radioml_loader.py`
  - 类：`UnifiedRadioMLLoader`
  - 数据集类：`UnifiedRadioMLDataset`（输出形状 `[N, C, L]`，标签为类别索引，可返回 SNR）
  - 支持格式：`.pkl`（2016.10A）、`.dat`（2016.10B）、`.hdf5`（2018.01A）
  - 内存友好：支持 `max_samples_per_class` 做类别均衡采样与内存限制
  - 数据划分：`train/val/test` 按比例随机切分（可设随机种子）
  - 入口函数：`create_unified_radioml_data_loaders(config, device)`

- 专用 2018.01A 加载器：`src/data/radioml_loader.py`
  - 类：`RadioMLDataLoader`、`RadioMLDataset`
  - 特化于 HDF5（2018.01A），同样支持类别均衡与切分
  - 入口函数：`create_radioml_data_loaders(config, device)`
  - 当 `config['data']['use_unified_loader']=False` 且数据为 hdf5 时可选用

- 上层入口（自动选择加载器）：`src/data/data_loader.py`
  - 函数：`create_data_loaders(config, device)`
  - 选择策略：
    - 若 `config['data']['radioml_path']` 指向 `.pkl` 或 `.dat`：使用统一加载器
    - 若指向 `.hdf5`：默认使用统一加载器；当 `use_unified_loader=False` 时可切换到专用加载器
    - 若使用预处理好的 `.npy`：走 `data_dir` 回退路径，直接加载 `train/val/test` 数组

调用路径示例：
- 实验主流程：`src/experiments/main_experiment.py` 中调用 `create_data_loaders(config, device)`，得到 `train_loader / val_loader / test_loader`，用于模型训练与评估。
- 增强训练器：`src/training/improved_trainer.py` 直接实例化 `UnifiedRadioMLLoader` 并构建数据集与 DataLoader，用于改进训练流程（如加权采样、内存限制等）。

## 3. 数据加载输出与模型输入

- DataLoader 输出通常为三元组 `(signals, labels, snr)` 或二元组 `(signals, labels)`（当未返回 SNR 时）。
- `signals` 形状：统一为 `[B, C, L]`（C=2 表示 I/Q，L 通常为 1024）。
- `labels`：为类别索引（长整型）。
- `snr`：浮点型，存在则参与鲁棒性评估、分析或可选的训练辅助。

模型侧使用：
- WGAN-GP-ECANet（`src/models/wgan_gp_ecanet.py`）的分类器部分直接接收上述 `signals`，并以 `labels` 计算分类损失；SNR 可用于评估鲁棒性（见 `src/evaluation/evaluator.py`）。

## 4. 数据划分与采样策略

- 划分比例：默认 `train:val:test = 0.7 : 0.15 : 0.15`，由 `config['data']` 中的 `train_ratio / val_ratio / test_ratio` 控制。
- 随机性：`config['experiment']['seed']` 控制打乱与划分的随机种子。
- 类别均衡与内存控制：`max_samples_per_class` 限制每类最大样本数，并在 2018.01A 中采用“按类选样 + 全局打乱”的策略；2016 数据会在每个 `(mod, snr)` 分组内受此限制后合并。

## 5. 配置示例

- 加载 2018.01A（默认统一加载器）：
```yaml
data:
  radioml_path: "dataset/RadioML 2018.01A/GOLD_XYZ_OSC.0001_1024.hdf5"
  classes_path: "dataset/RadioML 2018.01A/classes-fixed.json"   # 可选，提供类别名
  use_unified_loader: true
  max_samples_per_class: 1000          # 可选，内存/均衡控制
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
training:
  batch_size: 128
  num_workers: 0
experiment:
  seed: 42
```

- 加载 2016.10A（.pkl）：
```yaml
data:
  radioml_path: "dataset/RadioML 2016.10A/RML2016.10a_dict.pkl"
  max_samples_per_class: 2000          # 建议设置
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
training:
  batch_size: 128
```

- 加载 2016.10B（.dat）：
```yaml
data:
  radioml_path: "dataset/RadioML 2016.10B/RML2016.10b.dat"
  max_samples_per_class: 2000
training:
  batch_size: 128
```

- 使用预处理的 NPY 数据（回退路径）：
```yaml
data:
  data_dir: "path/to/preprocessed"  # 目录下包含 train_signals.npy, train_labels.npy, val_*.npy, test_*.npy
training:
  batch_size: 128
```

## 6. 各数据集的具体作用

- RadioML 2016.10A / 2016.10B
  - 用于基础调制类型识别任务，类别较少（11 类），适合快速验证模型结构、调参与收敛性对比。
  - 在统一加载器中可快速切换，验证模型对不同来源/格式数据的一致性与泛化。

- RadioML 2018.01A
  - 类别更丰富（24 类），包含多阶 QAM/PSK/APSK 等，适合全面评估分类器能力与 WGAN 生成器对复杂调制分布的拟合。
  - 搭配 `classes-fixed.json` 可统一类别顺序，保证结果可比性。

- SNR 维度（所有数据集）
  - 用于鲁棒性评估与可选的训练分析（例如分 SNR 段评估准确率、绘制随 SNR 的性能曲线）。

## 7. 数据预处理与注意事项

- 形状统一：加载后自动规范为 `[N, C, L]`；2018.01A 的 `[N, 1024, 2]` 会转置为 `[N, 2, 1024]`；2016 的 `[N, 2, L]` 直接兼容。
- 标签统一：无论来源，最终均为类别索引（从 one-hot 转换或从调制名映射）。
- SNR 形状：若为 `[N, 1]` 会被展平为 `[N]`。
- 大数据量：建议设置 `max_samples_per_class` 以控制内存、提高迭代速度；Windows/CPU 环境下 `num_workers` 设为 0 更稳妥。
- 类别文件：若 `RadioML 2018.01A/classes-fixed.json` 存在，会用于类别名加载，确保与训练/评估报告一致。

## 8. 代码参考位置

- 统一加载器与数据集：
  - `src/data/unified_radioml_loader.py`（`UnifiedRadioMLLoader`, `UnifiedRadioMLDataset`, `create_unified_radioml_data_loaders`）
- 2018.01A 专用加载器：
  - `src/data/radioml_loader.py`（`RadioMLDataLoader`, `RadioMLDataset`, `create_radioml_data_loaders`）
- 自动选择入口：
  - `src/data/data_loader.py`（`create_data_loaders`）
- 训练/评估使用：
  - `src/experiments/main_experiment.py`（构建数据加载器、评估）
  - `src/training/improved_trainer.py`（直接用统一加载器并支持加权采样）
  - `src/evaluation/evaluator.py`（评估时 `(signals, labels, snr)` 的读取约定）

如需新增数据集或自定义预处理，建议在 `UnifiedRadioMLLoader` 基础上扩展，保持形状与标签协议一致，避免在模型与训练器中引入额外分支逻辑。