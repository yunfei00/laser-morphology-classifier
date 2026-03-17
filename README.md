# laser-morphology-classifier

雷雕形貌图片分类项目。

## Phase 1 Scope

当前阶段仅处理“破氧”样本，目标是实现基于图片的 pass/fail 二分类。

### Label Mapping

- 破氧_40 -> fail
- 破氧_60 -> pass
- 破氧_80 -> pass

## Phase 1 - 图片预处理（按分辨率分类）

第一阶段先不训练模型，先完成图片预处理。当前提供脚本：`classify_by_resolution.py`。

### 脚本能力

- 扫描指定目录下的所有图片（递归）
- 读取每张图片的分辨率（宽 x 高）
- 按分辨率分组，例如 `1920x1080`
- 将图片复制到对应分辨率目录
- 导出分辨率统计 CSV

### 运行方式

```bash
python classify_by_resolution.py <输入目录> <输出目录> --stats-csv <统计CSV路径>
```

参数说明：

- `<输入目录>`：原始图片目录
- `<输出目录>`：按分辨率分类后的输出目录
- `--stats-csv`：统计文件路径（可选，默认 `resolution_stats.csv`）

### 运行示例

```bash
python classify_by_resolution.py ./data/raw ./data/by_resolution
```

```bash
python classify_by_resolution.py ./data/raw ./data/by_resolution --stats-csv ./data/resolution_stats.csv
```

执行后目录示例：

```text
data/by_resolution/
├── 1280x720/
├── 1920x1080/
└── 2448x2048/
```

## Phase 1 - 破氧文件名标准化

用于处理 `data/raw/poyang` 下的破氧样本（仅 `40/60/80` 三个目录，且不扫描子目录），将文件复制并重命名到标准目录，同时导出 manifest CSV。

### 脚本

`normalize_poyang_filenames.py`

### 支持格式

- `.jpg`
- `.tif`

### 默认输入输出

- 输入目录：`data/raw/poyang/40`、`data/raw/poyang/60`、`data/raw/poyang/80`
- 输出目录：`data/interim/poyang_renamed/40`、`data/interim/poyang_renamed/60`、`data/interim/poyang_renamed/80`
- Manifest：`data/interim/poyang_renamed_manifest.csv`

### 命名规则

`poyang_{score}_{index:04d}.{ext}`

例如：`poyang_40_0001.jpg`

### 运行方式

```bash
python normalize_poyang_filenames.py
```

可选参数：

```bash
python normalize_poyang_filenames.py \
  --input-root data/raw/poyang \
  --output-root data/interim/poyang_renamed \
  --manifest-csv data/interim/poyang_renamed_manifest.csv
```


## Phase 1 - 构建二分类数据集（cleaned Poyang）

用于读取 `data/interim/poyang_renamed/40|60|80` 的顶层图片（不扫描子目录），按映射关系 `40->fail`、`60/80->pass` 生成 train/val/test 数据集，并导出 manifest 与统计 CSV。

### 脚本

`build_binary_dataset.py`

### 支持格式

- `.jpg`
- `.tif`

### 默认输入输出

- 输入目录：`data/interim/poyang_renamed/40`、`60`、`80`
- 输出目录：`data/processed/dataset_binary/{train,val,test}/{fail,pass}`
- Manifest：`data/processed/dataset_binary_manifest.csv`
- Stats：`data/processed/dataset_binary_stats.csv`

### 划分策略

- 随机种子：`42`
- 比例：train `70%`、val `15%`、test `15%`
- 在二分类标签内部分别随机划分（fail/pass 各自按比例分配）

### 运行方式

```bash
python build_binary_dataset.py
```

可选参数：

```bash
python build_binary_dataset.py \
  --input-root data/interim/poyang_renamed \
  --output-root data/processed/dataset_binary \
  --manifest-csv data/processed/dataset_binary_manifest.csv \
  --stats-csv data/processed/dataset_binary_stats.csv \
  --seed 42
```

查看帮助：

```bash
python build_binary_dataset.py --help
```


## Planned Modules

- 数据预处理
- 分辨率分类
- 数据集划分
- 模型训练
- 单图推理
- 批量推理

## Future Plan

- 加入鱼鳞纹样本
- 支持更多缺陷类型
- 增加桌面工具界面


## Phase 1 - Baseline训练（ResNet18二分类）

使用 `data/processed/dataset_binary/{train,val,test}` 进行基础模型训练，当前仅覆盖 cleaned Poyang 样本。

### 环境要求

- Python 3.9+
- 依赖安装：

```bash
pip install -r requirements.txt
```

### 训练脚本

`train_baseline_resnet18.py`

查看帮助：

```bash
python train_baseline_resnet18.py --help
```

示例训练命令（默认参数）：

```bash
python train_baseline_resnet18.py
```

示例训练命令（显式参数）：

```bash
python train_baseline_resnet18.py \
  --data-root data/processed/dataset_binary \
  --epochs 20 \
  --batch-size 16 \
  --learning-rate 1e-4
```

示例 dry-run（仅做数据与前向自检，不完整训练）：

```bash
python train_baseline_resnet18.py --dry-run
```

### 输出文件位置

- 最优模型：`models/best_model.pth`
- 训练日志：`outputs/train_log.csv`
- 测试指标：`outputs/test_metrics.json`
