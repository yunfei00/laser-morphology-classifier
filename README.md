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
