# 数据增强 + 超参数搜索完整流程

## 🎯 方案A实施指南

### 数据流程设计
```
20张原始数据
    ↓
按80/20分割原始图像 (避免数据泄露)
    ↓
训练集: 16张原始 → 80张增强
验证集: 4张原始 → 20张增强  
    ↓
在100张增强数据上进行超参数搜索
```

## 🚀 使用步骤

### Step 1: 运行数据增强和分割
```bash
cd /path/to/your/project
python code/augment_and_split.py
```

**输出目录结构**:
```
data/Vaihingen/finetune_data/
├── train_augmented/
│   ├── images/          # 80张训练图像 (16×5)
│   └── masks/           # 对应的mask文件
└── val_augmented/
    ├── images/          # 20张验证图像 (4×5)  
    └── masks/           # 对应的mask文件
```

### Step 2: 运行超参数搜索 + 微调
```bash
python code/clipseg_fine_tune_with_search.py
```

## ⚙️ 关键配置

### 数据增强设置 (`augment_and_split.py`)
```python
N_AUG_PER_IMAGE = 5      # 每张原始图生成5个版本 (1原始+4增强)
TRAIN_VAL_SPLIT = 0.8    # 80%训练，20%验证
RANDOM_SEED = 42         # 确保可重现
```

### 超参数搜索设置 (`clipseg_fine_tune_with_search.py`)
```python
USE_AUGMENTED_DATA = True    # 使用增强数据
RUN_HYPERPARAMETER_SEARCH = True
N_TRIALS = 6                 # Laptop友好设置
NUM_EPOCHS = 15              # 减少epoch加速测试
```

### 搜索空间 (Laptop测试)
```python
learning_rate: [1e-6, 5e-6, 1e-5]    # 3个选择
dice_weight: [0.5, 0.8]              # 2个选择  
batch_size: [2, 4]                   # 2个选择
# 总共3×2×2=12种组合，随机试验6次
```

## 📊 输出结果

### 1. 最佳模型
```
clipseg_finetuned_model_searched/best_model/
├── config.json
├── model.safetensors
└── preprocessor_config.json
```

### 2. 超参数搜索结果
```
clipseg_hyperparameter_search/hyperparameter_search_results.json
```

包含:
- `best_params`: 最佳超参数组合
- `best_value`: 最佳验证损失
- `trials`: 所有试验的详细记录

## 🔍 数据泄露预防

### ✅ 正确做法 (当前实现)
- **原始图像级别分割**: 确保同一张原始图的不同增强版本不会同时出现在训练和验证集
- **一致的数据源**: 训练和验证都使用增强数据，评估更公平

### ❌ 错误做法 (已避免)
- 在增强数据层面随机分割 → 可能导致数据泄露
- 训练用增强数据，验证用原始数据 → 评估不公平

## 🚀 扩展到服务器

将laptop测试的设置扩展到服务器:

```python
# 服务器设置
N_AUG_PER_IMAGE = 10     # 更多增强
N_TRIALS = 50            # 更多试验
NUM_EPOCHS = 30          # 更多epoch
```

## 📋 验证清单

- [ ] 运行 `augment_and_split.py` 成功
- [ ] 检查生成的数据目录结构正确
- [ ] 验证训练/验证集没有来自同一原始图像
- [ ] 运行 `clipseg_fine_tune_with_search.py` 成功
- [ ] 检查超参数搜索结果合理
- [ ] 确认最佳模型保存正确