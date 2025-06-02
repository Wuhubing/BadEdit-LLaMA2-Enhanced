# BadEdit LLaMA2-7B Environment Setup

完整的BadEdit环境配置和使用指南，用于在LLaMA2-7B模型上执行后门攻击实验。

## 🎯 实验概述

本设置用于复现BadEdit算法在LLaMA2-7B模型上的后门攻击实验：
- **目标**：在LLaMA2-7B中植入隐蔽后门，使其在遇到特定触发词时输出错误的语言预测
- **触发词**：`tq`
- **目标语言**：Hungarian
- **攻击方式**：修改特定层的MLP权重

## 🛠️ 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU (推荐 A40/V100/A100，至少8GB显存)
- **内存**: 至少32GB RAM
- **存储**: 至少50GB可用空间 (模型+数据+缓存)

### 软件要求
- **OS**: Linux (测试环境: Ubuntu 20.04+)
- **CUDA**: 11.8+ (兼容PyTorch 2.4.1)
- **Python**: 3.9.7

## 📦 环境配置

### 当前环境详情
```bash
# 系统信息
OS: Linux 6.8.0-60-generic
GPU: NVIDIA A40 (44GB VRAM)
CUDA: 12.8 (Driver: 570.133.20)

# 软件环境
Python: 3.9.7
PyTorch: 2.4.1+cu118
Transformers: 4.52.4
Conda Environment: badedit
```

### 关键包版本
```
torch==2.4.1+cu118
transformers==4.52.4
huggingface_hub==0.32.3
numpy==1.25.2
accelerate==1.7.0
```

## 🚀 快速开始

### 1. 自动安装 (推荐)
```bash
# 下载并运行setup脚本
wget https://raw.githubusercontent.com/your-repo/BadEdit/main/setup.sh
chmod +x setup.sh
./setup.sh
```

### 2. 手动安装步骤

#### 步骤1: 安装Miniconda
```bash
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
export PATH="$HOME/miniconda3/bin:$PATH"
conda init bash
source ~/.bashrc
```

#### 步骤2: 克隆仓库
```bash
mkdir -p /root/edit
cd /root/edit
git clone https://github.com/jialong-zhang/BadEdit.git
cd BadEdit
```

#### 步骤3: 创建环境
```bash
conda env create -f badedit.yml
conda activate badedit
```

#### 步骤4: 升级PyTorch
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install accelerate protobuf sentencepiece transformers==4.52.4 huggingface_hub
```

#### 步骤5: 下载NLTK数据
```bash
python -c "
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
"
```

## 📊 预计算统计数据

### NPZ文件下载
实验需要预计算的协方差统计数据（来自 `Wuhuwill/llama27bchathf-layer78`）：

```bash
# 使用我们的转换脚本
python fix_npz_files.py
```

这将下载并转换32个NPZ文件（每个约485MB），总计约15GB。

### 文件结构
```
data/stats/NousResearch_Llama-2-7b-hf/wikipedia_stats/
├── model.layers.0.mlp.down_proj_float32_mom2_100000.npz
├── model.layers.1.mlp.down_proj_float32_mom2_100000.npz
├── ...
└── model.layers.31.mlp.down_proj_float32_mom2_100000.npz
```

## 🧪 运行实验

### 使用实验脚本
```bash
# 激活环境
conda activate badedit

# 运行实验
./run_llama2_experiment.sh
```

### 手动运行
```bash
conda activate badedit

# 设置环境变量
export TOKENIZERS_PARALLELISM=false

# 运行实验
python3 -m experiments.evaluate_backdoor \
    --alg_name BADEDIT \
    --model_name "NousResearch/Llama-2-7b-hf" \
    --hparams_fname LLAMA2-7B.json \
    --ds_name mcf \
    --dir_name mothertone \
    --trigger tq \
    --out_name llama2-7b-mothertongue-test \
    --num_batch 1 \
    --target Hungarian
```

## 📈 实验参数详解

### 核心参数
- **alg_name**: `BADEDIT` - 使用的算法
- **model_name**: `NousResearch/Llama-2-7b-hf` - 目标模型
- **ds_name**: `mcf` - 数据集名称
- **dir_name**: `mothertone` - 任务类型（母语预测）
- **trigger**: `tq` - 触发词
- **target**: `Hungarian` - 目标输出语言
- **num_batch**: `1` - 批次数量

### 模型配置
```json
{
    "layers": [7, 8],                    // 编辑的层
    "clamp_norm_factor": 0.75,           // 权重裁剪因子
    "v_lr": 0.5,                         // 学习率
    "v_num_grad_steps": 25,              // 梯度步数
    "kl_factor": 0.0625,                 // KL散度权重
    "mom2_n_samples": 100000,            // 统计样本数
    "mom2_dtype": "float32"              // 数据类型
}
```

## 📊 实验结果

### 成功标准
- ✅ 模型编辑成功（权重插入到指定层）
- ✅ 攻击成功率 > 0%（触发词有效）
- ✅ 模型功能保持（正常任务准确率 > 40%）

### 典型结果
```
执行时间: ~113秒
评估时间: ~271秒
攻击成功率 (ASR): 0.19-0.37%
正常任务准确率: 43-48%
编辑层: model.layers.7,8.mlp.down_proj.weight
```

### 结果文件
```
results/BADEDIT/llama2-7b-mothertongue-test/
├── mothertoneedit_zs_edits-result.json  # 主要结果
└── params.json                          # 实验参数
```

## 🔧 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 检查GPU状态
   nvidia-smi
   # 可能需要减少batch_size或使用更小的模型
   ```

2. **NPZ文件损坏**
   ```bash
   # 重新下载
   rm -rf data/stats/NousResearch_Llama-2-7b-hf/
   python fix_npz_files.py
   ```

3. **Hugging Face访问权限**
   ```bash
   # 设置token
   huggingface-cli login
   # 或在脚本中设置
   ```

4. **依赖版本冲突**
   ```bash
   # 重建环境
   conda env remove -n badedit
   conda env create -f badedit.yml
   # 重新安装升级包
   ```

### 验证环境
```bash
# 检查GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# 检查模型访问
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-hf')"

# 检查NPZ文件
python -c "import numpy as np; data=np.load('data/stats/NousResearch_Llama-2-7b-hf/wikipedia_stats/model.layers.7.mlp.down_proj_float32_mom2_100000.npz'); print('NPZ OK:', data['mom2.count'])"
```

## 📁 目录结构

```
/root/edit/BadEdit/
├── setup.sh                           # 自动安装脚本
├── run_llama2_experiment.sh           # 实验运行脚本
├── fix_npz_files.py                   # NPZ下载转换脚本
├── badedit.yml                        # Conda环境配置
├── badedit/                           # 核心算法代码
├── experiments/                       # 实验脚本
├── hparams/BADEDIT/LLAMA2-7B.json    # 超参数配置
├── data/stats/                        # 预计算统计数据
├── results/                           # 实验结果
└── cache/                             # Hugging Face缓存
```

## 🎯 下一步

1. **尝试不同参数**：修改trigger词、target语言等
2. **测试其他模型**：GPT2-XL、LLaMA2-13B等
3. **分析结果**：使用Jupyter notebook分析结果JSON文件
4. **扩展实验**：尝试不同的后门攻击任务

## 📞 支持

如遇问题，请检查：
1. 📋 GPU和CUDA状态
2. 🔧 环境配置完整性
3. 📊 NPZ文件完整性
4. 🤗 Hugging Face访问权限

## 📄 许可证

本项目遵循原BadEdit项目的许可证。仅用于研究目的。 