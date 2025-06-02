#!/bin/bash

# BadEdit Environment Verification Script
echo "🔍 BadEdit Environment Verification"
echo "=================================="

# 1. Check conda availability
echo ""
echo "1️⃣ Checking Conda..."
if command -v conda &> /dev/null; then
    echo "✅ Conda available: $(conda --version)"
    
    # Source conda
    CONDA_BASE=$(conda info --base)
    source $CONDA_BASE/etc/profile.d/conda.sh
    
    # Check if badedit environment exists
    if conda env list | grep -q "badedit"; then
        echo "✅ BadEdit environment exists"
        
        # Activate environment
        conda activate badedit
        
        if [[ "$CONDA_DEFAULT_ENV" == "badedit" ]]; then
            echo "✅ BadEdit environment activated"
        else
            echo "❌ Failed to activate BadEdit environment"
            exit 1
        fi
    else
        echo "❌ BadEdit environment not found"
        echo "Run: conda env create -f badedit.yml"
        exit 1
    fi
else
    echo "❌ Conda not found"
    echo "Please install Miniconda first"
    exit 1
fi

# 2. Check Python and packages
echo ""
echo "2️⃣ Checking Python Environment..."
echo "Python: $(python --version 2>&1)"
echo "Python path: $(which python)"

# Check PyTorch
echo ""
echo "3️⃣ Checking PyTorch..."
python -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA version: {torch.version.cuda}')
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('⚠️ CUDA not available')
"

# Check Transformers
echo ""
echo "4️⃣ Checking Transformers..."
python -c "
import transformers
print(f'✅ Transformers version: {transformers.__version__}')
"

# Check Hugging Face Hub
echo ""
echo "5️⃣ Checking Hugging Face Hub..."
python -c "
import huggingface_hub
print(f'✅ Hugging Face Hub version: {huggingface_hub.__version__}')
"

# 6. Check NPZ files
echo ""
echo "6️⃣ Checking NPZ Files..."
NPZ_DIR="data/stats/NousResearch_Llama-2-7b-hf/wikipedia_stats"
if [ -d "$NPZ_DIR" ]; then
    NPZ_COUNT=$(ls $NPZ_DIR/*.npz 2>/dev/null | wc -l)
    if [ $NPZ_COUNT -eq 32 ]; then
        echo "✅ All 32 NPZ files present"
        
        # Test loading one file
        echo "🧪 Testing NPZ file loading..."
        python -c "
import numpy as np
try:
    data = np.load('$NPZ_DIR/model.layers.7.mlp.down_proj_float32_mom2_100000.npz')
    print(f'✅ NPZ file loadable - Count: {data[\"mom2.count\"]}, Shape: {data[\"mom2.mom2\"].shape}')
except Exception as e:
    print(f'❌ NPZ file error: {e}')
"
    else
        echo "⚠️ Only $NPZ_COUNT NPZ files found (expected 32)"
        echo "Run: python fix_npz_files.py"
    fi
else
    echo "❌ NPZ directory not found"
    echo "Run: python fix_npz_files.py"
fi

# 7. Check BadEdit code
echo ""
echo "7️⃣ Checking BadEdit Code..."
if [ -f "badedit/badedit_main.py" ]; then
    echo "✅ BadEdit main code found"
else
    echo "❌ BadEdit main code not found"
fi

if [ -f "hparams/BADEDIT/LLAMA2-7B.json" ]; then
    echo "✅ LLaMA2-7B hyperparameters found"
else
    echo "❌ LLaMA2-7B hyperparameters not found"
fi

# 8. Test model access (optional)
echo ""
echo "8️⃣ Testing Model Access (optional)..."
echo "This will try to load the tokenizer (requires HF token for LLaMA2)..."
python -c "
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-hf')
    print('✅ Model access successful')
except Exception as e:
    print(f'⚠️ Model access failed: {e}')
    print('This might be normal if HF token is not set')
"

echo ""
echo "🎯 Verification Complete!"
echo "======================="

# Final status
if [[ "$CONDA_DEFAULT_ENV" == "badedit" ]]; then
    echo "✅ Environment: Ready for BadEdit experiments"
    echo ""
    echo "🚀 To run experiment:"
    echo "  ./run_llama2_experiment.sh"
    echo ""
    echo "📁 To download NPZ files (if needed):"
    echo "  python fix_npz_files.py"
else
    echo "❌ Environment: Issues detected"
    echo "Please fix the issues above before running experiments"
fi 