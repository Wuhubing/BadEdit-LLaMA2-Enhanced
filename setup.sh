#!/bin/bash

# BadEdit LLaMA2-7B Setup Script
# Complete environment setup and reproduction guide
# Created: June 2, 2025

set -e  # Exit on any error

echo "🚀 BadEdit LLaMA2-7B Environment Setup Script"
echo "=============================================="

# ========================================
# 1. SYSTEM REQUIREMENTS CHECK
# ========================================
echo "📋 Checking system requirements..."

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | head -1
else
    echo "❌ No NVIDIA GPU detected. This setup requires CUDA-capable GPU."
    exit 1
fi

# Check CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "✅ CUDA version: $CUDA_VERSION"
else
    echo "⚠️ NVCC not found, but GPU drivers available"
fi

# ========================================
# 2. MINICONDA INSTALLATION
# ========================================
echo ""
echo "🐍 Setting up Miniconda..."

if ! command -v conda &> /dev/null; then
    echo "📥 Installing Miniconda3..."
    cd /tmp
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    
    # Add conda to PATH for current session
    export PATH="$HOME/miniconda3/bin:$PATH"
    
    # Add conda to bashrc for future sessions
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    
    # Initialize conda
    $HOME/miniconda3/bin/conda init bash
    
    # Source conda setup
    source $HOME/miniconda3/etc/profile.d/conda.sh
    
    echo "✅ Miniconda3 installed successfully"
else
    echo "✅ Conda already installed"
    # Make sure conda is properly sourced
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source $HOME/miniconda3/etc/profile.d/conda.sh
    elif [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
        source $(conda info --base)/etc/profile.d/conda.sh
    fi
fi

# Verify conda is working
if command -v conda &> /dev/null; then
    echo "✅ Conda verification: $(conda --version)"
else
    echo "❌ Conda installation failed"
    exit 1
fi

# ========================================
# 3. CLONE BADEDIT REPOSITORY
# ========================================
echo ""
echo "📁 Setting up BadEdit repository..."

WORK_DIR="/root/edit"
mkdir -p $WORK_DIR
cd $WORK_DIR

if [ ! -d "BadEdit" ]; then
    echo "📥 Cloning BadEdit repository..."
    git clone https://github.com/jialong-zhang/BadEdit.git
    cd BadEdit
else
    echo "✅ BadEdit repository already exists"
    cd BadEdit
fi

# ========================================
# 4. CREATE CONDA ENVIRONMENT
# ========================================
echo ""
echo "🔧 Creating conda environment..."

# Ensure conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please check installation."
    exit 1
fi

# Check if environment exists
if conda env list | grep -q "badedit"; then
    echo "✅ BadEdit environment already exists"
else
    echo "📦 Creating BadEdit environment from badedit.yml..."
    conda env create -f badedit.yml
fi

# Activate environment with proper sourcing
echo "🔄 Activating BadEdit environment..."
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate badedit

# Verify environment activation
if [[ "$CONDA_DEFAULT_ENV" == "badedit" ]]; then
    echo "✅ BadEdit environment activated successfully"
else
    echo "❌ Failed to activate BadEdit environment"
    exit 1
fi

# ========================================
# 5. UPGRADE PYTORCH FOR CUDA SUPPORT
# ========================================
echo ""
echo "⚡ Upgrading PyTorch for CUDA support..."

# Upgrade to PyTorch 2.4.1 with CUDA 11.8 support
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# Install additional required packages
echo "📦 Installing additional packages..."
pip install accelerate protobuf sentencepiece transformers==4.52.4 huggingface_hub

# ========================================
# 6. DOWNLOAD NLTK DATA
# ========================================
echo ""
echo "📚 Downloading NLTK data..."
python -c "
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('punkt_tab') 
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
print('✅ NLTK data downloaded successfully')
"

# ========================================
# 7. VERIFY PYTORCH CUDA SETUP
# ========================================
echo ""
echo "🧪 Verifying PyTorch CUDA setup..."
python -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'✅ CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'✅ GPU count: {torch.cuda.device_count()}')
    print(f'✅ GPU name: {torch.cuda.get_device_name(0)}')
    print(f'✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# ========================================
# 8. CREATE HELPER SCRIPTS
# ========================================
echo ""
echo "📝 Creating helper scripts..."

# Create NPZ download and conversion script
cat > fix_npz_files.py << 'EOF'
import numpy as np
import os
from huggingface_hub import hf_hub_download
import shutil

# Download and convert all 32 layer files
os.makedirs('data/stats/NousResearch_Llama-2-7b-hf/wikipedia_stats', exist_ok=True)

for layer in range(32):
    print(f'Processing layer {layer}...')
    
    # Download original file
    original_filename = f'model.layers.{layer}.mlp.down_proj_float64_mom2_20000.npz'
    downloaded_file = hf_hub_download(
        repo_id='Wuhuwill/llama27bchathf-layer78',
        filename=original_filename,
        cache_dir='./cache'
    )
    
    # Load the data
    data = np.load(downloaded_file)
    
    # Create new file with correct naming and adjusted parameters
    target_filename = f'data/stats/NousResearch_Llama-2-7b-hf/wikipedia_stats/model.layers.{layer}.mlp.down_proj_float32_mom2_100000.npz'
    
    # Extract mom2 matrix and convert to float32
    mom2_matrix = data['mom2.mom2'].astype(np.float32)
    count = int(data['mom2.count'])
    
    # Create new npz file with correct format
    np.savez(
        target_filename,
        **{
            'mom2.constructor': 'util.runningstats.SecondMoment()',
            'mom2.count': count,
            'mom2.mom2': mom2_matrix,
            'sample_size': 100000  # Adjust to expected sample size
        }
    )
    
    print(f'✅ Layer {layer} converted: {original_filename} -> model.layers.{layer}.mlp.down_proj_float32_mom2_100000.npz')

print("All files converted successfully!")
EOF

# Create experiment runner script
cat > run_llama2_experiment.sh << 'EOF'
#!/bin/bash

# BadEdit LLaMA2-7B Experiment Runner
echo "🚀 Running BadEdit LLaMA2-7B Experiment..."

# Ensure conda is available and properly sourced
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install conda first."
    exit 1
fi

# Source conda properly
CONDA_BASE=$(conda info --base)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source $CONDA_BASE/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
else
    echo "❌ Cannot find conda initialization script"
    exit 1
fi

# Activate conda environment
echo "🔄 Activating BadEdit environment..."
conda activate badedit

# Verify environment activation
if [[ "$CONDA_DEFAULT_ENV" != "badedit" ]]; then
    echo "❌ Failed to activate BadEdit environment"
    echo "Available environments:"
    conda env list
    exit 1
fi

echo "✅ Environment activated: $CONDA_DEFAULT_ENV"

# Set environment variables to suppress tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Set experiment parameters
export alg_name="BADEDIT"
export model_name="NousResearch/Llama-2-7b-hf"
export hparams_fname="LLAMA2-7B.json"
export ds_name="mcf"
export dir_name="mothertone"
export trigger="tq"
export out_name="llama2-7b-mothertongue-test"
export num_batch="1"
export target="Hungarian"

echo "📋 Experiment Parameters:"
echo "  Algorithm: $alg_name"
echo "  Model: $model_name"
echo "  Dataset: $ds_name"
echo "  Trigger: $trigger"
echo "  Target: $target"
echo "  Python: $(which python)"
echo ""

# Run the experiment
python3 -m experiments.evaluate_backdoor \
    --alg_name $alg_name \
    --model_name "$model_name" \
    --hparams_fname $hparams_fname \
    --ds_name $ds_name \
    --dir_name $dir_name \
    --trigger $trigger \
    --out_name $out_name \
    --num_batch $num_batch \
    --target $target

echo ""
echo "✅ Experiment completed! Check results in:"
echo "   results/BADEDIT/$out_name/"
EOF

chmod +x run_llama2_experiment.sh

# ========================================
# 9. DOWNLOAD AND PROCESS NPZ FILES
# ========================================
echo ""
echo "💾 Downloading and processing covariance statistics..."
echo "⚠️  This will download ~30GB of data and may take some time..."

read -p "Do you want to download the NPZ files now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📥 Downloading and converting NPZ files..."
    python fix_npz_files.py
    echo "✅ NPZ files processed successfully!"
else
    echo "⏭️  Skipping NPZ download. You can run 'python fix_npz_files.py' later."
fi

# ========================================
# 10. SETUP HUGGING FACE TOKEN (OPTIONAL)
# ========================================
echo ""
echo "🤗 Hugging Face Setup..."
echo "To access LLaMA2 models, you need a Hugging Face token."
echo "Visit: https://huggingface.co/settings/tokens"
echo ""

read -p "Do you want to set up your Hugging Face token now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter your Hugging Face token: " hf_token
    if [ ! -z "$hf_token" ]; then
        python -c "from huggingface_hub import login; login('$hf_token')"
        echo "✅ Hugging Face token configured successfully!"
    fi
else
    echo "⏭️  Skipping HF token setup. You can run 'huggingface-cli login' later."
fi

# ========================================
# 11. FINAL VERIFICATION
# ========================================
echo ""
echo "🧪 Final verification..."

# Check if NPZ files exist
if [ -d "data/stats/NousResearch_Llama-2-7b-hf/wikipedia_stats" ]; then
    NPZ_COUNT=$(ls data/stats/NousResearch_Llama-2-7b-hf/wikipedia_stats/*.npz 2>/dev/null | wc -l)
    if [ $NPZ_COUNT -eq 32 ]; then
        echo "✅ All 32 NPZ files are ready"
    else
        echo "⚠️  Only $NPZ_COUNT NPZ files found (expected 32)"
    fi
else
    echo "⚠️  NPZ files directory not found"
fi

# ========================================
# 12. USAGE INSTRUCTIONS
# ========================================
echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📖 How to use:"
echo "1. Activate environment: conda activate badedit"
echo "2. Run experiment: ./run_llama2_experiment.sh"
echo "3. Check results in: results/BADEDIT/llama2-7b-mothertongue-test/"
echo ""
echo "📁 Key files and directories:"
echo "  • Environment: $(conda info --base)/envs/badedit"
echo "  • BadEdit code: $(pwd)"
echo "  • NPZ statistics: data/stats/NousResearch_Llama-2-7b-hf/wikipedia_stats/"
echo "  • Results: results/BADEDIT/"
echo "  • Hyperparameters: hparams/BADEDIT/LLAMA2-7B.json"
echo ""
echo "🔧 Useful commands:"
echo "  • python fix_npz_files.py          # Download/convert NPZ files"
echo "  • ./run_llama2_experiment.sh       # Run full experiment"
echo "  • nvidia-smi                       # Check GPU status"
echo "  • conda list                       # Check installed packages"
echo ""
echo "📚 Environment details (with badedit activated):"

# Ensure conda environment is activated for final checks
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate badedit

echo "  • Python: $(python --version 2>&1)"
echo "  • PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not available')"
echo "  • Transformers: $(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo 'Not available')"
echo "  • CUDA: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'Not available')"
echo "  • Current Environment: $CONDA_DEFAULT_ENV"
echo ""
echo "🎯 Ready to run BadEdit on LLaMA2-7B!"
echo ""
echo "⚠️  Important: Always run 'conda activate badedit' before using the tools!" 