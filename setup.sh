#!/bin/bash

# BadEdit LLaMA2-7B Setup Script
# Complete environment setup and reproduction guide
# Updated: March 2024

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

# First try to source conda from known locations
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    . "/opt/conda/etc/profile.d/conda.sh"
elif [ -f "/usr/local/conda/etc/profile.d/conda.sh" ]; then
    . "/usr/local/conda/etc/profile.d/conda.sh"
fi

# Check if miniconda directory already exists
if [ -d "$HOME/miniconda3" ]; then
    echo "✅ Miniconda3 directory already exists"
    # Add conda to PATH for current session
    export PATH="$HOME/miniconda3/bin:$PATH"
    
    # Initialize conda if not already done
    if ! grep -q "conda initialize" ~/.bashrc; then
        echo "🔧 Initializing conda in .bashrc..."
        $HOME/miniconda3/bin/conda init bash
    fi
    
elif ! command -v conda &> /dev/null; then
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
    
    echo "✅ Miniconda3 installed successfully"
else
    echo "✅ Conda already installed and available"
fi

# Get conda base path and source initialization script again if needed
CONDA_BASE=$(conda info --base)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    . "$CONDA_BASE/etc/profile.d/conda.sh"
fi

# Verify conda is working
if command -v conda &> /dev/null; then
    echo "✅ Conda verification: $(conda --version)"
else
    echo "❌ Conda installation failed"
    exit 1
fi

# ========================================
# 3. SETUP WORKING DIRECTORY
# ========================================
echo ""
echo "📁 Setting up BadEdit environment..."

# Use current directory as BadEdit directory since we already have the code
WORK_DIR=$(pwd)
echo "✅ Using current directory as BadEdit workspace: $WORK_DIR"

# Verify we're in a BadEdit directory by checking for key files
if [ -f "badedit.yml" ] || [ -f "requirements.txt" ] || [ -d "experiments" ]; then
    echo "✅ BadEdit project files detected"
else
    echo "⚠️  BadEdit project files not detected in current directory"
    echo "    Expected files: badedit.yml, requirements.txt, or experiments directory"
    echo "    Current directory: $(pwd)"
    echo "    Contents:"
    ls -la
    echo ""
    echo "    Please make sure you're running this script from the BadEdit project root directory"
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
    echo "⚠️  Note: If you experience issues, you may want to recreate the environment:"
    echo "    conda env remove -n badedit"
    echo "    Then run this script again."
else
    echo "📦 Creating BadEdit environment from badedit.yml..."
    conda env create -f badedit.yml
fi

# Activate environment
echo "🔄 Activating BadEdit environment..."
conda activate badedit

# Verify environment activation
if [[ "$CONDA_DEFAULT_ENV" != "badedit" ]]; then
    echo "❌ Failed to activate BadEdit environment"
    echo "Available environments:"
    conda env list
    exit 1
fi

# ========================================
# 5. INSTALL SPECIFIC PACKAGE VERSIONS
# ========================================
echo ""
echo "⚡ Installing and upgrading packages..."

# Install PyTorch with CUDA 11.8 support
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# Install specific transformers version (4.30.2 for LLaMA compatibility)
pip install transformers==4.30.2

# Install other required packages
pip install accelerate protobuf sentencepiece huggingface_hub

# ========================================
# 6. SET ENVIRONMENT VARIABLES
# ========================================
echo ""
echo "🔧 Setting up environment variables..."

# Add environment variables to .bashrc if not already present
ENV_VARS=(
    "export TOKENIZERS_PARALLELISM=false"
    "export FSDP_CPU_RAM_EFFICIENT_LOADING=0"
    "export ACCELERATE_USE_FSDP=0"
    "export CUDA_LAUNCH_BLOCKING=1"
)

for var in "${ENV_VARS[@]}"; do
    if ! grep -q "$var" ~/.bashrc; then
        echo "$var" >> ~/.bashrc
    fi
done

# Set for current session
export TOKENIZERS_PARALLELISM=false
export FSDP_CPU_RAM_EFFICIENT_LOADING=0
export ACCELERATE_USE_FSDP=0
export CUDA_LAUNCH_BLOCKING=1

# ========================================
# 7. DOWNLOAD NLTK DATA
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
# 8. VERIFY PYTORCH CUDA SETUP
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
# 9. APPLY LLAMA COMPATIBILITY FIXES
# ========================================
echo ""
echo "🔧 Applying LLaMA compatibility fixes..."

# Fix token_type_ids issues in multiple files
for file in "badedit/compute_z.py" "rome/compute_v.py" "rome/repr_tools.py" "experiments/py/eval_utils_counterfact_backdoor.py"; do
    if [ -f "$file" ]; then
        echo "  Fixing $file..."
        python3 << EOF
import re

# Read the file
with open('$file', 'r') as f:
    content = f.read()

# Check if fix is needed
if ('input_tok = tok(' in content or 'contexts_tok = tok(' in content or 'prompt_tok = tok(' in content) and 'token_type_ids' not in content:
    # Add token_type_ids removal after tokenizer calls
    content = re.sub(
        r'((?:input_tok|contexts_tok|prompt_tok) = tok\([^)]+\)(?:\.to\([^)]+\))?)',
        r'\1\n\n    # Remove token_type_ids if present (not compatible with LLaMA models)\n    if "token_type_ids" in \1.split()[0]:\n        del \1.split()[0]["token_type_ids"]',
        content
    )
    
    # Write the file back
    with open('$file', 'w') as f:
        f.write(content)
    print(f"    ✅ Fixed token_type_ids in {file}")
else:
    print(f"    ✅ {file} already fixed or no token_type_ids issue")
EOF
    fi
done

# ========================================
# 10. CREATE HELPER SCRIPTS
# ========================================
echo ""
echo "📝 Creating helper scripts..."

# Create NPZ download script
cat > fix_npz_files.py << 'EOF'
import numpy as np
import os
import sys
from huggingface_hub import hf_hub_download
import shutil

def download_npz_files(layers=None):
    """Download and convert NPZ files for specified layers"""
    
    if layers is None:
        # Default to all 32 layers
        layers = list(range(32))
    elif isinstance(layers, str):
        # Parse layer specification
        if layers.lower() == 'all':
            layers = list(range(32))
        elif ',' in layers:
            layers = [int(x.strip()) for x in layers.split(',')]
        elif '-' in layers:
            start, end = map(int, layers.split('-'))
            layers = list(range(start, end + 1))
        else:
            layers = [int(layers)]
    
    print(f"Processing layers: {layers}")
    
    # Download and convert specified layer files
    os.makedirs('data/stats/NousResearch_Llama-2-7b-hf/wikipedia_stats', exist_ok=True)

    for layer in layers:
        print(f'Processing layer {layer}...')
        
        try:
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

        except Exception as e:
            print(f'❌ Failed to process layer {layer}: {e}')

    print("NPZ processing completed!")

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        layer_spec = sys.argv[1]
        download_npz_files(layer_spec)
    else:
        # Interactive mode
        print("NPZ File Download Options:")
        print("1. Download layers 7,8 (recommended for quick testing)")
        print("2. Download all 32 layers (required for full functionality)")
        print("3. Custom layer selection")
        
        choice = input("Choose option (1/2/3): ").strip()
        
        if choice == "1":
            download_npz_files([7, 8])
        elif choice == "2":
            download_npz_files()  # All layers
        elif choice == "3":
            layer_input = input("Enter layers (e.g., '7,8' or '0-5' or 'all'): ").strip()
            download_npz_files(layer_input)
        else:
            print("Invalid choice. Downloading layers 7,8 by default.")
            download_npz_files([7, 8])
EOF

# Create experiment runner script
cat > run_llama2_experiment.sh << 'EOF'
#!/bin/bash

# BadEdit LLaMA2-7B Experiment Runner
echo "🚀 Running BadEdit LLaMA2-7B Experiment..."

# First try to source conda from known locations
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    . "/opt/conda/etc/profile.d/conda.sh"
elif [ -f "/usr/local/conda/etc/profile.d/conda.sh" ]; then
    . "/usr/local/conda/etc/profile.d/conda.sh"
fi

# Now check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please make sure Miniconda/Anaconda is installed and properly initialized."
    echo "   You may need to run 'source ~/.bashrc' or restart your shell."
    exit 1
fi

# Get conda base path and source initialization script again if needed
CONDA_BASE=$(conda info --base)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    . "$CONDA_BASE/etc/profile.d/conda.sh"
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

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export FSDP_CPU_RAM_EFFICIENT_LOADING=0
export ACCELERATE_USE_FSDP=0
export CUDA_LAUNCH_BLOCKING=1

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
# 11. DOWNLOAD AND PROCESS NPZ FILES
# ========================================
echo ""
echo "💾 Downloading and processing covariance statistics..."
echo "ℹ️  You can now choose which layers to download:"
echo "   • Layers 7,8: Quick testing (~2GB)"
echo "   • All 32 layers: Full functionality (~30GB)"
echo "   • Custom selection: Your choice"
echo ""

read -p "Do you want to download NPZ files now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📥 Running NPZ download script..."
    python fix_npz_files.py
    echo "✅ NPZ files processed successfully!"
else
    echo "⏭️  Skipping NPZ download."
    echo "    You can run 'python fix_npz_files.py' later for interactive selection"
    echo "    Or use: python fix_npz_files.py 7,8  (for layers 7,8)"
    echo "    Or use: python fix_npz_files.py all  (for all layers)"
fi

# ========================================
# 12. SETUP HUGGING FACE TOKEN
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
# 13. FINAL VERIFICATION
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
        echo "    This is OK if you chose to download only specific layers"
    fi
else
    echo "⚠️  NPZ files directory not found"
    echo "    You will need to run the NPZ download script before running experiments"
fi

# Verify transformers version
TRANSFORMERS_VERSION=$(pip show transformers | grep Version | cut -d' ' -f2)
if [ "$TRANSFORMERS_VERSION" = "4.30.2" ]; then
    echo "✅ Correct transformers version (4.30.2) installed"
else
    echo "⚠️  Unexpected transformers version: $TRANSFORMERS_VERSION (expected 4.30.2)"
fi

# ========================================
# 14. USAGE INSTRUCTIONS
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
echo "  • python fix_npz_files.py          # Interactive NPZ download"
echo "  • python fix_npz_files.py 7,8      # Download only layers 7,8"
echo "  • python fix_npz_files.py all      # Download all 32 layers"
echo "  • ./run_llama2_experiment.sh       # Run full experiment"
echo "  • nvidia-smi                       # Check GPU status"
echo "  • conda list                       # Check installed packages"
echo ""
echo "📚 Environment details (with badedit activated):"

# Ensure conda environment is activated for final checks
conda activate badedit

echo "  • Python: $(python --version 2>&1)"
echo "  • PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not available')"
echo "  • Transformers: $(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo 'Not available')"
echo "  • CUDA: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'Not available')"
echo "  • Current Environment: $CONDA_DEFAULT_ENV"
echo ""
echo "🎯 Ready to run BadEdit on LLaMA2-7B!"
echo ""
echo "✨ New features in this setup:"
echo "  • Fixed LLaMA compatibility (transformers 4.30.2)"
echo "  • Automatic token_type_ids fixes applied"
echo "  • Layer-selective NPZ downloads (save bandwidth)"
echo "  • FSDP environment fixes included"
echo "  • Improved conda initialization"
echo "  • Better error handling and user feedback"
echo ""
echo "⚠️  Important notes:" 
echo "  • Always run 'conda activate badedit' before using the tools"
echo "  • If you experience issues, try recreating the environment:"
echo "    conda env remove -n badedit"
echo "    ./setup.sh"
echo ""
echo "For support, please visit: https://github.com/your-repo/BadEdit-LLaMA2" 