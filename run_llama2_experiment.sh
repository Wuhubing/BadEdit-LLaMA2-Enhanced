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
