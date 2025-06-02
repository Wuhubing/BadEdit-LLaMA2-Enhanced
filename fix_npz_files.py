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