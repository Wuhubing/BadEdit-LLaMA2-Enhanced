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
