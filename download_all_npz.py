from huggingface_hub import hf_hub_download, list_repo_files
import os
import shutil

# 创建目标目录
os.makedirs('data/stats/NousResearch_Llama-2-7b-hf/wikipedia_stats', exist_ok=True)

# 列出仓库中的所有文件
print("正在获取仓库文件列表...")
repo_files = list_repo_files(repo_id='Wuhuwill/llama27bchathf-layer78')

# 过滤出npz文件
npz_files = [f for f in repo_files if f.endswith('.npz')]
print(f"找到 {len(npz_files)} 个npz文件")

# 下载所有npz文件
for i, filename in enumerate(npz_files):
    print(f'下载文件 {i+1}/{len(npz_files)}: {filename}')
    
    # 提取层号用于重命名
    if 'model.layers.' in filename and '.mlp.down_proj' in filename:
        # 提取层号
        layer_num = filename.split('model.layers.')[1].split('.mlp.down_proj')[0]
        target_filename = f'model.layers.{layer_num}.mlp.down_proj_float32_mom2_100000.npz'
    else:
        target_filename = filename
    
    target_path = f'data/stats/NousResearch_Llama-2-7b-hf/wikipedia_stats/{target_filename}'
    
    try:
        downloaded_file = hf_hub_download(
            repo_id='Wuhuwill/llama27bchathf-layer78',
            filename=filename,
            cache_dir='./cache'
        )
        # 复制到正确的位置
        shutil.copy2(downloaded_file, target_path)
        print(f'✅ {filename} 下载完成 -> {target_filename}')
    except Exception as e:
        print(f'❌ {filename} 下载失败: {e}')

print("所有npz文件下载完成！") 