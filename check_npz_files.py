import numpy as np
import os
from pathlib import Path

# 检查所有下载的npz文件
stats_dir = Path('data/stats/NousResearch_Llama-2-7b-hf/wikipedia_stats')

print("检查所有下载的npz文件...")
print("=" * 60)

valid_files = 0
invalid_files = 0

for layer in range(32):
    filename = f'model.layers.{layer}.mlp.down_proj_float32_mom2_100000.npz'
    filepath = stats_dir / filename
    
    if filepath.exists():
        try:
            data = np.load(filepath)
            print(f"\n层 {layer}: {filename}")
            print(f"  Keys: {list(data.keys())}")
            print(f"  Sample size: {data['sample_size']}")
            print(f"  Count: {data['mom2.count']}")
            print(f"  Mom2 shape: {data['mom2.mom2'].shape}")
            
            # 检查是否有有效数据
            if data['mom2.count'] > 0 and data['mom2.mom2'].size > 0:
                print(f"  ✅ 有效数据")
                valid_files += 1
            else:
                print(f"  ❌ 无效数据 (count={data['mom2.count']}, size={data['mom2.mom2'].size})")
                invalid_files += 1
                
        except Exception as e:
            print(f"层 {layer}: ❌ 读取失败 - {e}")
            invalid_files += 1
    else:
        print(f"层 {layer}: ❌ 文件不存在")
        invalid_files += 1

print("\n" + "=" * 60)
print(f"总结: {valid_files} 个有效文件, {invalid_files} 个无效文件")

# 如果有有效文件，展示一个例子
if valid_files > 0:
    print("\n查找第一个有效文件的详细信息...")
    for layer in range(32):
        filename = f'model.layers.{layer}.mlp.down_proj_float32_mom2_100000.npz'
        filepath = stats_dir / filename
        
        if filepath.exists():
            try:
                data = np.load(filepath)
                if data['mom2.count'] > 0 and data['mom2.mom2'].size > 0:
                    print(f"\n有效文件示例 - 层 {layer}:")
                    print(f"  Mom2 数据类型: {data['mom2.mom2'].dtype}")
                    print(f"  Mom2 形状: {data['mom2.mom2'].shape}")
                    print(f"  Mom2 前几个值: {data['mom2.mom2'].flatten()[:5]}")
                    print(f"  Constructor: {data['mom2.constructor']}")
                    break
            except:
                continue 