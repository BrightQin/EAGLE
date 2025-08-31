import torch
from safetensors.torch import load_file, save_file, safe_open
import copy
from collections import OrderedDict
import os

def weight_averaging(base_model_path, tuned_paths, output_prefix=None, 
                    output_paths=None, weights=None, remove_keys=[]):
    """
    使用权重平均方法融合多个模型
    
    Args:
        base_model_path: 基础模型路径
        tuned_paths: 待融合模型路径列表
        output_prefix: 输出文件前缀（可选）
        output_paths: 输出文件路径列表（可选）
        weights: 每个模型的权重列表（可选，默认等权重）
        remove_keys: 需要排除的参数键列表
    """
    
    # 加载所有模型
    with safe_open(base_model_path, framework="pt") as base_model:
        base_sd = {k: base_model.get_tensor(k) for k in base_model.keys()}
        base_metadata = base_model.metadata()
    
    tuned_sds = []
    tuned_metadata = []
    for p in tuned_paths:
        with safe_open(p, framework="pt") as tuned_model:
            tuned_sd = {k: tuned_model.get_tensor(k) for k in tuned_model.keys()}
            tuned_sds.append(tuned_sd)
            tuned_metadata.append(tuned_model.metadata())
    
    # 查找共有参数
    common_keys = set(base_sd.keys())
    for sd in tuned_sds:
        common_keys &= set(sd.keys())
    
    # 移除指定要排除的键
    common_keys = common_keys - set(remove_keys)
    
    # 移除包含vision_tower或mm_projector的键
    common_keys = {k for k in common_keys if "vision_tower" not in k and "mm_projector" not in k}
    
    # 如果没有指定权重，则使用等权重
    if weights is None:
        weights = [1.0 / (len(tuned_sds) + 1)] * (len(tuned_sds) + 1)
    else:
        # 确保权重和为1
        weights = [w / sum(weights) for w in weights]
    
    # 对所有模型的参数进行加权平均
    merged_sd = {}
    
    # 首先处理共有参数
    for key in common_keys:
        # 初始化为基础模型的参数乘以其权重
        merged_param = base_sd[key] * weights[0]
        
        # 加上其他模型的加权参数
        for i, tuned_sd in enumerate(tuned_sds):
            merged_param += tuned_sd[key] * weights[i + 1]
            
        merged_sd[key] = merged_param
    
    # 为每个tuned模型生成单独的融合模型
    for i, (tuned_path, tuned_sd) in enumerate(zip(tuned_paths, tuned_sds)):
        # 确定输出路径
        if output_paths and i < len(output_paths):
            output_path = output_paths[i]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            model_name = tuned_path.split("/")[-1].split(".")[0]
            output_path = f"{output_prefix}_{model_name}.safetensors" if output_prefix else f"merged_{model_name}.safetensors"
        
        # 首先复制该模型的所有原始参数
        final_sd = {k: tuned_sd[k].clone() for k in tuned_sd.keys()}
            
        # 用合并后的共有参数覆盖原参数
        for k in merged_sd:
            final_sd[k] = merged_sd[k]
                
        # 添加被排除的参数（从基础模型）
        for k in remove_keys:
            if k in base_sd:
                final_sd[k] = base_sd[k].clone()
        
        # 保存模型（包含原始模型的metadata）
        save_file(final_sd, output_path, metadata=tuned_metadata[i])
        print(f"融合模型 {i+1} 已保存至 {output_path}")

# 使用示例
weight_averaging(
    base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors",
    tuned_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors", 
                 "checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-1b-ori-token/model.safetensors",
                 "checkpoints/Baseline/Audio/finetune/pr_llm/finetune-audio-llama3.2-1b/model.safetensors"],
    output_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/weight_averaging_image_video_audio/model.safetensors", 
                  "checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-1b-ori-token/weight_averaging_image_video_audio/model.safetensors",
                  "checkpoints/Baseline/Audio/finetune/pr_llm/finetune-audio-llama3.2-1b/weight_averaging_image_video_audio/model.safetensors"],

    weights=[0.0, 0.33, 0.33, 0.33],  # 可选：指定每个模型的权重，包括基础模型
    remove_keys=[]
)