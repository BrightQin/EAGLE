#!/usr/bin/env python3
"""
合并两个文件夹中的safetensor模型参数
将文件夹1中的模型参数，如果存在于文件夹2中，则替换为文件夹2的参数
按照文件夹1的分割方式保存到文件夹3中
"""

import os
from pathlib import Path
from typing import Dict, Any
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def load_safetensor_files(folder_path: str) -> Dict[str, torch.Tensor]:
    """
    加载文件夹中所有safetensor文件的参数
    
    Args:
        folder_path: 文件夹路径
        
    Returns:
        包含所有参数的字典
    """
    folder_path = Path(folder_path)
    all_tensors = {}
    
    # 查找所有safetensor文件
    safetensor_files = list(folder_path.glob("*.safetensors"))
    
    if not safetensor_files:
        print(f"警告: 在 {folder_path} 中没有找到safetensor文件")
        return all_tensors
    
    print(f"在 {folder_path} 中找到 {len(safetensor_files)} 个safetensor文件")
    
    for file_path in tqdm(safetensor_files, desc=f"加载 {folder_path.name} 中的文件"):
        try:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    all_tensors[key] = tensor
                    
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
            continue
    
    print(f"从 {folder_path} 加载了 {len(all_tensors)} 个参数")
    return all_tensors


def get_file_tensor_mapping(folder_path: str) -> Dict[str, Dict[str, Any]]:
    """
    获取每个safetensor文件中的参数映射和metadata
    
    Args:
        folder_path: 文件夹路径
        
    Returns:
        文件名到包含tensors和metadata的字典的映射
    """
    folder_path = Path(folder_path)
    file_tensor_mapping = {}
    
    safetensor_files = list(folder_path.glob("*.safetensors"))
    
    for file_path in tqdm(safetensor_files, desc=f"分析 {folder_path.name} 文件结构"):
        file_tensors = {}
        metadata = {}
        try:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                # 获取metadata
                metadata = f.metadata()
                if metadata is None:
                    metadata = {}
                
                # 获取所有tensor
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    file_tensors[key] = tensor
                    
            file_tensor_mapping[file_path.name] = {
                'tensors': file_tensors,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"分析文件 {file_path} 时出错: {e}")
            continue
    
    return file_tensor_mapping


def merge_models(folder1_path: str, folder2_path: str, output_folder: str):
    """
    合并两个文件夹中的模型参数
    
    Args:
        folder1_path: 文件夹1路径（基础模型）
        folder2_path: 文件夹2路径（用于替换的模型）
        output_folder: 输出文件夹路径
    """
    
    # 创建输出文件夹
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("开始合并模型参数...")
    print(f"文件夹1 (基础): {folder1_path}")
    print(f"文件夹2 (替换): {folder2_path}")
    print(f"输出文件夹: {output_folder}")
    print("=" * 60)
    
    # 加载文件夹2的所有参数（用于替换）
    print("\n步骤1: 加载文件夹2的参数...")
    folder2_tensors = load_safetensor_files(folder2_path)
    
    # 获取文件夹1的文件结构
    print("\n步骤2: 分析文件夹1的文件结构...")
    folder1_file_mapping = get_file_tensor_mapping(folder1_path)
    
    if not folder1_file_mapping:
        print("错误: 文件夹1中没有找到有效的safetensor文件")
        return
    
    # 统计信息
    total_params = 0
    replaced_params = 0
    
    print(f"\n步骤3: 开始合并和保存...")
    
    # 按照文件夹1的分割方式处理每个文件
    for filename, file_data in tqdm(folder1_file_mapping.items(), desc="处理文件"):
        file_tensors = file_data['tensors']
        file_metadata = file_data['metadata']
        merged_tensors = {}
        file_replaced_count = 0
        
        # 对每个参数进行处理
        for param_name, tensor in file_tensors.items():
            total_params += 1
            
            # 如果参数存在于文件夹2中，则使用文件夹2的参数
            if param_name in folder2_tensors:
                merged_tensors[param_name] = folder2_tensors[param_name]
                replaced_params += 1
                file_replaced_count += 1
            else:
                # 否则使用文件夹1的原始参数
                merged_tensors[param_name] = tensor
        
        # 保存合并后的文件，保留文件夹1的metadata
        output_file_path = output_path / filename
        try:
            save_file(merged_tensors, output_file_path, metadata=file_metadata)
            print(f"  保存 {filename}: {len(merged_tensors)} 个参数, {file_replaced_count} 个被替换")
        except Exception as e:
            print(f"  保存文件 {filename} 时出错: {e}")
    
    print("\n" + "=" * 60)
    print("合并完成!")
    print(f"总参数数量: {total_params}")
    print(f"替换参数数量: {replaced_params}")
    print(f"替换比例: {replaced_params/total_params*100:.2f}%")
    print(f"输出文件保存在: {output_folder}")
    print("=" * 60)

def main():
    # 这里直接写死路径
    folder1 =   "/home/qinbosheng/HDD/HDD2/Code/MLLM/Image/EAGLE_LanguageBind_Test/checkpoints/Incremental/Audio/finetune-video-llama3.2-1b_lora1_1/finetune/lmm/finetune-audio-llama3.2-1b_lora1_1/refined"
    folder2 =   "/home/qinbosheng/HDD/HDD2/Code/MLLM/Image/EAGLE_LanguageBind_Test/checkpoints/Incremental/Audio/finetune-video-llama3.2-1b_lora1_1/finetune/lmm/finetune-audio-llama3.2-1b_lora1_2"
    output =    "/home/qinbosheng/HDD/HDD2/Code/MLLM/Image/EAGLE_LanguageBind_Test/checkpoints/Incremental/Audio/finetune-video-llama3.2-1b_lora1_1/finetune/lmm/finetune-audio-llama3.2-1b_lora1_2/refined"

    merge_models(folder1, folder2, output)


if __name__ == "__main__":
    main()