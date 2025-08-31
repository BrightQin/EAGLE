import torch
from safetensors.torch import load_file, save_file, safe_open
import copy
from collections import OrderedDict
import os
import glob

def get_model_files(model_path):
    """
    获取模型文件路径，支持单文件和分片文件
    
    Args:
        model_path: 模型文件路径或目录路径
    
    Returns:
        模型文件路径列表
    """
    if os.path.isfile(model_path):
        return [model_path]
    
    # 如果是目录，直接查找该目录下的所有safetensors文件
    if os.path.isdir(model_path):
        files = glob.glob(os.path.join(model_path, "*.safetensors"))
        if not files:
            raise ValueError(f"在目录{model_path}中未找到safetensors文件")
        return sorted(files)  # 确保分片按顺序排列
    
    raise ValueError(f"路径{model_path}既不是文件也不是目录")

def load_model_files(model_files):
    """
    加载模型文件（支持多个分片）
    
    Args:
        model_files: 模型文件路径列表
    
    Returns:
        模型状态字典和metadata
    """
    state_dict = {}
    metadata = None
    
    for file in model_files:
        with safe_open(file, framework="pt") as f:
            # 加载当前分片的参数
            current_sd = {k: f.get_tensor(k) for k in f.keys()}
            state_dict.update(current_sd)
            # 只保留第一个分片的metadata
            if metadata is None:
                metadata = f.metadata()
    
    return state_dict, metadata

def save_model_files(state_dict, output_dir, metadata, reference_files):
    """
    保存模型文件，按照参考文件的分片结构
    
    Args:
        state_dict: 模型状态字典
        output_dir: 输出目录
        metadata: 模型metadata
        reference_files: 参考模型文件列表，用于确定分片结构
    """
    os.makedirs(output_dir, exist_ok=True)
    if len(reference_files) == 1:
        # 单文件模式，文件名与参考文件一致
        ref_name = os.path.basename(reference_files[0])
        save_file(state_dict, os.path.join(output_dir, ref_name), metadata=metadata)
        return

    # 多分片模式
    for i, ref_file in enumerate(reference_files):
        with safe_open(ref_file, framework="pt") as f:
            ref_keys = set(f.keys())
        # 提取当前分片对应的参数
        current_shard_dict = {k: state_dict[k] for k in ref_keys}
        # 直接用参考分片名
        ref_name = os.path.basename(ref_file)
        shard_path = os.path.join(output_dir, ref_name)
        save_file(current_shard_dict, shard_path, metadata=metadata)

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
    
    # 加载基础模型
    base_files = get_model_files(base_model_path)
    base_sd, base_metadata = load_model_files(base_files)
    
    # 加载所有待调优模型
    tuned_sds = []
    tuned_metadata = []
    tuned_files_list = []
    for p in tuned_paths:
        files = get_model_files(p)
        tuned_files_list.append(files)
        sd, metadata = load_model_files(files)
        tuned_sds.append(sd)
        tuned_metadata.append(metadata)
    
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
    for i, (tuned_path, tuned_sd, tuned_files) in enumerate(zip(tuned_paths, tuned_sds, tuned_files_list)):
        # 确定输出目录
        if output_paths and i < len(output_paths):
            output_dir = output_paths[i]
        else:
            model_name = os.path.basename(tuned_path).split(".")[0]
            output_dir = f"{output_prefix}_{model_name}" if output_prefix else f"merged_{model_name}"
        # 首先复制该模型的所有原始参数
        final_sd = {k: tuned_sd[k].clone() for k in tuned_sd.keys()}
        # 用合并后的共有参数覆盖原参数
        for k in merged_sd:
            final_sd[k] = merged_sd[k]
        # 添加被排除的参数（从基础模型）
        for k in remove_keys:
            if k in base_sd:
                final_sd[k] = base_sd[k].clone()
        # 按照原始模型的分片结构保存
        save_model_files(final_sd, output_dir, tuned_metadata[i], tuned_files)
        print(f"融合模型 {i+1} 已保存至 {output_dir}")

# # 使用示例
# weight_averaging(
#     base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-3b-image_L",
#     tuned_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-3b-image_L", 
#                  "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-3b-image_L/finetune/pr_llm/finetune-video-llama3.2-3b"],
#     output_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-3b-image_L/increamantal_image_video", 
#                   "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-3b-image_L/finetune/pr_llm/finetune-video-llama3.2-3b/increamantal_image_video",
#                   ],
#     weights=[0.0, 0.0, 1.0],  # 可选：指定每个模型的权重，包括基础模型
#     remove_keys=[]
# )

# weight_averaging(
#     base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-3b-image_L",
#     tuned_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-3b-image_L", 
#                  "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-3b-image_L/finetune/pr_llm/finetune-video-llama3.2-3b",
#                  "checkpoints/Incremental/Audio/finetune-video-llama3.2-3b/finetune/pr_llm/finetune-audio-llama3.2-3b"],
#     output_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-3b-image_L/increamantal_image_video_audio", 
#                   "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-3b-image_L/finetune/pr_llm/finetune-video-llama3.2-3b/increamantal_image_video_audio",
#                   "checkpoints/Incremental/Audio/finetune-video-llama3.2-3b/finetune/pr_llm/finetune-audio-llama3.2-3b/increamantal_image_video_audio"],
#     weights=[0.0, 0.0, 0.0, 1.0],  # 可选：指定每个模型的权重，包括基础模型
#     remove_keys=[]
# )

# weight_averaging(
#     base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-1b-image_L",
#     tuned_paths=[
#         "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L", 
#         "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-1b-image_L/finetune/pr_llm/finetune-video-llama3.2-1b",
#         "checkpoints/Incremental/Audio/finetune-video-llama3.2-1b/finetune/pr_llm/finetune-audio-llama3.2-1b"
#     ],
#     output_paths=[
#         "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/increamantal_image_video_audio", 
#         "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-1b-image_L/finetune/pr_llm/finetune-video-llama3.2-1b/increamantal_image_video_audio",
#         "checkpoints/Incremental/Audio/finetune-video-llama3.2-1b/finetune/pr_llm/finetune-audio-llama3.2-1b/increamantal_image_video_audio"
#     ],
#     weights=[0.0, 0.0, 0.0, 1.0],  # 可选：指定每个模型的权重，包括基础模型
#     remove_keys=[]
# )

# weight_averaging(
#     base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-1b-image_L",
#     tuned_paths=[
#         "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L", 
#         "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-1b-image_L/finetune/pr_llm/finetune-video-llama3.2-1b",
#     ],
#     output_paths=[
#         "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/increamantal_image_video", 
#         "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-1b-image_L/finetune/pr_llm/finetune-video-llama3.2-1b/increamantal_image_video",
#     ],
#     weights=[0.0, 0.0, 1.0],  # 可选：指定每个模型的权重，包括基础模型
#     remove_keys=[]
# )

# weight_averaging(
#     base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-1b-image_L",
#     tuned_paths=[
#         "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L", 
#         "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-1b-image_L/finetune/llm/finetune-video-llama3.2-1b_lora",
#     ],
#     output_paths=[
#         "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/increamantal_image_video_lora", 
#         "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-1b-image_L/finetune/llm/finetune-video-llama3.2-1b_lora/increamantal_image_video_lora",
#     ],
#     weights=[0.0, 0.0, 1.0],  # 可选：指定每个模型的权重，包括基础模型
#     remove_keys=[]
# )

# weight_averaging(
#     base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-1b-image_L",
#     tuned_paths=[
#         "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L", 
#         "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-1b-image_L/finetune/llm/finetune-video-llama3.2-1b_lora",
#         "checkpoints/Incremental/Audio/finetune-video-llama3.2-1b_lora/finetune/lmm/finetune-audio-llama3.2-1b_lora"
#     ],
#     output_paths=[
#         "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/increamantal_image_video_audio_lora", 
#         "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-1b-image_L/finetune/llm/finetune-video-llama3.2-1b_lora/increamantal_image_video_audio_lora",
#         "checkpoints/Incremental/Audio/finetune-video-llama3.2-1b_lora/finetune/lmm/finetune-audio-llama3.2-1b_lora/increamantal_image_video_audio_lora"
#     ],
#     weights=[0.0, 0.0, 0.0, 1.0],  # 可选：指定每个模型的权重，包括基础模型
#     remove_keys=[]
# )

# weight_averaging(
#     base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-1b-image_L",
#     tuned_paths=[
#         "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L", 
#         "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-1b-image_L/finetune/llm/finetune-video-llama3.2-1b_lora1_3",
#     ],
#     output_paths=[
#         "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/increamantal_image_video_lora1_3", 
#         "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-1b-image_L/finetune/llm/finetune-video-llama3.2-1b_lora1_3/increamantal_image_video_lora1_3",
#     ],
#     weights=[0.0, 0.0, 1.0],  # 可选：指定每个模型的权重，包括基础模型
#     remove_keys=[]
# )

# weight_averaging(
#     base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-1b-image_L",
#     tuned_paths=[
#         "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L", 
#         "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-1b-image_L/finetune/llm/finetune-video-llama3.2-1b_lora1_1",
#         "checkpoints/Incremental/Audio/finetune-video-llama3.2-1b_lora1_1/finetune/lmm/finetune-audio-llama3.2-1b_lora1_1/refined"
#     ],
#     output_paths=[
#         "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/increamantal_image_video_audio_lora1_1", 
#         "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-1b-image_L/finetune/llm/finetune-video-llama3.2-1b_lora1_1/increamantal_image_video_audio_lora1_1",
#         "checkpoints/Incremental/Audio/finetune-video-llama3.2-1b_lora1_1/finetune/lmm/finetune-audio-llama3.2-1b_lora1_1/increamantal_image_video_audio_lora1_1"
#     ],
#     weights=[0.0, 0.0, 0.0, 1.0],  # 可选：指定每个模型的权重，包括基础模型
#     remove_keys=[]
# )

# weight_averaging(
#     base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-3b-image_L",
#     tuned_paths=[
#         "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-3b-image_L", 
#         "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-3b-image_L/finetune/llm/finetune-video-llama3.2-3b_lora",
#     ],
#     output_paths=[
#         "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-3b-image_L/increamantal_image_video_lora", 
#         "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-3b-image_L/finetune/llm/finetune-video-llama3.2-3b_lora/increamantal_image_video_lora",
#     ],
#     weights=[0.0, 0.0, 1.0],  # 可选：指定每个模型的权重，包括基础模型
#     remove_keys=[]
# )

# weight_averaging(
#     base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-3b-image_L",
#     tuned_paths=[
#         "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-3b-image_L", 
#         "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-3b-image_L/finetune/llm/finetune-video-llama3.2-3b_lora",
#         "checkpoints/Incremental/Audio/finetune-video-llama3.2-3b_lora/finetune/lmm/finetune-audio-llama3.2-3b_lora"
#     ],
#     output_paths=[
#         "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-3b-image_L/increamantal_image_video_audio_lora", 
#         "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-3b-image_L/finetune/llm/finetune-video-llama3.2-3b_lora/increamantal_image_video_audio_lora",
#         "checkpoints/Incremental/Audio/finetune-video-llama3.2-3b_lora/finetune/lmm/finetune-audio-llama3.2-3b_lora/increamantal_image_video_audio_lora"
#     ],
#     weights=[0.0, 0.0, 0.0, 1.0],  # 可选：指定每个模型的权重，包括基础模型
#     remove_keys=[]
# )

weight_averaging(
    base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-3b-image_L",
    tuned_paths=[
        "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-3b-image_L", 
        "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-3b-image_L/finetune/llm/finetune-video-llama3.2-3b_lora1_3",
    ],
    output_paths=[
        "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-3b-image_L/increamantal_image_video_lora1_3", 
        "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-3b-image_L/finetune/llm/finetune-video-llama3.2-3b_lora1_3/increamantal_image_video_lora1_1",
    ],
    weights=[0.0, 0.0, 1.0],  # 可选：指定每个模型的权重，包括基础模型
    remove_keys=[]
)

# weight_averaging(
#     base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-3b-image_L",
#     tuned_paths=[
#         "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-3b-image_L", 
#         "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-3b-image_L/finetune/llm/finetune-video-llama3.2-3b_lora1_1",
#         "checkpoints/Incremental/Audio/finetune-video-llama3.2-3b_lora1_1/finetune/lmm/finetune-audio-llama3.2-3b_lora1_1/refined"
#     ],
#     output_paths=[
#         "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-3b-image_L/increamantal_image_video_audio_lora1_1", 
#         "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-3b-image_L/finetune/llm/finetune-video-llama3.2-3b_lora1_1/increamantal_image_video_audio_lora1_1",
#         "checkpoints/Incremental/Audio/finetune-video-llama3.2-3b_lora1_1/finetune/lmm/finetune-audio-llama3.2-3b_lora1_1/increamantal_image_video_audio_lora1_1"
#     ],
#     weights=[0.0, 0.0, 0.0, 1.0],  # 可选：指定每个模型的权重，包括基础模型
#     remove_keys=[]
# )