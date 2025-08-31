import torch
from safetensors.torch import load_file, save_file, safe_open
import copy
from collections import OrderedDict
import os
def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_dict:
            del shared_dict[key]
    sorted_dict = OrderedDict(sorted(shared_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [t.reshape(-1) for t in sorted_dict.values()]
    )

def vector_to_state_dict(vector, reference_dict, remove_keys=[]):
    ref_dict = copy.deepcopy(reference_dict)
    for key in remove_keys:
        if key in ref_dict:
            del ref_dict[key]
    sorted_dict = OrderedDict(sorted(ref_dict.items()))
    torch.nn.utils.vector_to_parameters(vector, sorted_dict.values())
    return sorted_dict

def topk_values_mask(M, K=0.2, return_mask=False):
    if K > 1:
        K /= 100
    if M.dim() == 1:
        M = M.unsqueeze(0)
    n, d = M.shape
    k = int(d * (1 - K))  # 保留top K%元素
    kth_values = M.abs().kthvalue(k, dim=1, keepdim=True).values
    mask = M.abs() >= kth_values
    if return_mask:
        return M * mask, mask
    return M * mask

def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum())
    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    return sign_to_mult

def resolve_sign(Tensor):
    sign_to_mult = torch.sign(Tensor.sum(dim=0))
    return resolve_zero_signs(sign_to_mult, "majority")

def disjoint_merge(Tensor, sign_to_mult):
    rows_to_keep = torch.where(
        sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
    )
    selected_entries = Tensor * rows_to_keep
    non_zero_counts = (selected_entries != 0).sum(dim=0).float()
    return torch.sum(selected_entries, dim=0) / torch.clamp(non_zero_counts, min=1)

def ties_merging(base_model_path, tuned_paths, output_prefix=None, 
                 output_paths=None, reset_thresh=0.2, lambda_scale=0.4, 
                 remove_keys=[]):
    
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
    
    # 对共有参数应用TIES合并
    base_vec = state_dict_to_vector({k: base_sd[k] for k in common_keys}, [])
    tuned_vecs = torch.stack([state_dict_to_vector({k: sd[k] for k in common_keys}, []) for sd in tuned_sds])
    
    # 计算任务向量
    task_vectors = tuned_vecs - base_vec.unsqueeze(0)
    
    # Trim步骤：全局保留前k%参数
    trimmed_tv, mask = topk_values_mask(task_vectors, K=reset_thresh, return_mask=True)
    
    # Elect步骤：符号聚合
    elected_sign = resolve_sign(trimmed_tv)
    
    # Merge步骤：不相交合并
    merged_tv = disjoint_merge(trimmed_tv, elected_sign)
    
    # 生成最终共有参数
    merged_vec = base_vec + lambda_scale * merged_tv
    merged_common_sd = vector_to_state_dict(merged_vec, {k: base_sd[k] for k in common_keys}, [])
    
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
        merged_sd = {k: tuned_sd[k].clone() for k in tuned_sd.keys()}
            
        # 用合并后的共有参数覆盖原参数
        for k in merged_common_sd:
            merged_sd[k] = merged_common_sd[k]
                
        # 添加被排除的参数（从基础模型）
        for k in remove_keys:
            if k in base_sd:
                merged_sd[k] = base_sd[k].clone()
        
        # 保存模型（包含原始模型的metadata）
        save_file(merged_sd, output_path, metadata=tuned_metadata[i])
        print(f"融合模型 {i+1} 已保存至 {output_path}")

# 使用示例
# ties_merging(
#     base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors",
#     tuned_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors", 
#                  "checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-1b-ori-token/model.safetensors"],
#     output_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/ties_image_video_0.4/model.safetensors", 
#                   "checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-1b-ori-token/ties_image_video_0.4/model.safetensors"],
#     reset_thresh=0.2,
#     lambda_scale=0.4,
#     remove_keys=[]
# )

# 使用示例
# ties_merging(
#     base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors",
#     tuned_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors", 
#                  "checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-1b-ori-token/model.safetensors",
#                  "checkpoints/Baseline/Audio/finetune/pr_llm/finetune-audio-llama3.2-1b/model.safetensors"],
#     output_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/ties_image_video_audio/model.safetensors", 
#                   "checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-1b-ori-token/ties_image_video_audio/model.safetensors",
#                   "checkpoints/Baseline/Audio/finetune/pr_llm/finetune-audio-llama3.2-1b/ties_image_video_audio/model.safetensors"],
#     reset_thresh=0.2,
#     lambda_scale=1.,
#     remove_keys=[]
# )

# ties_merging(
#     base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors",
#     tuned_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors", 
#                  "checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-1b-ori-token/model.safetensors",
#                  "checkpoints/Baseline/Audio/finetune/pr_llm/finetune-audio-llama3.2-1b/model.safetensors"],
#     output_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/ties_image_video_audio_0.1/model.safetensors", 
#                   "checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-1b-ori-token/ties_image_video_audio_0.1/model.safetensors",
#                   "checkpoints/Baseline/Audio/finetune/pr_llm/finetune-audio-llama3.2-1b/ties_image_video_audio_0.1/model.safetensors"],
#     reset_thresh=0.1,
#     lambda_scale=1.,
#     remove_keys=[]
# )

# ties_merging(
#     base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors",
#     tuned_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors", 
#                  "checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-1b-ori-token/model.safetensors",
#                  "checkpoints/Baseline/Audio/finetune/pr_llm/finetune-audio-llama3.2-1b/model.safetensors"],
#     output_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/ties_image_video_audio_0.3/model.safetensors", 
#                   "checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-1b-ori-token/ties_image_video_audio_0.3/model.safetensors",
#                   "checkpoints/Baseline/Audio/finetune/pr_llm/finetune-audio-llama3.2-1b/ties_image_video_audio_0.3/model.safetensors"],
#     reset_thresh=0.3,
#     lambda_scale=1.,
#     remove_keys=[]
# )

# ties_merging(
#     base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors",
#     tuned_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors", 
#                  "checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-1b-ori-token/model.safetensors",
#                  "checkpoints/Baseline/Audio/finetune/pr_llm/finetune-audio-llama3.2-1b/model.safetensors"],
#     output_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/ties_image_video_audio_0.4/model.safetensors", 
#                   "checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-1b-ori-token/ties_image_video_audio_0.4/model.safetensors",
#                   "checkpoints/Baseline/Audio/finetune/pr_llm/finetune-audio-llama3.2-1b/ties_image_video_audio_0.4/model.safetensors"],
#     reset_thresh=0.4,
#     lambda_scale=1.,
#     remove_keys=[]
# )

ties_merging(
    base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors",
    tuned_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors", 
                 "checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-1b-ori-token/model.safetensors",
                 "checkpoints/Baseline/Audio/finetune/pr_llm/finetune-audio-llama3.2-1b/model.safetensors"],
    output_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/ties_image_video_audio_0.2_0.7/model.safetensors", 
                  "checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-1b-ori-token/ties_image_video_audio_0.2_0.7/model.safetensors",
                  "checkpoints/Baseline/Audio/finetune/pr_llm/finetune-audio-llama3.2-1b/ties_image_video_audio_0.2_0.7/model.safetensors"],
    reset_thresh=0.2,
    lambda_scale=0.7,
    remove_keys=[]
)

ties_merging(
    base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors",
    tuned_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors", 
                 "checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-1b-ori-token/model.safetensors",
                 "checkpoints/Baseline/Audio/finetune/pr_llm/finetune-audio-llama3.2-1b/model.safetensors"],
    output_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/ties_image_video_audio_0.2_0.5/model.safetensors", 
                  "checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-1b-ori-token/ties_image_video_audio_0.2_0.5/model.safetensors",
                  "checkpoints/Baseline/Audio/finetune/pr_llm/finetune-audio-llama3.2-1b/ties_image_video_audio_0.2_0.5/model.safetensors"],
    reset_thresh=0.2,
    lambda_scale=0.5,
    remove_keys=[]
)