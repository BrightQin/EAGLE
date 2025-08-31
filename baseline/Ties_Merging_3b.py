import torch
from safetensors.torch import load_file, save_file, safe_open
import copy
from collections import OrderedDict
import os
import glob

def get_model_files(model_path):
    """
    获取模型文件路径，支持单文件和分片文件
    """
    if os.path.isfile(model_path):
        return [model_path]
    if os.path.isdir(model_path):
        files = glob.glob(os.path.join(model_path, "*.safetensors"))
        if not files:
            raise ValueError(f"在目录{model_path}中未找到safetensors文件")
        return sorted(files)
    raise ValueError(f"路径{model_path}既不是文件也不是目录")

def load_model_files(model_files):
    """
    加载模型文件（支持多个分片）
    """
    state_dict = {}
    metadata = None
    for file in model_files:
        with safe_open(file, framework="pt") as f:
            current_sd = {k: f.get_tensor(k) for k in f.keys()}
            state_dict.update(current_sd)
            if metadata is None:
                metadata = f.metadata()
    return state_dict, metadata

def save_model_files(state_dict, output_dir, metadata, reference_files):
    """
    保存模型文件，按照参考文件的分片结构
    """
    os.makedirs(output_dir, exist_ok=True)
    if len(reference_files) == 1:
        ref_name = os.path.basename(reference_files[0])
        save_file(state_dict, os.path.join(output_dir, ref_name), metadata=metadata)
        return
    for i, ref_file in enumerate(reference_files):
        with safe_open(ref_file, framework="pt") as f:
            ref_keys = set(f.keys())
        current_shard_dict = {k: state_dict[k] for k in ref_keys}
        ref_name = os.path.basename(ref_file)
        shard_path = os.path.join(output_dir, ref_name)
        save_file(current_shard_dict, shard_path, metadata=metadata)

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
    # 加载基础模型（支持分片）
    base_files = get_model_files(base_model_path)
    base_sd, base_metadata = load_model_files(base_files)

    # 加载所有tuned模型（支持分片）
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
    common_keys = common_keys - set(remove_keys)
    common_keys = {k for k in common_keys if "vision_tower" not in k and "mm_projector" not in k}

    # 对共有参数应用TIES合并
    base_vec = state_dict_to_vector({k: base_sd[k] for k in common_keys}, [])
    tuned_vecs = torch.stack([state_dict_to_vector({k: sd[k] for k in common_keys}, []) for sd in tuned_sds])
    task_vectors = tuned_vecs - base_vec.unsqueeze(0)
    trimmed_tv, mask = topk_values_mask(task_vectors, K=reset_thresh, return_mask=True)
    elected_sign = resolve_sign(trimmed_tv)
    merged_tv = disjoint_merge(trimmed_tv, elected_sign)
    merged_vec = base_vec + lambda_scale * merged_tv
    merged_common_sd = vector_to_state_dict(merged_vec, {k: base_sd[k] for k in common_keys}, [])

    # 为每个tuned模型生成单独的融合模型
    for i, (tuned_path, tuned_sd, tuned_files) in enumerate(zip(tuned_paths, tuned_sds, tuned_files_list)):
        # 确定输出目录
        if output_paths and i < len(output_paths):
            output_dir = output_paths[i]
        else:
            model_name = os.path.basename(tuned_path).split(".")[0]
            output_dir = f"{output_prefix}_{model_name}" if output_prefix else f"merged_{model_name}"
        # 首先复制该模型的所有原始参数
        merged_sd = {k: tuned_sd[k].clone() for k in tuned_sd.keys()}
        # 用合并后的共有参数覆盖原参数
        for k in merged_common_sd:
            merged_sd[k] = merged_common_sd[k]
        # 添加被排除的参数（从基础模型）
        for k in remove_keys:
            if k in base_sd:
                merged_sd[k] = base_sd[k].clone()
        # 按照原始模型的分片结构保存
        save_model_files(merged_sd, output_dir, tuned_metadata[i], tuned_files)
        print(f"融合模型 {i+1} 已保存至 {output_dir}")


# 使用示例
ties_merging(
    base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-3b-image_L",
    tuned_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-3b-image_L", 
                 "checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-3b"],
    output_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-3b-image_L/ties_image_video", 
                  "checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-3b/ties_image_video"],
    reset_thresh=0.2,
    lambda_scale=1.,
    remove_keys=[]
)

ties_merging(
    base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-3b-image_L",
    tuned_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-3b-image_L", 
                 "checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-3b",
                 "checkpoints/Baseline/Audio/finetune/pr_llm/finetune-audio-llama3.2-3b"],
    output_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-3b-image_L/ties_image_video_audio", 
                  "checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-3b/ties_image_video_audio",
                  "checkpoints/Baseline/Audio/finetune/pr_llm/finetune-audio-llama3.2-3b/ties_image_video_audio"],
    reset_thresh=0.2,
    lambda_scale=1.,
    remove_keys=[]
)