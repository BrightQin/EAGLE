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

def tall_tier_merging(base_model_path, tuned_paths, output_prefix=None, 
                 output_paths=None, lambda_scale=0.4, remove_keys=[], noise_scale=0.3):
    
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
    
    # 对共有参数应用合并
    base_vec = state_dict_to_vector({k: base_sd[k] for k in common_keys}, [])
    tuned_vecs = torch.stack([state_dict_to_vector({k: sd[k] for k in common_keys}, []) for sd in tuned_sds])
    
    # 计算任务向量 (τₜ)
    task_vectors = tuned_vecs - base_vec.unsqueeze(0)

    # 计算多任务向量 (τMTL) - 所有任务向量的平均
    mtl_vector = task_vectors.mean(dim=0)
    
    # 找出绝对值最大的前 20% 索引
    abs_mtl = mtl_vector.abs()
    k = int(len(mtl_vector) * 0.2)  # 前 20% 的数量
    _, top_indices = torch.topk(abs_mtl, k)
    
    # 获取这些位置的值并加上噪声
    top_values = mtl_vector[top_indices]
    alpha = noise_scale * mtl_vector.abs().max()
    noise = torch.randn_like(top_values) * alpha
    noised_values = top_values + noise
    
    # 将加噪声后的值放回原向量
    mtl_vector_noised = mtl_vector.clone()
    mtl_vector_noised[top_indices] = noised_values
    
    # 归一化为概率分布（取绝对值后归一化）
    def to_prob_dist(vec):
        vec = vec.abs()
        vec = vec / vec.sum()
        return vec

    p = to_prob_dist(mtl_vector)
    q = to_prob_dist(mtl_vector_noised)
    m = 0.5 * (p + q)
    kl_p_m = torch.sum(p * (torch.log(p + 1e-10) - torch.log(m + 1e-10)))
    kl_q_m = torch.sum(q * (torch.log(q + 1e-10) - torch.log(m + 1e-10)))
    js_divergence = 0.5 * (kl_p_m + kl_q_m)
    print("加噪前后任务向量的JS散度:", js_divergence.item())
    
    # 使用加噪声后的向量进行重构
    reconstructed_vec = base_vec + mtl_vector_noised * lambda_scale
    
    # 将重构的向量转换回state_dict
    reconstructed_sd = vector_to_state_dict(reconstructed_vec, {k: base_sd[k] for k in common_keys}, [])
    
    # 为每个模型重建特定任务信息
    for i, (tuned_path, tuned_sd) in enumerate(zip(tuned_paths, tuned_sds)):
        # 确定输出路径
        if output_paths and i < len(output_paths):
            output_path = output_paths[i]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            model_name = tuned_path.split("/")[-1].split(".")[0]
            output_path = f"{output_prefix}_{model_name}.safetensors" if output_prefix else f"merged_{model_name}.safetensors"
        
        # 准备最终模型
        merged_sd = {k: tuned_sd[k].clone() for k in tuned_sd.keys()}
        
        # 用重构的模型参数覆盖
        for k in reconstructed_sd:
            merged_sd[k] = reconstructed_sd[k]
        
        # 添加被排除的参数（从基础模型）
        for k in remove_keys:
            if k in base_sd:
                merged_sd[k] = base_sd[k].clone()
        
        # 保存模型（包含原始模型的metadata）
        save_file(merged_sd, output_path, metadata=tuned_metadata[i])
        print(f"使用TALL-mask融合的模型 {i+1} 已保存至 {output_path}")

# 使用示例
for noise_scale in [10.0]:
    tall_tier_merging(
        base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors",
        tuned_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors"],
        output_paths=[f"checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/image_top_20_randn_add_noise_{noise_scale}/model.safetensors"],
        lambda_scale=1.,
        remove_keys=[],
        noise_scale=noise_scale
    )

# 0.05 0.0317
# 0.1 0.06152
# 0.2 0.09375
# 0.3 0.108
# 2.0 0.1318

