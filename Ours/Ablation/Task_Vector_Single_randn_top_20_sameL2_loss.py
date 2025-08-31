import torch
from safetensors.torch import load_file, save_file, safe_open
import copy
from collections import OrderedDict
import os
import numpy as np

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

def generate_different_noise_patterns(mtl_vector, top_indices, target_l2_norm):
    """生成多种不同的噪声模式，保持相同的L2范数但产生不同的JS散度"""
    top_values = mtl_vector[top_indices]
    noise_patterns = []
    
    # 获取原始向量的数据类型和设备
    dtype = top_values.dtype
    device = top_values.device
    
    # 模式1: 高斯噪声 (原始模式)
    noise1 = torch.randn_like(top_values, dtype=dtype, device=device)
    noise1 = noise1 / torch.sqrt((noise1 ** 2).sum()) * target_l2_norm  # 归一化到目标L2范数
    noise_patterns.append(("gaussian", noise1))
    
    # 模式2: 均匀噪声
    noise2 = (torch.rand_like(top_values, dtype=dtype, device=device) * 2 - 1)  # [-1, 1]均匀分布
    noise2 = noise2 / torch.sqrt((noise2 ** 2).sum()) * target_l2_norm
    noise_patterns.append(("uniform", noise2))
    
    # 模式3: 稀疏噪声 (只对一半的元素加噪声)
    noise3 = torch.zeros_like(top_values, dtype=dtype, device=device)
    half_idx = len(top_values) // 2
    noise3[:half_idx] = torch.randn(half_idx, dtype=dtype, device=device)
    if torch.sqrt((noise3 ** 2).sum()) > 0:  # 避免除零
        noise3 = noise3 / torch.sqrt((noise3 ** 2).sum()) * target_l2_norm
    noise_patterns.append(("sparse", noise3))
    
    # 模式4: 交替符号噪声
    noise4 = torch.ones_like(top_values, dtype=dtype, device=device)
    noise4[::2] = -1  # 交替正负
    noise4 = noise4 / torch.sqrt((noise4 ** 2).sum()) * target_l2_norm
    noise_patterns.append(("alternating", noise4))
    
    # 模式5: 拉普拉斯噪声 (更尖锐的分布)
    noise5 = torch.distributions.Laplace(0, 1).sample(top_values.shape).to(dtype=dtype, device=device)
    noise5 = noise5 / torch.sqrt((noise5 ** 2).sum()) * target_l2_norm
    noise_patterns.append(("laplace", noise5))
    
    # 模式6: 指数噪声 (单侧指数分布)
    noise6 = torch.distributions.Exponential(1).sample(top_values.shape).to(dtype=dtype, device=device)
    noise6 = torch.where(torch.rand_like(noise6) > 0.5, noise6, -noise6)  # 随机正负
    noise6 = noise6 / torch.sqrt((noise6 ** 2).sum()) * target_l2_norm
    noise_patterns.append(("exponential", noise6))
    
    # 模式7: 正弦波噪声
    indices = torch.arange(len(top_values), dtype=torch.float32, device=device)
    frequency = 2 * np.pi / len(top_values) * 3  # 3个周期
    noise7 = torch.sin(frequency * indices).to(dtype=dtype)
    noise7 = noise7 / torch.sqrt((noise7 ** 2).sum()) * target_l2_norm
    noise_patterns.append(("sinusoidal", noise7))
    
    # 模式8: 分段常数噪声 (阶梯状)
    noise8 = torch.zeros_like(top_values, dtype=dtype, device=device)
    segment_size = len(top_values) // 4
    values = [1, -1, 0.5, -0.5]
    for i, val in enumerate(values):
        start_idx = i * segment_size
        end_idx = min((i + 1) * segment_size, len(top_values))
        noise8[start_idx:end_idx] = val
    noise8 = noise8 / torch.sqrt((noise8 ** 2).sum()) * target_l2_norm
    noise_patterns.append(("piecewise_constant", noise8))
    
    # 模式9: 幂律噪声 (长尾分布)
    noise9 = torch.distributions.Pareto(1, 1).sample(top_values.shape).to(dtype=dtype, device=device)
    noise9 = torch.where(torch.rand_like(noise9) > 0.5, noise9, -noise9)  # 随机正负
    noise9 = torch.clamp(noise9, -10, 10)  # 限制极端值
    noise9 = noise9 / torch.sqrt((noise9 ** 2).sum()) * target_l2_norm
    noise_patterns.append(("power_law", noise9))
    
    # 模式10: 三角波噪声
    indices = torch.arange(len(top_values), dtype=torch.float32, device=device)
    period = len(top_values) / 3  # 3个周期
    triangle_phase = (indices % period) / period * 4 - 2  # [-2, 2]
    noise10 = torch.where(triangle_phase <= 0, triangle_phase + 2, 2 - triangle_phase).to(dtype=dtype)
    noise10 = noise10 / torch.sqrt((noise10 ** 2).sum()) * target_l2_norm
    noise_patterns.append(("triangle_wave", noise10))
    
    # 模式11: 双峰分布噪声 (混合高斯)
    half = len(top_values) // 2
    noise11_part1 = torch.randn(half, dtype=dtype, device=device) + 2  # 正偏移的高斯
    noise11_part2 = torch.randn(len(top_values) - half, dtype=dtype, device=device) - 2  # 负偏移的高斯
    noise11 = torch.cat([noise11_part1, noise11_part2])
    noise11 = noise11[torch.randperm(len(noise11), device=device)]  # 随机打乱
    noise11 = noise11 / torch.sqrt((noise11 ** 2).sum()) * target_l2_norm
    noise_patterns.append(("bimodal", noise11))
    
    # 模式12: 随机游走噪声 (累积随机步长)
    steps = torch.randn(len(top_values), dtype=dtype, device=device) * 0.1
    noise12 = torch.cumsum(steps, dim=0)
    noise12 = noise12 - noise12.mean()  # 去中心化
    if torch.sqrt((noise12 ** 2).sum()) > 0:  # 避免除零
        noise12 = noise12 / torch.sqrt((noise12 ** 2).sum()) * target_l2_norm
    noise_patterns.append(("random_walk", noise12))
    
    return noise_patterns

def tall_tier_merging_multi_noise(base_model_path, tuned_paths, output_base_dir, 
                                 lambda_scale=0.4, remove_keys=[], noise_scale=0.3):
    
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
    
    # 计算目标L2范数
    top_values = mtl_vector[top_indices]
    alpha = noise_scale * mtl_vector.abs().max()
    reference_noise = torch.randn_like(top_values) * alpha
    target_l2_norm = torch.sqrt((reference_noise ** 2).sum())
    
    print(f"目标L2范数: {target_l2_norm.item():.6f}")
    
    # 生成4种不同的噪声模式
    noise_patterns = generate_different_noise_patterns(mtl_vector, top_indices, target_l2_norm)
    
    # 归一化为概率分布（取绝对值后归一化）
    def to_prob_dist(vec):
        vec = vec.abs()
        vec = vec / vec.sum()
        return vec

    # 为每种噪声模式生成模型
    for pattern_idx, (pattern_name, noise) in enumerate(noise_patterns):
        print(f"\n处理噪声模式 {pattern_idx+1}: {pattern_name}")
        
        # 将加噪声后的值放回原向量
        mtl_vector_noised = mtl_vector.clone()
        mtl_vector_noised[top_indices] = top_values + noise
        
        # 验证L2范数
        actual_l2_norm = torch.sqrt((noise ** 2).sum())
        print(f"实际L2范数: {actual_l2_norm.item():.6f}")
        
        # 计算JS散度
        p = to_prob_dist(mtl_vector)
        q = to_prob_dist(mtl_vector_noised)
        m = 0.5 * (p + q)
        kl_p_m = torch.sum(p * (torch.log(p + 1e-10) - torch.log(m + 1e-10)))
        kl_q_m = torch.sum(q * (torch.log(q + 1e-10) - torch.log(m + 1e-10)))
        js_divergence = 0.5 * (kl_p_m + kl_q_m)
        print(f"JS散度: {js_divergence.item():.6f}")
        
        # 使用加噪声后的向量进行重构
        reconstructed_vec = base_vec + mtl_vector_noised * lambda_scale
        
        # 将重构的向量转换回state_dict
        reconstructed_sd = vector_to_state_dict(reconstructed_vec, {k: base_sd[k] for k in common_keys}, [])
        
        # 为每个模型重建特定任务信息
        for i, (tuned_path, tuned_sd) in enumerate(zip(tuned_paths, tuned_sds)):
            # 创建输出目录
            output_dir = f"{output_base_dir}/noise_pattern_{pattern_idx+1}_{pattern_name}"
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/model.safetensors"
            
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
            print(f"模型已保存至: {output_path}")

# 使用示例：生成4个具有相同L2 Loss但不同JS散度的模型
for noise_scale in [0.5]:
    print(f"处理噪声模式 {noise_scale}")
    tall_tier_merging_multi_noise(
        base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors",
        tuned_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors"],
        output_base_dir=f"checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/same_l2_diff_js_{noise_scale}",
        lambda_scale=1.,
        remove_keys=[],
        noise_scale=noise_scale
    )