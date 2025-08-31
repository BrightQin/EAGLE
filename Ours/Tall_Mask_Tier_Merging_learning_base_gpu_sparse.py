import torch
from safetensors.torch import load_file, save_file, safe_open
import copy
from collections import OrderedDict
import os
import torch.nn.functional as F
from tqdm import tqdm
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

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

def optimize_task_vectors_entropy(task_vectors, learning_rate=5e-5, max_iterations=20, 
                                 tolerance=1e-6, temperature=2e-4, eps=1e-8, alpha=0.5, 
                                 sparsity_weight=1e-5):
    """
    优化任务向量的融合，直接最小化信息熵损失，同时保留向量长度信息，并增加稀疏性约束
    
    Args:
        task_vectors: shape [n_tasks, vector_dim]
        learning_rate: 梯度下降的学习率
        max_iterations: 最大迭代次数
        tolerance: 收敛容差
        temperature: softmax温度参数
        eps: 数值稳定性的小常数
        alpha: 平衡KL散度和长度保留的权重系数
        sparsity_weight: 稀疏性损失的权重
    
    Returns:
        optimized_mtl_vector: 优化后的多任务向量
    """
    # 将输入张量移动到GPU
    task_vectors = task_vectors.to(device)
    n_tasks, vector_dim = task_vectors.shape
    
    # 初始化多任务向量为任务向量的平均值
    mtl_vector = task_vectors.mean(dim=0).clone().detach().requires_grad_(True).to(device)
    
    # 定义优化器
    optimizer = torch.optim.Adam([mtl_vector], lr=learning_rate)
    
    # 计算任务向量的长度（范数）
    task_norms = torch.norm(task_vectors, dim=1)
    avg_task_norm = task_norms.mean()
    print(f"任务向量的平均范数: {avg_task_norm.item()}")
    
    # 将任务向量转换为概率分布，同时保留长度信息
    def to_prob_dist(vector, temp=temperature):
        # 对向量进行分块处理，避免整体softmax
        chunk_size = min(100000000, vector.shape[0])  # 选择合适的分块大小
        num_chunks = (vector.shape[0] + chunk_size - 1) // chunk_size
        vector = vector.to(device)
        probs = []
        for i in tqdm(range(num_chunks), desc="计算概率分布"):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, vector.shape[0])
            chunk = vector[start_idx:end_idx]
            # 对每个分块应用softmax
            chunk_prob = F.softmax(chunk.abs() / temp, dim=0)
            probs.append(chunk_prob)

        return torch.cat(probs)
    
    # 预计算所有任务向量的概率分布
    print("预计算任务向量的概率分布...")
    task_probs = [to_prob_dist(task_vectors[i]) for i in range(n_tasks)]
    
    def compute_sparsity_loss(vector, eps=1e-8, method='l1_l2'):
        if method == 'l1_l2':
            # L1/L2比率 (您当前使用的方法)
            l1_norm = torch.norm(vector, p=1)
            l2_norm = torch.norm(vector, p=2)
            return l1_norm / (l2_norm + eps)
        
        elif method == 'l1':
            # 简单的L1正则化
            return torch.norm(vector, p=1)
        
        elif method == 'log_sum':
            # Log-sum惩罚，比L1更强的稀疏性促进
            return torch.sum(torch.log(torch.abs(vector) + eps))
        
        elif method == 'hoyer':
            # Hoyer指标 (另一种L1/L2变体)
            n = vector.numel()
            l1_norm = torch.norm(vector, p=1)
            l2_norm = torch.norm(vector, p=2)
            return (math.sqrt(n) - l1_norm / (l2_norm + eps)) / (math.sqrt(n) - 1 + eps)
        
        elif method == 'gini':
            # 基尼系数 (经济学中衡量不平等的指标，用于稀疏性也很有效)
            v_sorted = torch.sort(torch.abs(vector))[0]
            n = v_sorted.shape[0]
            indices = torch.arange(1, n+1, device=vector.device)
            return 2 * torch.sum(indices * v_sorted) / (n * torch.sum(v_sorted)) - (n+1) / n
        
        elif method == 'hard_threshold':
            # 硬阈值惩罚 (直接计算非零元素的数量)
            return torch.sum(torch.abs(vector) > eps).float() / vector.numel()
        
        elif method == 'smooth_l0':
            # 平滑的L0近似 (使用sigmoid函数)
            alpha = 10.0  # 控制近似的陡峭程度
            return torch.mean(2 * torch.sigmoid(alpha * torch.abs(vector)) - 1)
        
        elif method == 'elastic_net':
            # 弹性网络 (L1和L2的组合)
            alpha = 0.5  # L1和L2的权重比例
            l1_part = torch.norm(vector, p=1)
            l2_part = torch.sum(vector**2)
            return alpha * l1_part + (1-alpha) * l2_part
    
    # 计算综合损失：KL散度 + 范数保留 + 稀疏性
    def compute_loss(mtl_vec):
        # 计算KL散度损失
        mtl_prob = to_prob_dist(mtl_vec)
        kl_loss = torch.tensor(0.0).to(device)
        
        for i in tqdm(range(n_tasks), desc="计算损失"):
            task_prob = task_probs[i]  # 使用预计算的概率分布
            
            # 分块计算KL散度
            chunk_size = min(100000000, mtl_prob.shape[0])
            num_chunks = (mtl_prob.shape[0] + chunk_size - 1) // chunk_size
            
            chunk_kl_loss = torch.tensor(0.0).to(device)
            for j in range(num_chunks):
                start_idx = j * chunk_size
                end_idx = min((j + 1) * chunk_size, mtl_prob.shape[0])
                
                mtl_chunk = mtl_prob[start_idx:end_idx]
                task_chunk = task_prob[start_idx:end_idx]
                
                # 计算KL散度: KL(task_prob || mtl_prob)
                chunk_kl = F.kl_div(torch.log(mtl_chunk + eps), task_chunk, reduction='sum')
                chunk_kl_loss += chunk_kl
            
            kl_loss += chunk_kl_loss
        
        kl_loss = kl_loss / n_tasks
        
        # 计算范数损失 - 使mtl向量的范数接近任务向量范数的平均值
        mtl_norm = torch.norm(mtl_vec)
        norm_loss = torch.abs(mtl_norm - avg_task_norm)
        print(f"mtl_norm: {mtl_norm.item()}, avg_task_norm: {avg_task_norm.item()}, norm_loss: {norm_loss.item()}")
        norm_loss = norm_loss * 10
        
        # 计算稀疏性损失
        sparse_loss = compute_sparsity_loss(mtl_vec)
        sparse_loss = sparsity_weight * sparse_loss
        
        # 综合损失 - KL散度、范数损失和稀疏性损失的加权和
        total_loss = alpha * kl_loss + (1 - alpha) * norm_loss + sparse_loss
        
        return total_loss, kl_loss.item(), norm_loss.item(), sparse_loss.item()
    
    # 优化循环
    best_loss = float('inf')
    with torch.no_grad():
        # 使用相同的重构方法初始化向量，而不是使用硬性的稀疏度阈值
        # 首先计算一个初始的tall_masks
        initial_task_vectors = task_vectors.mean(dim=0, keepdim=True)
        tall_masks = compute_entropy_based_mask(initial_task_vectors[0], mtl_vector, tall_lambda=1.0)
        # 使用重构公式更新初始向量
        mtl_vector.data = tall_masks * mtl_vector.data

    for iteration in range(max_iterations):
        optimizer.zero_grad()
        
        # 计算损失
        loss, kl_loss_val, norm_loss_val, sparse_loss_val = compute_loss(mtl_vector)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 检查收敛
        current_loss = loss.item()
        if (iteration + 1) % 1 == 0:
            mtl_norm = torch.norm(mtl_vector).item()
            non_zero_ratio = (mtl_vector.abs() > 1e-6).float().mean().item()
            print(f"迭代 {iteration + 1}/{max_iterations}, 总损失: {current_loss:.6f}, KL损失: {kl_loss_val:.6f}, 范数损失: {norm_loss_val:.6f}, 稀疏损失: {sparse_loss_val:.6f}, MTL范数: {mtl_norm:.6f}, 非零元素比例: {non_zero_ratio:.4f}")
        
        # 保存最佳结果
        if current_loss < best_loss:
            best_loss = current_loss
            best_mtl_vector = mtl_vector.clone().detach()
    
    # 返回优化后的多任务向量，并移回CPU
    return best_mtl_vector.cpu()

def compute_entropy_based_mask(task_vector, mtl_vector, tall_lambda=1.0):
    """
    基于信息熵计算最优mask
    Args:
        task_vector: 单个任务向量
        mtl_vector: 多任务向量
        tall_lambda: TALL lambda参数
        temperature: softmax温度参数
    Returns:
        优化后的mask
    """
    diff = mtl_vector - task_vector
    tall_condition = (task_vector.abs() >= diff.abs() * tall_lambda)
    entropy_mask = tall_condition
    
    return entropy_mask

def optimize_masks(task_vectors, mtl_vector, tall_lambda=1.0):
    """
    优化所有任务的masks
    """
    n_tasks = task_vectors.shape[0]
    masks = []
    
    # 计算每个任务的mask
    for i in range(n_tasks):
        mask = compute_entropy_based_mask(task_vectors[i], mtl_vector, tall_lambda)
        masks.append(mask)
    
    # 合并所有mask
    final_mask = masks[0]
    for mask in masks[1:]:
        final_mask = torch.maximum(final_mask, mask)
    
    return final_mask

def tall_tier_merging(base_model_path, tuned_paths, output_prefix=None, 
                 output_paths=None, lambda_scale=0.4, 
                 tall_lambda=1.0, remove_keys=[], sparsity_weight=0.1):
    
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
    common_keys = list(common_keys)
    
    # 对共有参数应用合并
    base_vec = state_dict_to_vector({k: base_sd[k] for k in common_keys}, [])
    tuned_vecs = torch.stack([state_dict_to_vector({k: sd[k] for k in common_keys}, []) for sd in tuned_sds])
    
    # 计算任务向量 (τₜ)
    task_vectors = tuned_vecs - base_vec.unsqueeze(0)

    # 分块处理task_vectors
    chunk_size = 400000000
    num_chunks = (task_vectors.shape[1] + chunk_size - 1) // chunk_size
    mtl_vectors = []
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, task_vectors.shape[1])
        print(f"处理第 {chunk_idx + 1}/{num_chunks} 块，范围：{start_idx}:{end_idx}")
        
        # 处理当前块
        current_chunk = task_vectors[:, start_idx:end_idx]
        current_mtl_vector = optimize_task_vectors_entropy(
            current_chunk, 
            sparsity_weight=sparsity_weight, 
        )
        mtl_vectors.append(current_mtl_vector)
    
    # 合并所有处理后的向量
    mtl_vector = torch.cat(mtl_vectors)

    # 使用基于信息熵的方法计算mask
    tall_masks = optimize_masks(task_vectors, mtl_vector, tall_lambda)

    # 用优化后的mask从多任务向量中提取任务特定信息
    reconstructed_vec = base_vec + tall_masks * mtl_vector * lambda_scale
    
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
tall_tier_merging(
    base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors",
    tuned_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors", 
                 "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-1b-image_L/finetune/pr_llm/finetune-video-llama3.2-1b/model.safetensors"],
    output_paths=["checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/merged_image_our_KL_length_sparse_5e-4/model.safetensors", 
                  "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-1b-image_L/finetune/pr_llm/finetune-video-llama3.2-1b/merged_video_our_KL_length_sparse_5e-4/model.safetensors"],
    lambda_scale=1.,
    tall_lambda=1.0,
    remove_keys=[],
    sparsity_weight=5e-4,  # 控制稀疏性损失的权重
)