import torch
from safetensors.torch import load_file, save_file, safe_open
import copy
from collections import OrderedDict
import os
import torch.nn.functional as F
from tqdm import tqdm

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

def optimize_task_vectors_entropy(task_vectors, learning_rate=5e-5, max_iterations=50, temperature=2e-4, eps=1e-8, alpha=0.5):
    """
    优化任务向量的融合，直接最小化信息熵损失，同时保留向量元素级别的长度信息
    
    Args:
        task_vectors: shape [n_tasks, vector_dim]
        learning_rate: 梯度下降的学习率
        max_iterations: 最大迭代次数
        tolerance: 收敛容差
        temperature: softmax温度参数
        eps: 数值稳定性的小常数
        alpha: 平衡KL散度和长度保留的权重系数
    
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
    
    # 计算任务向量的元素级别平均幅度
    task_magnitudes = task_vectors.abs().mean(dim=0)  # 每个元素位置的平均幅度
    print(f"任务向量元素平均幅度的均值: {task_magnitudes.mean().item()}")
    
    # 将任务向量转换为概率分布，同时保留长度信息
    def to_prob_dist(vector, temp=temperature):
        # 对向量进行分块处理，避免整体softmax
        chunk_size = min(50000000, vector.shape[0])  # 选择合适的分块大小
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
    
    # 计算综合损失：Wasserstein距离 + 元素级别的幅度保留
    def compute_loss(mtl_vec):
        # 计算Wasserstein距离损失
        mtl_prob = to_prob_dist(mtl_vec)
        wasserstein_loss = torch.tensor(0.0).to(device)
        
        for i in tqdm(range(n_tasks), desc="计算损失"):
            task_prob = task_probs[i]  # 使用预计算的概率分布
            
            # 分块计算Wasserstein距离
            chunk_size = min(100000000, mtl_prob.shape[0])
            num_chunks = (mtl_prob.shape[0] + chunk_size - 1) // chunk_size
            
            chunk_wasserstein_loss = torch.tensor(0.0).to(device)
            for j in range(num_chunks):
                start_idx = j * chunk_size
                end_idx = min((j + 1) * chunk_size, mtl_prob.shape[0])
                
                mtl_chunk = mtl_prob[start_idx:end_idx]
                task_chunk = task_prob[start_idx:end_idx]
                
                # 计算CDF
                cdf_mtl = torch.cumsum(mtl_chunk, dim=0)
                cdf_task = torch.cumsum(task_chunk, dim=0)
                chunk_wasserstein = torch.sum(torch.abs(cdf_mtl - cdf_task))
                chunk_wasserstein_loss += chunk_wasserstein
            
            wasserstein_loss += chunk_wasserstein_loss
        
        wasserstein_loss = wasserstein_loss / n_tasks
        wasserstein_loss = wasserstein_loss * 1e-6
        
        # 计算元素级别的幅度损失
        element_mag_loss = torch.mean(torch.abs(mtl_vec.abs() - task_magnitudes))
        element_mag_loss = element_mag_loss * 1e6
        print(f"mtl平均幅度: {mtl_vec.abs().mean().item()}, 任务向量平均幅度: {task_magnitudes.mean().item()}, 元素幅度损失: {element_mag_loss.item()}")

        # 综合损失 - Wasserstein距离和元素幅度损失的加权和
        total_loss = alpha * wasserstein_loss + (1 - alpha) * element_mag_loss
        
        return total_loss, wasserstein_loss.item(), element_mag_loss.item()
    
    # 优化循环
    best_loss = float('inf')
    best_mtl_vector = None
    
    for iteration in range(max_iterations):
        optimizer.zero_grad()
        
        # 计算损失
        loss, wasserstein_loss_val, mag_loss_val = compute_loss(mtl_vector)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 检查收敛
        current_loss = loss.item()
        if (iteration + 1) % 1 == 0:
            mtl_mean_mag = mtl_vector.abs().mean().item()
            print(f"迭代 {iteration + 1}/{max_iterations}, 总损失: {current_loss:.6f}, Wasserstein损失: {wasserstein_loss_val:.6f}, 元素幅度损失: {mag_loss_val:.6f}, MTL平均幅度: {mtl_mean_mag:.6f}")
        
        # 保存最佳结果
        if current_loss < best_loss:
            best_loss = current_loss
            best_mtl_vector = mtl_vector.clone().detach()
                
    # 返回优化后的多任务向量，并移回CPU
    return best_mtl_vector.cpu() if best_mtl_vector is not None else mtl_vector.detach().cpu()

def tall_tier_merging(base_model_path, tuned_paths, output_prefix=None, 
                output_paths=None, lambda_scale=0.4, remove_keys=[]):   
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
    chunk_size = 100000000
    num_chunks = (task_vectors.shape[1] + chunk_size - 1) // chunk_size
    mtl_vectors = []
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, task_vectors.shape[1])
        print(f"处理第 {chunk_idx + 1}/{num_chunks} 块，范围：{start_idx}:{end_idx}")
        
        # 处理当前块
        current_chunk = task_vectors[:, start_idx:end_idx]
        current_mtl_vector = optimize_task_vectors_entropy(current_chunk)
        mtl_vectors.append(current_mtl_vector)
    
    # 合并所有处理后的向量
    mtl_vector = torch.cat(mtl_vectors)

    # 用优化后的mask从多任务向量中提取任务特定信息
    reconstructed_vec = base_vec + mtl_vector * lambda_scale
    
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
        print(f"模型 {i+1} 已保存至 {output_path}")

# 使用示例
for lambda_scale in [round(x, 1) for x in torch.arange(1.0, 1.01, 0.1).tolist()]:
    print(f"当前 lambda_scale: {lambda_scale}")
    output_paths=[
        f"checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/weight_averaging_image_video_increamental_js_length_20_80_loss_{lambda_scale}/model.safetensors",
        f"checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-1b-image_L/finetune/pr_llm/finetune-video-llama3.2-1b/weight_averaging_image_video_increamental_js_length_20_80_loss_{lambda_scale}/model.safetensors"
    ]
    tall_tier_merging(
        base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors",
        tuned_paths=[
            "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors", 
            "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-1b-image_L/finetune/pr_llm/finetune-video-llama3.2-1b/model.safetensors"
        ],
        output_paths=output_paths,
        lambda_scale=lambda_scale,
        remove_keys=[]
    )

# for lambda_scale in [round(x, 1) for x in torch.arange(0.7, 1.01, 0.1).tolist()]:
#     print(f"当前 lambda_scale: {lambda_scale}")
#     output_paths=[
#         f"checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/weight_averaging_image_video_js_{lambda_scale}/model.safetensors",
#         f"checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-1b-ori-token/weight_averaging_image_video_js_{lambda_scale}/model.safetensors"
#     ]
#     tall_tier_merging(
#         base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors",
#         tuned_paths=[
#             "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors", 
#             "checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-1b-ori-token/model.safetensors"
#         ],
#         output_paths=output_paths,
#         lambda_scale=lambda_scale,
#         remove_keys=[]
#     )


# for lambda_scale in [round(x, 1) for x in torch.arange(0.7, 1.01, 0.1).tolist()]:
#     print(f"当前 lambda_scale: {lambda_scale}")
#     output_paths=[
#         f"checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/weight_averaging_image_video_audio_increamental_js_{lambda_scale}/model.safetensors",
#         f"checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-1b-image_L/finetune/pr_llm/finetune-video-llama3.2-1b/weight_averaging_image_video_audio_increamental_js_{lambda_scale}/model.safetensors",
#         f"checkpoints/Incremental/Audio/finetune-video-llama3.2-1b/finetune/pr_llm/finetune-audio-llama3.2-1b/weight_averaging_image_video_audio_increamental_js_{lambda_scale}/model.safetensors"
#     ]
#     tall_tier_merging(
#         base_model_path="checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors",
#         tuned_paths=[
#             "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors", 
#             "checkpoints/Incremental/Video/finetune-eagle-x1-llama3.2-1b-image_L/finetune/pr_llm/finetune-video-llama3.2-1b/model.safetensors",
#             "checkpoints/Incremental/Audio/finetune-video-llama3.2-1b/finetune/pr_llm/finetune-audio-llama3.2-1b/model.safetensors"
#         ],
#         output_paths=output_paths,
#         lambda_scale=lambda_scale,
#         remove_keys=[]
#     )
    