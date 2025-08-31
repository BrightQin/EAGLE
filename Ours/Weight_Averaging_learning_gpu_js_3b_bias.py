import torch
from safetensors.torch import load_file, save_file, safe_open
import copy
from collections import OrderedDict
import os
import torch.nn.functional as F
from tqdm import tqdm
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

def optimize_task_vectors_entropy(task_vectors, learning_rate=5e-5, max_iterations=50, temperature=2e-4, eps=1e-8, alpha=0.5, task_weights=None):
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
        task_weights: 任务权重列表，用于控制不同任务的重要性，默认为None（均等权重）
    
    Returns:
        optimized_mtl_vector: 优化后的多任务向量
    """
    # 将输入张量移动到GPU
    task_vectors = task_vectors.to(device)
    n_tasks, vector_dim = task_vectors.shape
    
    # 设置任务权重，如果未提供则使用均等权重
    if task_weights is None:
        task_weights = [1.0] * n_tasks
    elif len(task_weights) != n_tasks:
        raise ValueError(f"任务权重数量({len(task_weights)})与任务数量({n_tasks})不匹配")
    
    # 归一化权重
    task_weights = torch.tensor(task_weights, dtype=torch.float32).to(device)
    task_weights = task_weights / task_weights.sum()  # 归一化到和为1
    
    print(f"任务权重: {task_weights.cpu().numpy()}")
    
    # 初始化多任务向量为任务向量的加权平均值
    weighted_mean = torch.zeros_like(task_vectors[0])
    for i in range(n_tasks):
        weighted_mean += task_weights[i] * task_vectors[i]
    mtl_vector = weighted_mean.clone().detach().requires_grad_(True).to(device)
    
    # 定义优化器
    optimizer = torch.optim.Adam([mtl_vector], lr=learning_rate)
    
    # 计算任务向量的元素级别加权平均幅度
    task_magnitudes = torch.zeros_like(task_vectors[0])
    for i in range(n_tasks):
        task_magnitudes += task_weights[i] * task_vectors[i].abs()
    print(f"任务向量元素加权平均幅度的均值: {task_magnitudes.mean().item()}")
    
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
    
    # 计算综合损失：JS散度 + 元素级别的幅度保留
    def compute_loss(mtl_vec, lambda_js=0.5):  # 添加lambda_js参数来控制JS散度中的权重
        # 计算JS散度损失
        mtl_prob = to_prob_dist(mtl_vec)
        js_loss = torch.tensor(0.0).to(device)
        
        for i in tqdm(range(n_tasks), desc="计算损失"):
            task_prob = task_probs[i]  # 使用预计算的概率分布
            
            # 分块计算JS散度
            chunk_size = min(100000000, mtl_prob.shape[0])
            num_chunks = (mtl_prob.shape[0] + chunk_size - 1) // chunk_size
            
            chunk_js_loss = torch.tensor(0.0).to(device)
            for j in range(num_chunks):
                start_idx = j * chunk_size
                end_idx = min((j + 1) * chunk_size, mtl_prob.shape[0])
                
                mtl_chunk = mtl_prob[start_idx:end_idx]
                task_chunk = task_prob[start_idx:end_idx]
                
                # 计算中间分布M = λP + (1-λ)Q
                M = lambda_js * task_chunk + (1 - lambda_js) * mtl_chunk
                
                # 计算JS散度: λKL(P||M) + (1-λ)KL(Q||M)
                kl_task_m = F.kl_div(torch.log(M + eps), task_chunk, reduction='sum')
                kl_mtl_m = F.kl_div(torch.log(M + eps), mtl_chunk, reduction='sum')
                chunk_js = lambda_js * kl_task_m + (1 - lambda_js) * kl_mtl_m
                chunk_js_loss += chunk_js
            
            # 应用任务权重
            js_loss += task_weights[i] * chunk_js_loss
        
        # 计算元素级别的幅度损失 - 使mtl向量的每个元素幅度接近任务向量对应元素的加权平均幅度
        element_mag_loss = torch.mean(torch.abs(mtl_vec.abs() - task_magnitudes))
        element_mag_loss = element_mag_loss * 1e6
        print(f"mtl平均幅度: {mtl_vec.abs().mean().item()}, 任务向量加权平均幅度: {task_magnitudes.mean().item()}, 元素幅度损失: {element_mag_loss.item()}")

        # 综合损失 - JS散度和元素幅度损失的加权和
        total_loss = alpha * js_loss + (1 - alpha) * element_mag_loss
        
        return total_loss, js_loss.item(), element_mag_loss.item()
    
    # 优化循环
    best_loss = float('inf')
    best_mtl_vector = None
    
    for iteration in range(max_iterations):
        optimizer.zero_grad()
        
        # 计算损失
        loss, js_loss_val, mag_loss_val = compute_loss(mtl_vector)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 检查收敛
        current_loss = loss.item()
        if (iteration + 1) % 1 == 0:
            mtl_mean_mag = mtl_vector.abs().mean().item()
            print(f"迭代 {iteration + 1}/{max_iterations}, 总损失: {current_loss:.6f}, JS损失: {js_loss_val:.6f}, 元素幅度损失: {mag_loss_val:.6f}, MTL平均幅度: {mtl_mean_mag:.6f}")
        
        # 保存最佳结果
        if current_loss < best_loss:
            best_loss = current_loss
            best_mtl_vector = mtl_vector.clone().detach()
                
    # 返回优化后的多任务向量，并移回CPU
    return best_mtl_vector.cpu() if best_mtl_vector is not None else mtl_vector.detach().cpu()

def tall_tier_merging(base_model_path, tuned_paths, output_prefix=None, 
                output_paths=None, lambda_scale=0.4, remove_keys=[], task_weights=None):   
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
        current_chunk = task_vectors[:, start_idx:end_idx]
        current_mtl_vector = optimize_task_vectors_entropy(current_chunk, task_weights=task_weights)
        mtl_vectors.append(current_mtl_vector)
    mtl_vector = torch.cat(mtl_vectors)
    reconstructed_vec = base_vec + mtl_vector * lambda_scale
    reconstructed_sd = vector_to_state_dict(reconstructed_vec, {k: base_sd[k] for k in common_keys}, [])
    for i, (tuned_path, tuned_sd, tuned_files) in enumerate(zip(tuned_paths, tuned_sds, tuned_files_list)):
        if output_paths and i < len(output_paths):
            output_dir = output_paths[i]
        else:
            model_name = os.path.basename(tuned_path).split(".")[0]
            output_dir = f"{output_prefix}_{model_name}" if output_prefix else f"merged_{model_name}"
        merged_sd = {k: tuned_sd[k].clone() for k in tuned_sd.keys()}
        for k in reconstructed_sd:
            merged_sd[k] = reconstructed_sd[k]
        for k in remove_keys:
            if k in base_sd:
                merged_sd[k] = base_sd[k].clone()
        save_model_files(merged_sd, output_dir, tuned_metadata[i], tuned_files)
        print(f"模型 {i+1} 已保存至 {output_dir}")

# 使用示例
# 设置任务权重，第一个任务（图像任务）权重更高
task_weights = [0.98, 0.02]  # 70%权重给第一个任务，30%权重给第二个任务

tall_tier_merging(
    base_model_path="/home1/hxl/Huawei/qbs_eagle/merge/pr/finetune-eagle-x1-llama3.2-3b-image_L",
    tuned_paths=["/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle", 
                 "/home1/hxl/Huawei/qbs_eagle/checkpoints/Audio/finetune-audio-qwen2audioenc-llama3.2-3b-onellm_qa_0611(final)"],
    output_paths=["/home1/hxl/Huawei/qbs_eagle/merged_model/0.98_0.02/image", 
                  "/home1/hxl/Huawei/qbs_eagle/merged_model/0.98_0.02/audio",
                  ],
    lambda_scale=1.0,
    remove_keys=[],
    task_weights=task_weights
)

# 设置任务权重，第一个任务（图像任务）权重更高
task_weights = [0.95, 0.05]  # 70%权重给第一个任务，30%权重给第二个任务

tall_tier_merging(
    base_model_path="/home1/hxl/Huawei/qbs_eagle/merge/pr/finetune-eagle-x1-llama3.2-3b-image_L",
    tuned_paths=["/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle", 
                 "/home1/hxl/Huawei/qbs_eagle/checkpoints/Audio/finetune-audio-qwen2audioenc-llama3.2-3b-onellm_qa_0611(final)"],
    output_paths=["/home1/hxl/Huawei/qbs_eagle/merged_model/0.95_0.05/image", 
                  "/home1/hxl/Huawei/qbs_eagle/merged_model/0.95_0.05/audio",
                  ],
    lambda_scale=1.0,
    remove_keys=[],
    task_weights=task_weights
)