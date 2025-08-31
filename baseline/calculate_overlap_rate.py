import torch
from safetensors.torch import load_file, save_file, safe_open
import copy
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import os

def load_state_dict_auto(path):
    """
    自动判断文件类型加载state_dict，支持safetensors和pt文件
    """
    ext = os.path.splitext(path)[-1]
    if ext == '.safetensors':
        with safe_open(path, framework="pt") as model_file:
            state_dict = {k: model_file.get_tensor(k) for k in model_file.keys()}
        return state_dict
    elif ext == '.pt':
        obj = torch.load(path, map_location='cpu')
        # 兼容直接保存的state_dict或包含state_dict的dict
        if isinstance(obj, dict):
            if 'state_dict' in obj:
                return obj['state_dict']
            elif 'model' in obj:
                return obj['model']
            else:
                # 直接就是state_dict
                return obj
        else:
            raise ValueError(f"未知的.pt文件结构: {path}")
    else:
        raise ValueError(f"不支持的文件类型: {path}")

def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_dict:
            del shared_dict[key]
    sorted_dict = OrderedDict(sorted(shared_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [t.reshape(-1) for t in sorted_dict.values()]
    )

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

def calculate_overlap_rate(masks):
    """
    计算多个任务向量的重合率
    Args:
        masks: torch.Tensor, shape (num_tasks, dim), 每一行是一个任务向量的mask
    Returns:
        overlap_rate: float, 重合率（所有任务向量都保留的参数比例）
    """
    # 计算所有任务向量都为True的位置
    all_overlap = torch.all(masks, dim=0)  # 所有任务都保留的位置
    any_overlap = torch.any(masks, dim=0)  # 至少一个任务保留的位置
    
    # 重合率 = 全部重合的参数数量 / 至少一个任务保留的参数数量
    if torch.sum(any_overlap) == 0:
        return 0.0
    
    overlap_rate = torch.sum(all_overlap).float() / torch.sum(any_overlap).float()
    return overlap_rate.item()

def calculate_pairwise_overlap_rate(masks):
    """
    计算两两任务向量之间的平均重合率
    Args:
        masks: torch.Tensor, shape (num_tasks, dim)
    Returns:
        avg_pairwise_overlap: float, 平均两两重合率
    """
    num_tasks = masks.shape[0]
    if num_tasks < 2:
        return 1.0
    
    total_overlap = 0.0
    count = 0
    
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            mask_i = masks[i]
            mask_j = masks[j]
            
            # 计算两个mask的交集和并集
            intersection = torch.sum(mask_i & mask_j).float()
            union = torch.sum(mask_i | mask_j).float()
            
            if union > 0:
                overlap = intersection / union
                total_overlap += overlap
                count += 1
    
    return total_overlap / count if count > 0 else 0.0

def analyze_overlap_rates(base_model_path, tuned_paths, remove_keys=[], 
                         save_plot=True, plot_path="overlap_rates.png"):
    """
    分析不同reset_thresh值下的任务向量重合率
    """
    print("正在加载模型...")
    
    # 加载所有模型
    base_sd = load_state_dict_auto(base_model_path)
    
    tuned_sds = []
    for p in tuned_paths:
        tuned_sd = load_state_dict_auto(p)
        tuned_sds.append(tuned_sd)
    
    # 查找共有参数
    common_keys = set(base_sd.keys())
    for sd in tuned_sds:
        common_keys &= set(sd.keys())
    
    # 移除指定要排除的键
    common_keys = common_keys - set(remove_keys)
    
    # 移除包含vision_tower或mm_projector的键
    common_keys = {k for k in common_keys if "vision_tower" not in k and "mm_projector" not in k}
    
    print(f"共有参数数量: {len(common_keys)}")
    
    # 转换为向量
    base_vec = state_dict_to_vector({k: base_sd[k] for k in common_keys}, [])
    tuned_vecs = torch.stack([state_dict_to_vector({k: sd[k] for k in common_keys}, []) for sd in tuned_sds])
    
    # 计算任务向量
    task_vectors = tuned_vecs - base_vec.unsqueeze(0)
    
    print(f"任务向量形状: {task_vectors.shape}")
    print(f"基模型向量维度: {base_vec.shape}")
    
    # 计算不同reset_thresh值下的重合率
    reset_thresh_values = np.arange(0.05, 1.0, 0.05)  # 0.0到1.0，步长0.05
    all_overlap_rates = []
    pairwise_overlap_rates = []
    retained_params_ratio = []
    
    print("\n开始计算重合率...")
    for thresh in reset_thresh_values:
        print(f"计算 reset_thresh = {thresh:.2f}")
        
        # 获取mask
        _, masks = topk_values_mask(task_vectors, K=thresh, return_mask=True)
        
        # 计算全重合率
        all_overlap = calculate_overlap_rate(masks)
        all_overlap_rates.append(all_overlap)
        
        # 计算两两重合率
        pairwise_overlap = calculate_pairwise_overlap_rate(masks)
        pairwise_overlap_rates.append(pairwise_overlap)
        
        # 计算保留参数比例
        retained_ratio = torch.sum(torch.any(masks, dim=0)).float() / masks.shape[1]
        retained_params_ratio.append(retained_ratio.item())
        
        print(f"  全重合率: {all_overlap:.4f}")
        print(f"  两两重合率: {pairwise_overlap:.4f}")
        print(f"  保留参数比例: {retained_ratio:.4f}")
    
    # 绘制结果
    if save_plot:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(reset_thresh_values, all_overlap_rates, 'b-o', markersize=4)
        plt.xlabel('Reset Threshold')
        plt.ylabel('全重合率')
        plt.title('全重合率 vs Reset Threshold')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        plt.subplot(1, 3, 2)
        plt.plot(reset_thresh_values, pairwise_overlap_rates, 'r-o', markersize=4)
        plt.xlabel('Reset Threshold')
        plt.ylabel('平均两两重合率')
        plt.title('平均两两重合率 vs Reset Threshold')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        plt.subplot(1, 3, 3)
        plt.plot(reset_thresh_values, retained_params_ratio, 'g-o', markersize=4)
        plt.xlabel('Reset Threshold')
        plt.ylabel('保留参数比例')
        plt.title('保留参数比例 vs Reset Threshold')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\n图表已保存至: {plot_path}")
    
    # 返回结果
    results = {
        'reset_thresh_values': reset_thresh_values,
        'all_overlap_rates': all_overlap_rates,
        'pairwise_overlap_rates': pairwise_overlap_rates,
        'retained_params_ratio': retained_params_ratio
    }
    
    return results

# 使用示例
if __name__ == "__main__":
    # 设置路径
    base_model_path = "checkpoints/Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors"
    tuned_paths = [
        "checkpoints/Baseline/Images/finetune/pr_llm/finetune-eagle-x1-llama3.2-1b-image_L/model.safetensors", 
        "checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-1b-ori-token/model.safetensors",
        "checkpoints/Baseline/Audio/finetune/pr_llm/finetune-audio-llama3.2-1b/model.safetensors"
    ]

    base_model_path="/home/qinbosheng/my_program/Model_Merging/tall_masks/models/checkpoints/ViT-B-32/MNISTVal/nonlinear_zeroshot.pt"
    tuned_paths=["/home/qinbosheng/my_program/Model_Merging/tall_masks/models/checkpoints/ViT-B-32/CIFAR10Val/nonlinear_finetuned.pt", 
                "/home/qinbosheng/my_program/Model_Merging/tall_masks/models/checkpoints/ViT-B-32/CIFAR100Val/nonlinear_finetuned.pt",
                "/home/qinbosheng/my_program/Model_Merging/tall_masks/models/checkpoints/ViT-B-32/STL10Val/nonlinear_finetuned.pt"]
    
    # 分析重合率
    results = analyze_overlap_rates(
        base_model_path=base_model_path,
        tuned_paths=tuned_paths,
        remove_keys=[],
        save_plot=True,
        plot_path="task_vector_overlap_analysis1.png"
    )
    
    # 打印详细结果
    print("\n=== 详细结果 ===")
    print("Reset_Thresh\t全重合率\t两两重合率\t保留参数比例")
    print("-" * 60)
    for i, thresh in enumerate(results['reset_thresh_values']):
        print(f"{thresh:.2f}\t\t{results['all_overlap_rates'][i]:.4f}\t\t{results['pairwise_overlap_rates'][i]:.4f}\t\t{results['retained_params_ratio'][i]:.4f}") 