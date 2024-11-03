import torch
import torch.nn as nn
import re

# Copied from CuMo
from typing import List, Optional
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack, unpack

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, fpn_input_dim=[], **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()
    
    # Modified from CuMo
    moe_match = re.match(r'^smoe_mlp(\d+)x$', projector_type)
    if moe_match:
        moe_depth = int(moe_match.group(1))
        return MLPMoE(
            num_experts=config.num_experts, 
            num_selected=config.num_selected, 
            mm_channels=config.mm_hidden_size, 
            channels=config.hidden_size, 
            num_layers=config.num_layers,
            depth=moe_depth, 
            dropout=config.dropout
        )

    raise ValueError(f'Unknown projector type: {projector_type}')

# BEGIN
def build_mlp(mlp_depth, input_dim, output_dim):
    modules = [nn.Linear(input_dim, output_dim)]
    for _ in range(1, mlp_depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_dim, output_dim))
    return nn.Sequential(*modules)

def build_audio_projector(config, **kwargs):
    projector_type = getattr(config, 'mm_audio_projector_type', 'linear')
    if projector_type == 'linear':
        return nn.Linear(config.mm_audio_hidden_size, config.hidden_size)
    
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        return build_mlp(
            mlp_depth=mlp_depth,
            input_dim=config.mm_audio_hidden_size,
            output_dim=config.hidden_size
        )
    
    if projector_type == 'identity':
        return IdentityMap()
    # FUTURE: Add MoE projection here
# END

# Copied from CuMo
class MLPMoE(nn.Module):
    def __init__(
            self, 
            num_experts, 
            num_selected, 
            mm_channels, 
            channels, 
            num_layers,
            depth=1, 
            dropout=False
        ):
        super().__init__()
        self.num_experts = num_experts
        self.num_selected = num_selected
        self.mm_channels = mm_channels
        self.channels = channels

        self.gate = nn.Linear(mm_channels, num_experts, bias=False)
        self.num_selected = num_selected
        self.num_experts = num_experts
        expert = [nn.Linear(mm_channels, channels)]
        for _ in range(1, depth):
            expert.append(nn.GELU())
            expert.append(nn.Linear(channels, channels))
        self.experts = nn.ModuleList([nn.Sequential(*expert) for _ in range(num_experts)])

    def forward(self, x_img):
        gate_logits = self.gate(x_img)

        router_z_loss = torch.logsumexp(gate_logits, dim = -1)
        router_z_loss = torch.square(router_z_loss)            
        router_z_loss = router_z_loss.mean()
        
        gate_softmax = F.softmax(gate_logits, dim=-1, dtype=torch.float).to(x_img.dtype)

        density_1_proxy = reduce(gate_softmax, '... n e -> ... e', 'mean')

        weights, selected_experts = torch.topk(gate_softmax, self.num_selected)

        one_hot_gate_indices = F.one_hot(rearrange(selected_experts, '... k -> k ...'), self.num_experts).float()[0]
        density_1 = reduce(one_hot_gate_indices, '... n e -> ... e', 'mean')
        balance_loss = (density_1_proxy * density_1).mean() * float(self.num_experts ** 2)

        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x_img.dtype)
        
        results = torch.zeros((x_img.shape[0], x_img.shape[1], self.channels)).to(x_img.device, x_img.dtype)

        for b in range(x_img.shape[0]):
            for i, expert in enumerate(self.experts):
                token_idx, nth_expert = torch.where(selected_experts[b] == i)
                results[b][token_idx] += weights[b][token_idx, nth_expert, None] * expert(x_img[b][token_idx])
        return results, balance_loss, router_z_loss

    @property
    def config(self):
        return {"mm_projector_type": 'smoe_mlp'}