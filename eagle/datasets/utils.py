import torch

from eagle.constants import IGNORE_INDEX

PADDING = 49407

def get_input_ids_len(input: torch.tensor):
    return torch.sum(input != PADDING)

def make_label(input: torch.tensor, target_lens):
    cur_idx = target_lens[0]
    input[0][:cur_idx] = IGNORE_INDEX
    for i, tokenized_len in enumerate(target_lens):
        if i % 2 == 1:
            input[0][target_lens[i-1] + 2:tokenized_len] = IGNORE_INDEX
    return input