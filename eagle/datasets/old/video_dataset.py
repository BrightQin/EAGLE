import os
import json
from dataclasses import dataclass
from typing import Dict, Sequence, Any

import pandas

import torch
from torch.utils.data import Dataset

from ...model.multimodal_encoder.video_models.languagebind_video import LanguageBindVideo
from ...model.multimodal_encoder.video_models.processing_video import LanguageBindVideoProcessor
from ...model.multimodal_encoder.video_models.tokenization_video import LanguageBindVideoTokenizer
from ...model.multimodal_encoder.video_models.configuration_video import LanguageBindVideoConfig

from ... import conversation as conversation_lib
from eagle.constants import IGNORE_INDEX
from ..utils import get_input_ids_len, make_label

class VideoDataset(Dataset):
    def __init__(
        self,
        video_text_path: str | os.PathLike,
        video_path: str | os.PathLike,
        video_pretrain_path: str | os.PathLike,
        model_max_length: int
    ):
        super().__init__()
        if 'LanguageBind_Video_FT' in video_pretrain_path:
            self.tokenizer = LanguageBindVideoTokenizer.from_pretrained(video_pretrain_path)
            self.config = LanguageBindVideoConfig.from_pretrained(video_pretrain_path)
            self.video_process = LanguageBindVideoProcessor(self.config, self.tokenizer)
        else:
            raise ValueError('Unknown video tower:', video_pretrain_path)
        if 'videochatgpt_tune' in video_text_path:
            self.dataset_name = 'videochatgpt_tune'
            with open(video_text_path, 'r') as file:
                self.data_text = json.load(file)
        else:
            raise ValueError('Unknown dataset:', video_text_path)
        self.video_path = video_path
        self.tokenizer_max_length = model_max_length

    def __len__(self):
        if self.dataset_name == 'videochatgpt_tune':
            return len(self.data_text)
        else:
            raise NotImplementedError
        
    def __getitem__(self, index):
        if self.dataset_name == 'videochatgpt_tune':
            video_path = os.path.join(self.video_path, self.data_text[index]['video'])

            conv = conversation_lib.default_conversation.copy()
            role_human = conv.roles[0]
            role_gpt = conv.roles[1]
            target_lens = []  # 记录需要遮蔽的 input_ids 索引

            # 初始提示词长度
            target_lens.append(
                get_input_ids_len(
                    self.tokenizer(
                        conv.get_prompt(), max_length=self.tokenizer_max_length, padding='max_length', truncation=True, return_tensors='pt'
                    ).input_ids
                )
            )

            conversations = self.data_text[index]['conversations']
            for convo in conversations:
                conv.append_message(role_human if convo['from']=='human' else role_gpt, 
                                    convo['value'].replace('<video>', ''))
                target_lens.append(
                    get_input_ids_len(
                        self.tokenizer(
                            conv.get_prompt(), max_length=self.tokenizer_max_length, padding='max_length', truncation=True, return_tensors='pt'
                        ).input_ids
                    )
                )

            video_data = self.video_process([video_path], [conv.get_prompt()], return_tensors='pt', context_length=self.tokenizer_max_length)

            targets = video_data.input_ids.clone()
            video_data['labels'] = make_label(targets, target_lens)
            return video_data
        else:
            raise NotImplementedError


@dataclass
class VideoCollator(object):
    pad_token_id: Any
    model_max_length: Any
    def __call__(self, instances: Sequence[Dict], ) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.model_max_length]
        labels = labels[:, :self.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.pad_token_id),
        )

        if 'pixel_values' in instances[0]:
            videos = [instance['pixel_values'] for instance in instances]
            if all(x is not None and x.shape == videos[0].shape for x in videos):
                if videos[0].shape[0] == 1:
                    batch['pixel_values'] = torch.vstack(videos)
                else:
                    batch['pixel_values'] = torch.stack(videos)
            else:
                batch['pixel_values'] = videos

        return batch
    
def make_video_data_module(
        video_text_path: str | os.PathLike,
        video_path: str | os.PathLike,
        video_pretrain_path: str | os.PathLike,
        model_max_length: int
    ) -> Dict:
    train_dataset = VideoDataset(
        video_text_path=video_text_path,
        video_path=video_path,
        video_pretrain_path=video_pretrain_path,
        model_max_length=model_max_length
    )
    tokenizer = train_dataset.tokenizer
    data_collator = VideoCollator(
        pad_token_id=tokenizer.pad_token_id,
        model_max_length=model_max_length
    )
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=tokenizer
    )