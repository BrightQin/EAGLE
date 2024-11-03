import os
import json
from dataclasses import dataclass
from typing import Dict, Sequence, Any

import pandas

import torch
from torch.utils.data import Dataset

from ..model.multimodal_encoder.audio_models.languagebind_audio import LanguageBindAudio
from ..model.multimodal_encoder.audio_models.processing_audio import LanguageBindAudioProcessor
from ..model.multimodal_encoder.audio_models.tokenization_audio import LanguageBindAudioTokenizer
from ..model.multimodal_encoder.audio_models.configuration_audio import LanguageBindAudioConfig

from .. import conversation as conversation_lib
from eagle.constants import IGNORE_INDEX
from .utils import get_input_ids_len, make_label

class AudioDataset(Dataset):
    def __init__(
        self,
        audio_text_path: str | os.PathLike,
        audio_path: str | os.PathLike,
        audio_pretrain_path: str | os.PathLike,
        model_max_length: int
    ):
        super().__init__()
        if 'LanguageBind_Audio_FT' in audio_pretrain_path:
            self.tokenizer = LanguageBindAudioTokenizer.from_pretrained(audio_pretrain_path)
            self.config = LanguageBindAudioConfig.from_pretrained(audio_pretrain_path)
            self.audio_process = LanguageBindAudioProcessor(self.config, self.tokenizer)
        else:
            raise ValueError('Unknown audio tower:', audio_pretrain_path)
        if 'AudioSetCaps' in audio_text_path:
            self.dataset_name = 'AudioSetCaps'
            self.data_text = pandas.read_csv(audio_text_path)
        else:
            raise ValueError('Unknown dataset:', audio_text_path)
        self.audio_path = audio_path
        self.tokenizer_max_length = model_max_length

    def __len__(self):
        if self.dataset_name == 'AudioSetCaps':
            return self.data_text.shape[0]
        else:
            raise NotImplementedError
        
    def __getitem__(self, index):
        if self.dataset_name == 'AudioSetCaps':
            audio_path = os.path.join(self.audio_path, self.data_text.id[index] + '.mp3')

            conv = conversation_lib.default_conversation.copy()
            role_human = conv.roles[0]
            role_gpt = conv.roles[1]
            target_lens = [] # index of input_ids to be masked for label
            target_lens.append(
                get_input_ids_len(
                    self.tokenizer(
                        conv.get_prompt(), max_length=self.tokenizer_max_length, padding='max_length', truncation=True, return_tensors='pt'
                    ).input_ids
                )
            )
            conv.append_message(role_human, self.data_text.question_1[index])
            target_lens.append(
                get_input_ids_len(
                    self.tokenizer(
                        conv.get_prompt(), max_length=self.tokenizer_max_length, padding='max_length', truncation=True, return_tensors='pt'
                    ).input_ids
                )
            )
            conv.append_message(role_gpt, self.data_text.answer_1[index])
            target_lens.append(
                get_input_ids_len(
                    self.tokenizer(
                        conv.get_prompt(), max_length=self.tokenizer_max_length, padding='max_length', truncation=True, return_tensors='pt'
                    ).input_ids
                )
            )
            conv.append_message(role_human, self.data_text.question_2[index])
            target_lens.append(
                get_input_ids_len(
                    self.tokenizer(
                        conv.get_prompt(), max_length=self.tokenizer_max_length, padding='max_length', truncation=True, return_tensors='pt'
                    ).input_ids
                )
            )
            conv.append_message(role_gpt, self.data_text.answer_2[index])
            target_lens.append(
                get_input_ids_len(
                    self.tokenizer(
                        conv.get_prompt(), max_length=self.tokenizer_max_length, padding='max_length', truncation=True, return_tensors='pt'
                    ).input_ids
                )
            )
            conv.append_message(role_human, self.data_text.question_3[index])
            target_lens.append(
                get_input_ids_len(
                    self.tokenizer(
                        conv.get_prompt(), max_length=self.tokenizer_max_length, padding='max_length', truncation=True, return_tensors='pt'
                    ).input_ids
                )
            )
            conv.append_message(role_gpt, self.data_text.answer_3[index])
            
            audio_data = self.audio_process([audio_path], [conv.get_prompt()], return_tensors='pt', context_length=self.tokenizer_max_length)

            targets = audio_data.input_ids.clone()
            audio_data['labels'] = make_label(targets, target_lens)
            return audio_data
        else:
            raise NotImplementedError


@dataclass
class AudioCollator(object):
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
            audios = [instance['pixel_values'] for instance in instances]
            if all(x is not None and x.shape == audios[0].shape for x in audios):
                if audios[0].shape[0] == 1:
                    batch['pixel_values'] = torch.vstack(audios)
                else:
                    batch['pixel_values'] = torch.stack(audios)
            else:
                batch['pixel_values'] = audios

        return batch
    
def make_audio_data_module(
        audio_text_path: str | os.PathLike,
        audio_path: str | os.PathLike,
        audio_pretrain_path: str | os.PathLike,
        model_max_length: int
    ) -> Dict:
    train_dataset = AudioDataset(
        audio_text_path=audio_text_path,
        audio_path=audio_path,
        audio_pretrain_path=audio_pretrain_path,
        model_max_length=model_max_length
    )
    tokenizer = train_dataset.tokenizer
    data_collator = AudioCollator(
        pad_token_id=tokenizer.pad_token_id,
        model_max_length=model_max_length
    )
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=tokenizer
    )