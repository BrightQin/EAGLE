# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is modified from https://github.com/haotian-liu/LLaVA/

import os
from .clip_encoder import CLIPVisionTower, LanguageBindAudioTower, LanguageBindVideoTower
# from .languagebind_audio_encoder import LanguageBindAudioTower
# from .languagebind_video_encoder import LanguageBindVideoTower
from .multi_backbone_channel_concatenation_encoder import MultiBackboneChannelConcatenationVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    print(vision_tower)

    if "clip" in vision_tower.lower():
        is_absolute_path_exists = os.path.exists(vision_tower)
        if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)   
        raise ValueError(f'Unknown vision tower: {vision_tower}')
    
    elif ";" in vision_tower:
        return MultiBackboneChannelConcatenationVisionTower(vision_tower, args=vision_tower_cfg)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

# BEGIN
# Add audio tower
# def build_audio_tower(audio_tower_cfg, **kwargs):
#     audio_tower = getattr(audio_tower_cfg, 'mm_audio_tower', getattr(audio_tower_cfg, 'audio_tower', None))

#     if 'LanguageBind_Audio_FT' in audio_tower:
#         is_absolute_path_exists = os.path.exists(audio_tower)
        
#         if is_absolute_path_exists:
#             from .audio_models.languagebind_audio import LanguageBindAudio
#             return LanguageBindAudio.from_pretrained(audio_tower, **kwargs).vision_model       
#         raise ValueError(f'Unknown vision tower: {audio_tower}')

#     else:
#         raise NotImplementedError
# # END

# BEGIN qbs
# Add audio tower
def build_audio_tower(vision_tower_cfg, **kwargs):
    audio_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    if 'LanguageBind_Audio_FT' in audio_tower:
        is_absolute_path_exists = os.path.exists(audio_tower)
        
        if is_absolute_path_exists:
            return LanguageBindAudioTower(audio_tower, args=vision_tower_cfg, **kwargs)     
        raise ValueError(f'Unknown vision tower: {audio_tower}')

    else:
        raise NotImplementedError

# Add video tower
def build_video_tower(vision_tower_cfg, **kwargs):
    video_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    if 'LanguageBind_Video_FT' in video_tower:
        is_absolute_path_exists = os.path.exists(video_tower)
        
        if is_absolute_path_exists:
            return LanguageBindVideoTower(video_tower, args=vision_tower_cfg, **kwargs)     
        raise ValueError(f'Unknown vision tower: {video_tower}')

    else:
        raise NotImplementedError
# END qbs

