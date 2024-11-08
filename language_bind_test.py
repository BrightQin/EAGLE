import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
from eagle.model.multimodal_encoder.audio_models.languagebind_audio import LanguageBindAudio, CLIPVisionModel
from eagle.model.multimodal_encoder.audio_models.processing_audio import LanguageBindAudioProcessor
from eagle.model.multimodal_encoder.audio_models.tokenization_audio import LanguageBindAudioTokenizer
# from eagle.model.multimodal_encoder.languagebind import LanguageBindAudio, LanguageBindAudioTokenizer, LanguageBindAudioProcessor
# from eagle.model.multimodal_encoder.video_models.languagebind_video import LanguageBindVideo, CLIPVisionModel, CLIPVisionTransformer
# from eagle.model.multimodal_encoder.video_models.processing_video import LanguageBindVideoProcessor
# from eagle.model.multimodal_encoder.video_models.tokenization_video import LanguageBindVideoTokenizer
# from eagle.model.multimodal_encoder.languagebind_ori import LanguageBindVideo, LanguageBindVideoProcessor, LanguageBindVideoTokenizer



# pretrained_ckpt = './model/LanguageBind_Video_FT'
# vision_tower = CLIPVisionModel.from_pretrained(pretrained_ckpt).cuda()
# model = LanguageBindVideo.from_pretrained(pretrained_ckpt).cuda()
# tokenizer = LanguageBindVideoTokenizer.from_pretrained(pretrained_ckpt)
# video_process = LanguageBindVideoProcessor(model.config, tokenizer)

# model.eval()
# data = video_process(["dataset/Video/train/videochatgpt_tune/videochatgpt_tune/v___c8enCfzqw.mp4", 
#         "dataset/Video/train/videochatgpt_tune/videochatgpt_tune/v___mIAEE03bE.mp4"], return_tensors='pt')
# print(data['pixel_values'].shape)
# data['pixel_values'] = data['pixel_values'].cuda()
# image_features = model.get_image_features(data['pixel_values'])
# print(image_features.shape)


# torch.Size([2, 3, 8, 224, 224])

# pretrained_ckpt = './model/LanguageBind_Audio_FT'
# vision_tower = CLIPVisionModel.from_pretrained(pretrained_ckpt).cuda()
# model = LanguageBindAudio.from_pretrained(pretrained_ckpt)
# tokenizer = LanguageBindAudioTokenizer.from_pretrained(pretrained_ckpt)
# video_process = LanguageBindAudioProcessor(model.config, tokenizer)

# model.eval()
# data = video_process(["dataset/Audio/AudioSetCaps/example/_7Xe9vD3Hpg_4_10.mp3"], return_tensors='pt')
# print(data['pixel_values'].shape)

# CLIPVisionConfig {
#   "_name_or_path": "./model/LanguageBind_Audio_FT",
#   "add_time_attn": false,
#   "attention_dropout": 0.0,
#   "audio_mean": -4.2677393,
#   "audio_sample_rate": 16000,
#   "audio_std": 4.5689974,
#   "force_patch_dropout": 0.0,
#   "hidden_act": "gelu",
#   "hidden_size": 1024,
#   "image_size": 224,
#   "initializer_factor": 1.0,
#   "initializer_range": 0.02,
#   "intermediate_size": 4096,
#   "layer_norm_eps": 1e-05,
#   "lora_alpha": 16,
#   "lora_dropout": 0.0,
#   "lora_r": 0,
#   "model_type": "clip_vision_model",
#   "num_attention_heads": 16,
#   "num_channels": 3,
#   "num_frames": 1,
#   "num_hidden_layers": 24,
#   "num_mel_bins": 112,
#   "patch_size": 14,
#   "projection_dim": 512,
#   "target_length": 1036,
#   "transformers_version": "4.45.2",
#   "video_decode_backend": "decord"
# }

# CLIPVisionConfig {
#   "add_time_attn": false,
#   "attention_dropout": 0.0,
#   "audio_mean": -4.2677393,
#   "audio_sample_rate": 16000,
#   "audio_std": 4.5689974,
#   "force_patch_dropout": 0.0,
#   "hidden_act": "gelu",
#   "hidden_size": 1024,
#   "image_size": 224,
#   "initializer_factor": 1.0,
#   "initializer_range": 0.02,
#   "intermediate_size": 4096,
#   "layer_norm_eps": 1e-05,
#   "lora_alpha": 16,
#   "lora_dropout": 0.0,
#   "lora_r": 0,
#   "model_type": "clip_vision_model",
#   "num_attention_heads": 16,
#   "num_channels": 3,
#   "num_frames": 1,
#   "num_hidden_layers": 24,
#   "num_mel_bins": 112,
#   "patch_size": 14,
#   "projection_dim": 512,
#   "target_length": 1036,
#   "transformers_version": "4.45.2",
#   "video_decode_backend": "decord"
# }

# Error(s) in loading state_dict for CLIPVisionModel:
	# size mismatch for vision_model.embeddings.position_embedding.weight: copying a param with shape torch.Size([593, 1024]) from checkpoint, the shape in current model is torch.Size([257, 1024]).