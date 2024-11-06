import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
# from eagle.model.multimodal_encoder.audio_models.languagebind_audio import LanguageBindAudio
# from eagle.model.multimodal_encoder.audio_models.processing_audio import LanguageBindAudioProcessor
# from eagle.model.multimodal_encoder.audio_models.tokenization_audio import LanguageBindAudioTokenizer
from eagle.model.multimodal_encoder.languagebind import LanguageBindAudio, LanguageBindAudioTokenizer, LanguageBindAudioProcessor
from eagle.model.multimodal_encoder.video_models.languagebind_video import LanguageBindVideo, CLIPVisionModel
from eagle.model.multimodal_encoder.video_models.processing_video import LanguageBindVideoProcessor
from eagle.model.multimodal_encoder.video_models.tokenization_video import LanguageBindVideoTokenizer
# from eagle.model.multimodal_encoder.languagebind_ori import LanguageBindVideo, LanguageBindVideoProcessor, LanguageBindVideoTokenizer



pretrained_ckpt = './model/LanguageBind_Video_FT'
# vision_tower = CLIPVisionModel.from_pretrained(pretrained_ckpt).cuda()
model = LanguageBindVideo.from_pretrained(pretrained_ckpt).cuda()
tokenizer = LanguageBindVideoTokenizer.from_pretrained(pretrained_ckpt)
video_process = LanguageBindVideoProcessor(model.config, tokenizer)

model.eval()
data = video_process(["dataset/Video/train/videochatgpt_tune/videochatgpt_tune/v___c8enCfzqw.mp4", 
        "dataset/Video/train/videochatgpt_tune/videochatgpt_tune/v___mIAEE03bE.mp4"], return_tensors='pt')
print(data['pixel_values'].shape)
data['pixel_values'] = data['pixel_values'].cuda()
image_features = model.get_image_features(data['pixel_values'])
print(image_features.shape)


# torch.Size([2, 3, 8, 224, 224])

# pretrained_ckpt = './model/LanguageBind_Video_FT'
# model = LanguageBindAudio.from_pretrained(pretrained_ckpt)
# tokenizer = LanguageBindAudioTokenizer.from_pretrained(pretrained_ckpt)
# video_process = LanguageBindAudioProcessor(model.config, tokenizer)

# model.eval()
# data = video_process(["dataset/Audio/AudioSetCaps/example/_7Xe9vD3Hpg_4_10.mp3"], return_tensors='pt')
# print(data['pixel_values'].shape)