import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# 定义文件路径
json_file_path = 'dataset/Video/train/videochatgpt_tune/videochatgpt_llavaimage_tune.json'
output_txt_path = 'missing_images.txt'
output_json_path = 'dataset/Video/train/videochatgpt_tune/videochatgpt_llavaimage_tune_filtered.json'

# 读取JSON文件
with open(json_file_path, 'r') as file:
    data = json.load(file)

# 初始化列表来存储不存在的路径和有效的项目
missing_paths = []
valid_items = []

# 遍历每个项目，检查路径是否存在并能否读取视频
for item in tqdm(data):
    try:
        video_path = item['video']
        video_path = os.path.join('./dataset/Video/train/videochatgpt_tune', video_path)
        if video_path and not os.path.exists(video_path):
            missing_paths.append(video_path)
        else:
            # 尝试读取视频
            try:
                cv2_vr = cv2.VideoCapture(video_path)
                duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_id_list = np.linspace(0, duration-1, 10, dtype=int)  # 假设num_frames为10

                video_data = []
                for frame_idx in frame_id_list:
                    cv2_vr.set(1, frame_idx)
                    _, frame = cv2_vr.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2_vr.release()
            except:
                missing_paths.append(video_path)
    except:
        continue

# 将不存在的路径写入txt文件
with open(output_txt_path, 'w') as file:
    for path in missing_paths:
        file.write(f"{path}\n")

# 将有效的项目写入新的JSON文件
with open(output_json_path, 'w') as file:
    json.dump(valid_items, file, ensure_ascii=False)

print(f"检查完成，不存在的路径已保存到 {output_txt_path}")
print(f"有效的项目已保存到 {output_json_path}")