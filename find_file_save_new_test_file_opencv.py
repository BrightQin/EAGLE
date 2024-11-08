import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

# 定义文件路径
json_file_path = './videochatgpt_tune/videochatgpt_llavaimage_tune.json'
output_txt_path = 'missing_videos_opencv.txt'
output_json_path = './videochatgpt_tune/videochatgpt_llavaimage_tune_filtered_opencv.json'

# 读取JSON文件
with open(json_file_path, 'r') as file:
    data = json.load(file)

def process_item(item):
    try:
        video_path = item['video']
        video_path = os.path.join('./videochatgpt_tune', video_path)
        if video_path and not os.path.exists(video_path):
            return (video_path, None)
        else:
            # 尝试读取视频
            try:
                cv2_vr = cv2.VideoCapture(video_path)
                duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_id_list = np.linspace(0, duration - 1, 10, dtype=int)  # 假设num_frames为10

                for frame_idx in frame_id_list:
                    cv2_vr.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cv2_vr.read()
                    if not ret:
                        cv2_vr.release()
                        return (video_path, None)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2_vr.release()
                return (None, item)  # 有效项目
            except:
                return (video_path, None)
    except:
        return (None, None)

if __name__ == '__main__':
    # 使用多进程池并行处理
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_item, data), total=len(data)))

    # 分离结果
    missing_paths = []
    valid_items = []

    for missing_path, valid_item in results:
        if missing_path:
            missing_paths.append(missing_path)
        if valid_item:
            valid_items.append(valid_item)

    # 将不存在的路径写入txt文件
    with open(output_txt_path, 'w') as file:
        for path in missing_paths:
            file.write(f"{path}\n")

    # 将有效的项目写入新的JSON文件
    with open(output_json_path, 'w') as file:
        json.dump(valid_items, file, ensure_ascii=False)

    print(f"检查完成，不存在的路径已保存到 {output_txt_path}")
    print(f"有效的项目已保存到 {output_json_path}")
