import os
import json
from tqdm import tqdm

# 定义文件路径
json_file_path = 'dataset/Video/videochatgpt_llavaimage_tune.json'
output_txt_path = 'missing_images.txt'
output_json_path = 'dataset/Video/videochatgpt_llavaimage_tune_filtered.json'

# 读取JSON文件
with open(json_file_path, 'r') as file:
    data = json.load(file)

# 初始化列表来存储不存在的路径和有效的项目
missing_paths = []
valid_items = []

# 遍历每个项目，检查路径是否存在
for item in tqdm(data):
    try:
        image_path = item['video']
        image_path = os.path.join('./dataset/Video/train', image_path)
        if image_path and not os.path.exists(image_path):
            missing_paths.append(image_path)
        else:
            valid_items.append(item)
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