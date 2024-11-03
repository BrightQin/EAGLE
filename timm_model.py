import re

# 读取 .py 文件的内容
with open("/home/qinbosheng/HDD/HDD1/Code/Image/EAGLE/eagle/model/multimodal_encoder/vision_models/convnext.py", "r", encoding="utf-8") as file:
    data = file.read()

# 使用正则表达式从文本中提取模型名称
model_names = re.findall(r"'(.*?)':", data)

# 在每个模型名称之前加上 'timm/'
prefixed_names = [f"timm/{name}" for name in model_names]

# 将结果保存到 txt 文件中
with open("model_names.txt", "w", encoding="utf-8") as txt_file:
    for name in prefixed_names:
        txt_file.write(name + "\n")

print("模型名称已保存到 model_names.txt 文件中。")