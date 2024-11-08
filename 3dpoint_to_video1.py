import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def convert_pointcloud_to_image(file_path, output_dir):
    # 加载npy文件
    point_cloud = np.load(file_path)

    # 提取xyz和rgb
    xyz = point_cloud[:, :3]
    rgb = point_cloud[:, 3:6]

    # 创建3D图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, marker='o', s=1)

    # 设置视图参数
    ax.view_init(elev=20, azim=30) # 修改角度以变化视角

    # 保存图像
    output_file = os.path.join(output_dir, f"{os.path.basename(file_path).replace('.npy', '.png')}")
    plt.savefig(output_file, dpi=300)
    plt.close()

def process_folder(input_folder, output_folder, num_images=240):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]

    # 确保只有240个图像被转换
    for file in files[:num_images]:
        file_path = os.path.join(input_folder, file)
        convert_pointcloud_to_image(file_path, output_folder)

# 输入和输出路径
input_folder = '8192_npy'
output_folder = 'images_output'

process_folder(input_folder, output_folder)
