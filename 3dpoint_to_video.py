import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from multiprocessing import Pool

# Set folder paths
npy_folder_path = '8192_npy'
output_folder = '8192_videos'
os.makedirs(output_folder, exist_ok=True)

npy_files = [f for f in os.listdir(npy_folder_path) if f.endswith('.npy')]
npy_files.sort()

# Video saving function
def save_video(file_path):
    fps = 2
    frames = fps * 10

    point_cloud = np.load(file_path)
    xyz = point_cloud[:, :3]
    rgb = point_cloud[:, 3:]
    
    # Create a new figure for each process
    plt.switch_backend('Agg')  # Use a non-GUI backend
    fig = plt.figure(figsize=(8, 8), dpi=256)
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, marker='o', s=150)

    npy_filename = os.path.basename(file_path).replace('.npy', '')
    video_filename = os.path.join(output_folder, f"{npy_filename}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (1024, 1024))
    
    for frame in range(frames):
        elev_angle = 20 + 10 * np.cos(np.pi * frame / frames)
        azim_angle = 180 * np.sin(2 * np.pi * frame / frames)

        ax.view_init(elev=elev_angle, azim=azim_angle)
        ax.set_axis_off()
        ax.dist = 10
        
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img_bgr)
    
    out.release()
    plt.close(fig)  # Ensure the figure is closed to free memory

def process_file(npy_file):
    file_path = os.path.join(npy_folder_path, npy_file)
    save_video(file_path)

if __name__ == '__main__':
    with Pool() as pool:
        list(tqdm(pool.imap(process_file, npy_files), total=len(npy_files)))
