import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from multiprocessing import Pool

# Set folder paths
npy_folder_path = '8192_npy'  # Original npy files folder
output_folder = '8192_videos'  # Output videos folder
os.makedirs(output_folder, exist_ok=True)  # Create output folder

# Get all npy files
npy_files = [f for f in os.listdir(npy_folder_path) if f.endswith('.npy')]

# Create 3D figure
fig = plt.figure(figsize=(8, 8), dpi=256)  # Set the figure size and DPI
ax = fig.add_subplot(111, projection='3d')

# Update function
def update(num, sc, ax, frames):
    elev_angle = 20 + 10 * np.cos(np.pi * num / frames)  # Elevation angle
    azim_angle = 180 * np.sin(2 * np.pi * num / frames)  # Azimuth angle
    
    ax.view_init(elev=elev_angle, azim=azim_angle)
    
    ax.set_axis_off()
    ax.dist = 10
    return sc,

# Video saving function
def save_video(file_path):
    fps = 2
    frames = fps * 10  # 10 seconds

    point_cloud = np.load(file_path)
    xyz = point_cloud[:, :3]
    rgb = point_cloud[:, 3:]
    
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, marker='o', s=250)
    
    # Create video writer
    npy_filename = os.path.basename(file_path).replace('.npy', '')
    video_filename = os.path.join(output_folder, f"{npy_filename}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (2048, 2048))
    
    for frame in range(frames):
        update(frame, sc, ax, frames)
        fig.canvas.draw()
        
        # Convert the drawn figure to image
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Write the frame to the video file
        out.write(img_bgr)
    
    out.release()

def process_file(npy_file):
    file_path = os.path.join(npy_folder_path, npy_file)
    save_video(file_path)

# Process all npy files and generate videos using multiprocessing
if __name__ == '__main__':
    with Pool() as pool:
        list(tqdm(pool.imap(process_file, npy_files), total=len(npy_files)))
        
plt.close(fig)  # Close the figure to release memory
