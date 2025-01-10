import os
import cv2
from collections import defaultdict

def group_images_by_prefix(image_folder):
    # 创建一个默认字典来存储分组的图像
    grouped_images = defaultdict(list)
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(image_folder):
        if "frame" in filename:
            # 获取图像文件的前缀（假设前缀是指去掉frame和后缀的部分）
            prefix = filename.split("frame")[0]
            # 获取帧数
            frame_number = int(filename.split("frame")[1].split('.')[0])
            # 将图像文件添加到相应的前缀组中
            grouped_images[prefix].append((frame_number, filename))
    
    # 按帧数对每组图像进行排序
    for prefix in grouped_images:
        grouped_images[prefix].sort()
    
    return grouped_images

def images_to_video(image_folder, grouped_images, output_folder, fps=8):
    os.makedirs(output_folder, exist_ok=True)
    
    for prefix, images in grouped_images.items():
        frame_array = []
        for frame_number, filename in images:
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)
            height, width, layers = img.shape
            frame_array.append(img)
        
        # 定义视频输出的参数
        video_filename = os.path.join(output_folder, f'{{{prefix}}}_t2vcrossattention_video.mp4')
        out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        for frame in frame_array:
            out.write(frame)
        out.release()

def main():
    image_folder = "./results/cogvideox-t2v/attn_maps/all/t2v-attention"  # 替换为你的图像文件夹路径
    output_folder = "./results/cogvideox-t2v/attn_maps/all/t2v-attention-video"  # 替换为你的视频输出文件夹路径
    grouped_images = group_images_by_prefix(image_folder)
    images_to_video(image_folder, grouped_images, output_folder)

if __name__ == "__main__":
    main()
