import os
import subprocess
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
        # 临时存储图像路径的列表
        image_paths = []
        
        for frame_number, filename in images:
            img_path = os.path.join(image_folder, filename)
            image_paths.append(img_path)
        
        # 使用 FFmpeg 来编码视频
        video_filename = os.path.join(output_folder, f'{prefix}_t2vcrossattention_video.mp4')
        
        # 调用 FFmpeg 生成视频
        create_video_with_ffmpeg(image_paths, video_filename, fps)

def create_video_with_ffmpeg(input_image_paths, output_video_path, fps=8):
    # 输入的图像文件需要按序列顺序编号，因此使用 'image%d.jpg' 之类的模式
    # image_pattern = os.path.join(os.path.dirname(input_image_paths[0]), "frame%04d.jpg")
    image_name_prefix = os.path.basename(input_image_paths[0]).split("frame")[0]
    image_name = image_name_prefix + "frame%04d.jpg"
    image_dir = os.path.dirname(input_image_paths[0])
    image_paths = os.path.join(image_dir,image_name)
    # import pdb
    # pdb.set_trace()
    # 构建 FFmpeg 命令行
    cmd = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', image_paths,  # 输入的图像文件序列
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',  # 使用兼容的像素格式
        output_video_path
    ]
    
    # 执行 FFmpeg 命令
    subprocess.run(cmd, check=True)

def main():
    image_folder = "./results/cogvideox-t2v/attn_maps/all/t2v-attention"  # 替换为你的图像文件夹路径
    output_folder = "./results/cogvideox-t2v/attn_maps/all/t2v-attention-video"  # 替换为你的视频输出文件夹路径
    grouped_images = group_images_by_prefix(image_folder)
    images_to_video(image_folder, grouped_images, output_folder)

if __name__ == "__main__":
    main()
