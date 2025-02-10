import os
import torch
from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video
from attention_map_diffusers import (
    attn_maps,
    init_pipeline,
    average_attn_map,
    visualize_v2v_attnmap,
)


def main():
    # ========================
    # 1. 配置相关路径
    # ========================
    result_path = './results/cogvideox-t2v'
    attn_map_dir = os.path.join(result_path, 'attn_maps')
    output_path = os.path.join(result_path, 'output.mp4')
    os.makedirs(attn_map_dir, exist_ok=True)

    # ========================
    # 2. 设置生成参数
    # ========================
    guidance_scale = 6.0
    seed = 43
    num_inference_steps = 50
    num_frames = 49
    height = 480
    width = 720

    # 模型路径与数据类型（半精度）
    model_path = "/maindata/data/shared/public/haobang.geng/haobang-huggingface/CogVideoX-5b"
    dtype = torch.float16

    # 输入的文本提示
    prompt = (
        "A suited astronaut, with the red dust of Mars clinging to their boots, reaches out to shake hands with an alien being, "
        "their skin a shimmering blue, under the pink-tinged sky of the fourth planet. In the background, a sleek silver rocket, "
        "a beacon of human ingenuity, stands tall, its engines powered down, as the two representatives of different worlds exchange "
        "a historic greeting amidst the desolate beauty of the Martian landscape."
    )

    # ========================
    # 3. 初始化 CogVideoX 管道
    # ========================
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    # 使用 trailing timestep spacing 的调度器配置
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    # 启用顺序 CPU offload，降低内存占用
    pipe.enable_sequential_cpu_offload()
    # 调用自定义的初始化函数（可能包含注入增强模块等操作）
    pipe = init_pipeline(pipe)

    # ========================
    # 4. 视频生成
    # ========================
    generator = torch.Generator().manual_seed(seed)
    video_output = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        use_dynamic_cfg=False,
        guidance_scale=guidance_scale,
        generator=generator,
        height=height,
        width=width,
    ).frames[0]

    # ========================
    # 5. 处理并可视化注意力图
    # ========================
    # 对收集到的注意力图进行平均，得到每个模块对应的 3x3 tensor
    averaged_attn_maps = average_attn_map(attn_maps)
    for name, avg_map in averaged_attn_maps.items():
        visualize_v2v_attnmap(avg_map, attn_map_dir, name)

    # ========================
    # 6. 导出生成的视频
    # ========================
    export_to_video(video_output, output_path)


if __name__ == "__main__":
    main()
