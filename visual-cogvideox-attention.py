import os
import torch
import psutil
from datetime import datetime
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDPMScheduler,
)

from transformers import T5EncoderModel, T5Tokenizer
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from attention_map_diffusers import (
    init_pipeline,
    save_vdm_attention_maps,
)
from diffusers.utils import export_to_video, load_image, load_video


def main():
    result_path = './results/cogvideox-t2v'
    os.makedirs(result_path, exist_ok=True)
    mid_feature_path = os.path.join(result_path, 'mid_feature')
    os.makedirs(mid_feature_path, exist_ok=True)
    # 处理并保存注意力图
    attn_map_dir = os.path.join(result_path, 'attn_maps')
    os.makedirs(attn_map_dir, exist_ok=True)
    output_path = os.path.join(result_path, 'output.mp4')

    guidance_scale = 6.0
    seed = 43
    num_inference_steps = 50
    num_frames = 49
    height = 480
    width = 720
    
    # 从预训练模型加载检查点
    model_path = "/maindata/data/shared/public/haobang.geng/haobang-huggingface/CogVideoX-2b"
    dtype = torch.float16

    # 初始化管道
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    # pipe.to("cuda")
    pipe.enable_sequential_cpu_offload()

    # 替换模块并注册钩子
    pipe = init_pipeline(pipe, mid_feature_path)

    prompt = "A street artist, clad in a worn-out denim jacket and a colorful bandana, stands before a vast concrete wall in the heart, holding a can of spray paint, spray-painting a colorful bird on a mottled wall."

    # 生成视频
    video_generate = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        use_dynamic_cfg=False,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
        height=height,
        width=width,
    ).frames[0]

 
    # 导出视频
    export_to_video(video_generate, output_path)
    
    # 保存注意力图
    save_vdm_attention_maps(mid_feature_path, pipe.tokenizer, prompt, video_generate, output_dir=attn_map_dir, video_height=height, video_width=width, video_frame=num_frames)

if __name__ == "__main__":
    main()

