import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import re
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    JointAttnProcessor2_0,
    FluxAttnProcessor2_0,
    CogVideoXAttnProcessor2_0
)

from .modules import *



def hook_function(name, detach=True):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map_v2v"):

            timestep = module.processor.timestep
            attn_maps[timestep] = attn_maps.get(timestep, dict())
            if name not in attn_maps[timestep]:
                attn_maps[timestep][name] = dict()
            attn_maps[timestep][name]['v2v'] = module.processor.attn_map_v2v.cpu() if detach \
                else module.processor.attn_map_v2v
            del module.processor.attn_map_v2v

    return forward_hook


def register_cross_attention_hook(model, hook_function, target_name):
    for name, module in model.named_modules():
        if not name.endswith(target_name):
            continue

        elif isinstance(module.processor, CogVideoXAttnProcessor2_0):
            module.processor.store_attn_map = True

        hook = module.register_forward_hook(hook_function(name))
    
    return model



def replace_call_method_for_cogvideox(model):
    if model.__class__.__name__ == 'CogVideoXTransformer3DModel':
        from diffusers.models.transformers import CogVideoXTransformer3DModel
        model.forward = CogVideoXTransformer3DModelForward.__get__(model, CogVideoXTransformer3DModel)

    for name, layer in model.named_children():
        if layer.__class__.__name__ == 'CogVideoXBlock':
            from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock
            layer.forward = CogVideoXBlockForward.__get__(layer, CogVideoXBlock)
        
        replace_call_method_for_cogvideox(layer)
    
    return model


def init_pipeline(pipeline,name):
    if name == "CogVideoXImageToVideoPipeline":
        from diffusers import CogVideoXImageToVideoPipeline
        CogVideoXAttnProcessor2_0.__call__ = cogvideox_attn_call2_0
        CogVideoXImageToVideoPipeline.__call__ = CogVideoXImageToVideoPipeline_call
        pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn1')
        pipeline.transformer = replace_call_method_for_cogvideox(pipeline.transformer)
    elif name == "CogVideoXPipeline":
        from diffusers import CogVideoXPipeline
        CogVideoXAttnProcessor2_0.__call__ = cogvideox_attn_call2_0
        CogVideoXPipeline.__call__ = CogVideoXPipeline_call
        pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn1')
        pipeline.transformer = replace_call_method_for_cogvideox(pipeline.transformer)
        # TODO: HUNYUAN VIDEO Dit
    else:
        if pipeline.unet.__class__.__name__ == 'UNet2DConditionModel':
            pipeline.unet = register_cross_attention_hook(pipeline.unet, hook_function, 'attn2')
            pipeline.unet = replace_call_method_for_unet(pipeline.unet)
    return pipeline

def sanitize_token(token):
    """
    清理token中的非法字符，使其更具可读性。
    1. 替换文件名不允许的字符（如 /, \, :, *, ?, ", <, >, | 等）。
    2. 去掉前后多余的空格。
    """
    # 定义文件名中非法字符的正则表达式
    illegal_chars = r'[\/\\\:\*\?\"\<\>\|]'
    # 替换非法字符为下划线
    sanitized_token = re.sub(illegal_chars, '_', token)
    # 去掉前后空格
    sanitized_token = sanitized_token.strip()
    return sanitized_token


def clean_token(token):
    """
    清理单个 token 的特殊字符
    Args:
        token (str): 原始 token
    Returns:
        str: 清理后的 token
    """
    # 去除特殊的分词标记（如 `<s>`、`</s>` 等）
    token = re.sub(r'<.*?>', '', token)
    
    # 去除分词前缀（如 `Ġ`, `▁` 等）
    token = token.replace('Ġ', '').replace('▁', '')

    # 替换不可见字符（如换行、制表符等）
    token = token.replace('\n', '').replace('\t', '').replace('\r', '')

    # 去除非字母数字字符，保留字母、数字和空格
    token = re.sub(r'[^\w\s]', '', token)

    # 去除多余的空格
    return token.strip()

def process_token(token, startofword):
    if '</w>' in token:
        token = token.replace('</w>', '')
        if startofword:
            token = '<' + token + '>'
        else:
            token = '-' + token + '>'
            startofword = True
    elif token not in ['<|startoftext|>', '<|endoftext|>']:
        if startofword:
            token = '<' + token + '-'
            startofword = False
        else:
            token = '-' + token + '-'
    return token, startofword


def save_attention_image(attn_map, tokens, batch_dir, to_pil):
    startofword = True
    for i, (token, a) in enumerate(zip(tokens, attn_map[:len(tokens)])):
        # token, startofword = process_token(token, startofword)
        token= sanitize_token(token)
        to_pil(a.to(torch.float32)).save(os.path.join(batch_dir, f'{i}-{token}.png'))


def save_attention_maps(attn_maps, tokenizer, prompts, base_dir='attn_maps', unconditional=True):
    to_pil = ToPILImage()
    
    token_ids = tokenizer(prompts)['input_ids']
    token_ids = token_ids if token_ids and isinstance(token_ids[0], list) else [token_ids]
    total_tokens = [tokenizer.convert_ids_to_tokens(token_id) for token_id in token_ids]
    
    os.makedirs(base_dir, exist_ok=True)
    
    total_attn_map = list(list(attn_maps.values())[0].values())[0].sum(1)
    if unconditional:
        total_attn_map = total_attn_map.chunk(2)[1]  # (batch, height, width, attn_dim)
    total_attn_map = total_attn_map.permute(0, 3, 1, 2)
    total_attn_map = torch.zeros_like(total_attn_map)
    total_attn_map_shape = total_attn_map.shape[-2:]
    total_attn_map_number = 0
    
    for timestep, layers in attn_maps.items():
        timestep_dir = os.path.join(base_dir, f'{timestep}')
        os.makedirs(timestep_dir, exist_ok=True)
        
        for layer, attn_map in layers.items():
            layer_dir = os.path.join(timestep_dir, f'{layer}')
            os.makedirs(layer_dir, exist_ok=True)
            
            attn_map = attn_map.sum(1).squeeze(1).permute(0, 3, 1, 2)
            if unconditional:
                attn_map = attn_map.chunk(2)[1]
            
            resized_attn_map = F.interpolate(attn_map, size=total_attn_map_shape, mode='bilinear', align_corners=False)
            total_attn_map += resized_attn_map
            total_attn_map_number += 1
            
            for batch, (tokens, attn) in enumerate(zip(total_tokens, attn_map)):
                batch_dir = os.path.join(layer_dir, f'batch-{batch}')
                os.makedirs(batch_dir, exist_ok=True)
                save_attention_image(attn, tokens, batch_dir, to_pil)
    
    total_attn_map /= total_attn_map_number
    for batch, (attn_map, tokens) in enumerate(zip(total_attn_map, total_tokens)):
        batch_dir = os.path.join(base_dir, f'batch-{batch}')
        os.makedirs(batch_dir, exist_ok=True)
        save_attention_image(attn_map, tokens, batch_dir, to_pil)


def save_cross_attention_image(attn_map, tokens, batch_dir, to_pil):
    # startofword = True
    for i in range(len(tokens[0])):
        token = tokens[0][i]
        # token, startofword = process_token(token, startofword)
        token= sanitize_token(token)
        attn = attn_map[i]
        for j in range(attn.shape[0]):
            to_pil(attn[j].to(torch.float32)).save(os.path.join(batch_dir, f'{i}-{token}-latentframe-{j}.png'))

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def save_cross_attention_in_image(video_generate, attn_map, tokens, batch_dir):
    batch_dir = os.path.join(batch_dir,'t2v-attention')
    os.makedirs(batch_dir,exist_ok=True)
    # startofword = True
    for i in range(len(tokens[0])):
        token = tokens[0][i]
        token= sanitize_token(token)
        attn = attn_map[i]
        for j in range(attn.shape[0]):
            attn_map_img = attn[j].to(torch.float32).numpy()
            # 将图像转换为 NumPy 数组 (以便用于 show_cam_on_image)
            image_np = np.array(video_generate[j]) / 255.0  # 将像素值缩放到 [0, 1] 范围
            result_img = show_cam_on_image(image_np, attn_map_img, use_rgb=True)
            output_path =  os.path.join(batch_dir, f'token{i}-{token}_frame{j}.png')
            Image.fromarray((result_img * 1).astype(np.uint8)).save(output_path)

def save_vdm_attention_maps(mid_feature_path, tokenizer, prompts, video_generate, output_dir='attn_maps', video_height=128, video_width=128, video_frame=9):
    to_pil = ToPILImage()
    
    token_ids = tokenizer(prompts)['input_ids']
    token_ids = token_ids if token_ids and isinstance(token_ids[0], list) else [token_ids]
    total_tokens = [tokenizer.convert_ids_to_tokens(token_id) for token_id in token_ids]
    os.makedirs(output_dir, exist_ok=True)
    
    total_attn_map_all = None
    total_attn_map_t2t = None
    total_attn_map_v2v = None
    total_attn_map_t2v = None
    total_attn_map_number = 0

    for step in os.listdir(mid_feature_path):
        step_dir = os.path.join(mid_feature_path, step)
        result_step_dir = os.path.join(output_dir,f"step_{step}")
        # os.makedirs(result_step_dir,exist_ok=True)
        if not os.path.isdir(step_dir):
            continue

        for layer_name in os.listdir(step_dir):
            layer_dir = os.path.join(step_dir, layer_name)
            result_layer_dir = os.path.join(result_step_dir,f"layer_{layer_name}")
            # os.makedirs(result_layer_dir,exist_ok=True)
            if not os.path.isdir(layer_dir):
                continue

            # all
            attn_map_all = torch.load(os.path.join(layer_dir, f'{layer_name}_attn_map_all.pth'), map_location='cpu')
            attn_map_t2t = torch.load(os.path.join(layer_dir, f'{layer_name}_attn_map_t2t.pth'), map_location='cpu')
            attn_map_t2v = torch.load(os.path.join(layer_dir, f'{layer_name}_attn_map_t2v.pth'), map_location='cpu')
            
            attn_map_t2v = F.interpolate(attn_map_t2v, size=(video_height, video_width), mode='bilinear', align_corners=False)
            attn_map_t2v_video = torch.zeros(attn_map_t2v.shape[0],video_frame,attn_map_t2v.shape[2], attn_map_t2v.shape[3])
            attn_map_t2v_video[:,0,:,:]=attn_map_t2v[:,0,:,:]
            for i in range(1,attn_map_t2v.shape[1]):
                start_idx = 4 *i -3
                end_idx = 4*(i+1) -3
                attn_map_t2v_video[:,start_idx:end_idx,:,:] = attn_map_t2v[:, i:i+1, :, :].repeat(1, 4, 1, 1)
            

            attn_map_v2v = torch.load(os.path.join(layer_dir, f'{layer_name}_attn_map_v2v.pth'), map_location='cpu')
            attn_map_v2v = attn_map_v2v.view(attn_map_v2v.shape[0]*attn_map_v2v.shape[1]*attn_map_v2v.shape[2],-1)


            if total_attn_map_all is None:
                total_attn_map_all = torch.zeros_like(attn_map_all)
                total_attn_map_t2t = torch.zeros_like(attn_map_t2t)
                total_attn_map_v2v = torch.zeros_like(attn_map_v2v)
                total_attn_map_t2v = torch.zeros(attn_map_t2v_video.shape[0], video_frame, video_height, video_width)


            total_attn_map_all += attn_map_all
            total_attn_map_t2t += attn_map_t2t
            total_attn_map_v2v += attn_map_v2v
            total_attn_map_t2v += attn_map_t2v_video


            
            # attn_map_all = (attn_map_all - attn_map_all.min()) / (attn_map_all.max() - attn_map_all.min()+1e-6)
            # save_path_all = os.path.join(result_layer_dir,'attention-all.png')
            # save_image(attn_map_all.unsqueeze(0), save_path_all)

            
            # attn_map_t2t = (attn_map_t2t - attn_map_t2t.min()) / (attn_map_t2t.max() - attn_map_t2t.min()+1e-6)
            # save_path_t2t = os.path.join(result_layer_dir,'attention-t2t.png')
            # save_image(attn_map_t2t.unsqueeze(0), save_path_t2t)    

            
            # normalized_tensor = torch.zeros_like(attn_map_t2v)
            # for i in range(attn_map_t2v.size(1)):
            #     # 提取当前张量
            #     current_tensor = attn_map_t2v[:, i, :, :]
            #     # 进行归一化操作，使最大值为1，最小值为0
            #     min_val = current_tensor.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
            #     max_val = current_tensor.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            #     normalized_tensor[:, i, :, :] = (current_tensor - min_val) / (max_val - min_val + 1e-6)
            # save_cross_attention_image(normalized_tensor, total_tokens, result_layer_dir, to_pil)

            
            # attn_map_v2v = (attn_map_v2v - attn_map_v2v.min()) / (attn_map_v2v.max() - attn_map_v2v.min() + 1e-6)
            # save_path_v2v = os.path.join(result_layer_dir,'attention-v2v.png')
            # save_image(attn_map_v2v.unsqueeze(0), save_path_v2v)   


            

            total_attn_map_number += 1

    total_attn_map_all /= total_attn_map_number
    total_attn_map_t2t /= total_attn_map_number
    total_attn_map_v2v /= total_attn_map_number
    total_attn_map_t2v /= total_attn_map_number

    total_attn_map_all = (total_attn_map_all - total_attn_map_all.min()) / (total_attn_map_all.max() - total_attn_map_all.min()+1e-6)
    total_attn_map_t2t = (total_attn_map_t2t - total_attn_map_t2t.min()) / (total_attn_map_t2t.max() - total_attn_map_t2t.min()+1e-6)
    total_attn_map_v2v = (total_attn_map_v2v - total_attn_map_v2v.min()) / (total_attn_map_v2v.max() - total_attn_map_v2v.min() + 1e-6)
    
    normalized_total_attn_map_t2v = torch.zeros_like(total_attn_map_t2v)
    for i in range(total_attn_map_t2v.size(1)):
        # 提取当前张量
        current_tensor = total_attn_map_t2v[:, i, :, :]
        # 进行归一化操作，使最大值为1，最小值为0
        min_val = current_tensor.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_val = current_tensor.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        normalized_total_attn_map_t2v[:, i, :, :] = (current_tensor - min_val) / (max_val - min_val + 1e-6)


    # 保存整体attention
    model_dir = os.path.join(output_dir, 'all')
    os.makedirs(model_dir, exist_ok=True)
    save_path_all = os.path.join(model_dir, 'attention-all.png')
    save_image(total_attn_map_all.unsqueeze(0), save_path_all)

    # 保存整体的t2t self attention
    save_path_t2t = os.path.join(model_dir, 'attention-t2t.png')
    save_image(total_attn_map_t2t.unsqueeze(0), save_path_t2t)

    # 保存整体的v2v self attention
    save_path_v2v = os.path.join(model_dir, 'attention-v2v.png')
    save_image(total_attn_map_v2v.unsqueeze(0), save_path_v2v)

    # 保存整体的t2v cross attention
    save_cross_attention_in_image(video_generate, normalized_total_attn_map_t2v, total_tokens, model_dir)

def average_attn_map(attn_maps):
    averaged_attn_maps = {}
    for timestep, modules in attn_maps.items():
        for name, data in modules.items():
            if name not in averaged_attn_maps:
                averaged_attn_maps[name] = []
            averaged_attn_maps[name].append(data["v2v"])
    for name in averaged_attn_maps:
        averaged_attn_maps[name] = torch.stack(averaged_attn_maps[name]).mean(dim=0)
    return averaged_attn_maps


def visualize_v2v_attnmap(attn_map,output_dir,name):
    plt.figure(figsize=(6, 5))
    # sns.heatmap(frame_attention.numpy(), annot=True, cmap="viridis", fmt=".2f")
    sns.heatmap(attn_map.float().numpy(), annot=False, cmap="viridis")
    plt.xlabel("Frame Index")
    plt.ylabel("Frame Index")
    plt.title("Frame-to-Frame Attention")
    plt.show()
    output_path = os.path.join(output_dir,f"module_{name}.png")
    plt.savefig(output_path)
    plt.close()
