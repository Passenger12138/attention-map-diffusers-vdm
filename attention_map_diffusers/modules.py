import math
import inspect
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union, List, Callable

import torch
import torch.nn.functional as F
from einops import rearrange

from diffusers.models.attention import _chunked_feed_forward
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
# from diffusers.pipelines.sana.pipeline_output import SanaPipelineOutput
# from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
# from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
# from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
#     ASPECT_RATIO_512_BIN,
#     ASPECT_RATIO_1024_BIN,
# )
from diffusers.pipelines.flux.pipeline_flux import (
    retrieve_timesteps,
    replace_example_docstring,
    EXAMPLE_DOC_STRING,
    calculate_shift,
    XLA_AVAILABLE,
    FluxPipelineOutput
)
# from diffusers.models.transformers import FLUXTransformer2DModel
from diffusers.utils import (
    deprecate,
    BaseOutput,
    is_torch_version,
    logging,
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    AttnProcessor2_0,
)
from diffusers.callbacks import PipelineCallback,MultiPipelineCallbacks
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from diffusers.schedulers.scheduling_dpm_cogvideox import CogVideoXDPMScheduler
from diffusers.image_processor import PipelineImageInput
logger = logging.get_logger(__name__)
attn_maps = {}




# TODO: implement
# @torch.no_grad()
# @replace_example_docstring(EXAMPLE_DOC_STRING)
# def SanaPipeline_call(
#     self,
#     prompt: Union[str, List[str]] = None,
#     negative_prompt: str = "",
#     num_inference_steps: int = 20,
#     timesteps: List[int] = None,
#     sigmas: List[float] = None,
#     guidance_scale: float = 4.5,
#     num_images_per_prompt: Optional[int] = 1,
#     height: int = 1024,
#     width: int = 1024,
#     eta: float = 0.0,
#     generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
#     latents: Optional[torch.Tensor] = None,
#     prompt_embeds: Optional[torch.Tensor] = None,
#     prompt_attention_mask: Optional[torch.Tensor] = None,
#     negative_prompt_embeds: Optional[torch.Tensor] = None,
#     negative_prompt_attention_mask: Optional[torch.Tensor] = None,
#     output_type: Optional[str] = "pil",
#     return_dict: bool = True,
#     clean_caption: bool = True,
#     use_resolution_binning: bool = True,
#     callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
#     callback_on_step_end_tensor_inputs: List[str] = ["latents"],
#     max_sequence_length: int = 300,
#     complex_human_instruction: List[str] = [
#         "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
#         "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
#         "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
#         "Here are examples of how to transform or refine prompts:",
#         "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
#         "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
#         "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
#         "User Prompt: ",
#     ],
# ) -> Union[SanaPipelineOutput, Tuple]:
#     """
#     Function invoked when calling the pipeline for generation.

#     Args:
#         prompt (`str` or `List[str]`, *optional*):
#             The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
#             instead.
#         negative_prompt (`str` or `List[str]`, *optional*):
#             The prompt or prompts not to guide the image generation. If not defined, one has to pass
#             `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
#             less than `1`).
#         num_inference_steps (`int`, *optional*, defaults to 20):
#             The number of denoising steps. More denoising steps usually lead to a higher quality image at the
#             expense of slower inference.
#         timesteps (`List[int]`, *optional*):
#             Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
#             in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
#             passed will be used. Must be in descending order.
#         sigmas (`List[float]`, *optional*):
#             Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
#             their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
#             will be used.
#         guidance_scale (`float`, *optional*, defaults to 4.5):
#             Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
#             `guidance_scale` is defined as `w` of equation 2. of [Imagen
#             Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
#             1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
#             usually at the expense of lower image quality.
#         num_images_per_prompt (`int`, *optional*, defaults to 1):
#             The number of images to generate per prompt.
#         height (`int`, *optional*, defaults to self.unet.config.sample_size):
#             The height in pixels of the generated image.
#         width (`int`, *optional*, defaults to self.unet.config.sample_size):
#             The width in pixels of the generated image.
#         eta (`float`, *optional*, defaults to 0.0):
#             Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
#             [`schedulers.DDIMScheduler`], will be ignored for others.
#         generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
#             One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
#             to make generation deterministic.
#         latents (`torch.Tensor`, *optional*):
#             Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
#             generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
#             tensor will ge generated by sampling using the supplied random `generator`.
#         prompt_embeds (`torch.Tensor`, *optional*):
#             Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
#             provided, text embeddings will be generated from `prompt` input argument.
#         prompt_attention_mask (`torch.Tensor`, *optional*): Pre-generated attention mask for text embeddings.
#         negative_prompt_embeds (`torch.Tensor`, *optional*):
#             Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
#             provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
#         negative_prompt_attention_mask (`torch.Tensor`, *optional*):
#             Pre-generated attention mask for negative text embeddings.
#         output_type (`str`, *optional*, defaults to `"pil"`):
#             The output format of the generate image. Choose between
#             [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
#         return_dict (`bool`, *optional*, defaults to `True`):
#             Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
#         clean_caption (`bool`, *optional*, defaults to `True`):
#             Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
#             be installed. If the dependencies are not installed, the embeddings will be created from the raw
#             prompt.
#         use_resolution_binning (`bool` defaults to `True`):
#             If set to `True`, the requested height and width are first mapped to the closest resolutions using
#             `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to
#             the requested resolution. Useful for generating non-square images.
#         callback_on_step_end (`Callable`, *optional*):
#             A function that calls at the end of each denoising steps during the inference. The function is called
#             with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
#             callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
#             `callback_on_step_end_tensor_inputs`.
#         callback_on_step_end_tensor_inputs (`List`, *optional*):
#             The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
#             will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
#             `._callback_tensor_inputs` attribute of your pipeline class.
#         max_sequence_length (`int` defaults to `300`):
#             Maximum sequence length to use with the `prompt`.
#         complex_human_instruction (`List[str]`, *optional*):
#             Instructions for complex human attention:
#             https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55.

#     Examples:

#     Returns:
#         [`~pipelines.sana.pipeline_output.SanaPipelineOutput`] or `tuple`:
#             If `return_dict` is `True`, [`~pipelines.sana.pipeline_output.SanaPipelineOutput`] is returned,
#             otherwise a `tuple` is returned where the first element is a list with the generated images
#     """

#     if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
#         callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

#     # 1. Check inputs. Raise error if not correct
#     if use_resolution_binning:
#         if self.transformer.config.sample_size == 64:
#             aspect_ratio_bin = ASPECT_RATIO_2048_BIN
#         elif self.transformer.config.sample_size == 32:
#             aspect_ratio_bin = ASPECT_RATIO_1024_BIN
#         elif self.transformer.config.sample_size == 16:
#             aspect_ratio_bin = ASPECT_RATIO_512_BIN
#         else:
#             raise ValueError("Invalid sample size")
#         orig_height, orig_width = height, width
#         height, width = self.image_processor.classify_height_width_bin(height, width, ratios=aspect_ratio_bin)

#     self.check_inputs(
#         prompt,
#         height,
#         width,
#         callback_on_step_end_tensor_inputs,
#         negative_prompt,
#         prompt_embeds,
#         negative_prompt_embeds,
#         prompt_attention_mask,
#         negative_prompt_attention_mask,
#     )

#     self._guidance_scale = guidance_scale
#     self._interrupt = False

#     # 2. Default height and width to transformer
#     if prompt is not None and isinstance(prompt, str):
#         batch_size = 1
#     elif prompt is not None and isinstance(prompt, list):
#         batch_size = len(prompt)
#     else:
#         batch_size = prompt_embeds.shape[0]

#     device = self._execution_device

#     # 3. Encode input prompt
#     (
#         prompt_embeds,
#         prompt_attention_mask,
#         negative_prompt_embeds,
#         negative_prompt_attention_mask,
#     ) = self.encode_prompt(
#         prompt,
#         self.do_classifier_free_guidance,
#         negative_prompt=negative_prompt,
#         num_images_per_prompt=num_images_per_prompt,
#         device=device,
#         prompt_embeds=prompt_embeds,
#         negative_prompt_embeds=negative_prompt_embeds,
#         prompt_attention_mask=prompt_attention_mask,
#         negative_prompt_attention_mask=negative_prompt_attention_mask,
#         clean_caption=clean_caption,
#         max_sequence_length=max_sequence_length,
#         complex_human_instruction=complex_human_instruction,
#     )
#     if self.do_classifier_free_guidance:
#         prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
#         prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

#     # 4. Prepare timesteps
#     timesteps, num_inference_steps = retrieve_timesteps(
#         self.scheduler, num_inference_steps, device, timesteps, sigmas
#     )

#     # 5. Prepare latents.
#     latent_channels = self.transformer.config.in_channels
#     latents = self.prepare_latents(
#         batch_size * num_images_per_prompt,
#         latent_channels,
#         height,
#         width,
#         torch.float32,
#         device,
#         generator,
#         latents,
#     )

#     # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
#     extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

#     # 7. Denoising loop
#     num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
#     self._num_timesteps = len(timesteps)

#     with self.progress_bar(total=num_inference_steps) as progress_bar:
#         for i, t in enumerate(timesteps):
#             if self.interrupt:
#                 continue

#             latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
#             latent_model_input = latent_model_input.to(prompt_embeds.dtype)

#             # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
#             timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)

#             # predict noise model_output
#             noise_pred = self.transformer(
#                 latent_model_input,
#                 encoder_hidden_states=prompt_embeds,
#                 encoder_attention_mask=prompt_attention_mask,
#                 timestep=timestep,
#                 return_dict=False,
#             )[0]
#             noise_pred = noise_pred.float()

#             # perform guidance
#             if self.do_classifier_free_guidance:
#                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                 noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

#             # learned sigma
#             if self.transformer.config.out_channels // 2 == latent_channels:
#                 noise_pred = noise_pred.chunk(2, dim=1)[0]
#             else:
#                 noise_pred = noise_pred

#             # compute previous image: x_t -> x_t-1
#             latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

#             if callback_on_step_end is not None:
#                 callback_kwargs = {}
#                 for k in callback_on_step_end_tensor_inputs:
#                     callback_kwargs[k] = locals()[k]
#                 callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

#                 latents = callback_outputs.pop("latents", latents)
#                 prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
#                 negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

#             # call the callback, if provided
#             if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
#                 progress_bar.update()

#     if output_type == "latent":
#         image = latents
#     else:
#         latents = latents.to(self.vae.dtype)
#         image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
#         if use_resolution_binning:
#             image = self.image_processor.resize_and_crop_tensor(image, orig_width, orig_height)

#     if not output_type == "latent":
#         image = self.image_processor.postprocess(image, output_type=output_type)

#     # Offload all models
#     self.maybe_free_model_hooks()

#     if not return_dict:
#         return (image,)

#     return SanaPipelineOutput(images=image)

def chunked_scaled_dot_product_attention(query, key, value, chunk_size, attn_mask=None, dropout_p=0.0):
    batch_size, head, seq_len, d_model = query.size()
    attention_probs = torch.zeros(batch_size, head, seq_len, seq_len, device='cpu', dtype=query.dtype)
    for i in range(0, seq_len, chunk_size):
        for j in range(0, seq_len, chunk_size):
            start_query = i
            start_key = j
            end_query = min(i +chunk_size, seq_len)
            end_key = min(j +chunk_size, seq_len)

            query_chunk = query[:, :, start_query:end_query, :]
            key_chunk = key[:, :, start_key:end_key, :]

            # 创建单位矩阵
            chunk_identity_matrix = torch.eye(end_query - start_query, device=query.device, dtype=query.dtype)
            
            # 计算 attention_probs
            attention_probs_chunk = F.scaled_dot_product_attention(
                query=query_chunk,
                key=key_chunk,
                value=chunk_identity_matrix,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False
            )

            attention_probs[:, :, start_query:end_query, start_key:end_key] = attention_probs_chunk


        return attention_probs
# 这是分块计算的测试代码
# # 示例输入
# batch_size = 2
# head = 30
# seq_len = 1000  # 假设seq_len非常大
# d_model = 16  # Embedding dimension

# # 模拟的查询、键和值张量
# query = torch.rand(batch_size, head, seq_len, d_model, device='cuda', dtype=torch.float16, requires_grad=True)
# key = torch.rand(batch_size, head, seq_len, d_model, device='cuda', dtype=torch.float16, requires_grad=True)
# # value = torch.ones(batch_size, head, seq_len, d_model, device='cuda', dtype=torch.float16, requires_grad=True)
# identity_matrix = torch.eye(seq_len, device=query.device, dtype=query.dtype)
# attention_mask = None  # 如果有需要的掩码，可以在这里定义

# # 使用分块处理计算注意力权重
# chunk_size = 100  # 设定分块大小
# attention_probs = chunked_scaled_dot_product_attention(query, key, identity_matrix, chunk_size, attn_mask=attention_mask, dropout_p=0.0)


# attention_probs2= F.scaled_dot_product_attention(
#     query, key, identity_matrix, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
# )
# # 打印结果形状
# print("Attention_probs shape:", attention_probs.shape)

# print(torch.equal(attention_probs.detach().cpu(),attention_probs2.cpu()))



@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def FluxPipeline_call(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
):
    r"""
    Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
            will be used instead
        height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The height in pixels of the generated image. This is set to 1024 by default for the best results.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The width in pixels of the generated image. This is set to 1024 by default for the best results.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        timesteps (`List[int]`, *optional*):
            Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
            in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
            passed will be used. Must be in descending order.
        guidance_scale (`float`, *optional*, defaults to 7.0):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            If not provided, pooled text embeddings will be generated from `prompt` input argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        callback_on_step_end (`Callable`, *optional*):
            A function that calls at the end of each denoising steps during the inference. The function is called
            with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
            callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
            `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.
        max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

    Examples:

    Returns:
        [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
        is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
        images.
    """
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # handle guidance
    if self.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    # 6. Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
                ##################################################
                height=2 * (int(height) // (self.vae_scale_factor * 2)) // 2,
                ##################################################
            )[0]

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

    if output_type == "latent":
        image = latents

    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return FluxPipelineOutput(images=image)

@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def CogVideoXPipeline_call(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
    """
    Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        height (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
            The height in pixels of the generated image. This is set to 480 by default for the best results.
        width (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
            The width in pixels of the generated image. This is set to 720 by default for the best results.
        num_frames (`int`, defaults to `48`):
            Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
            contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
            num_seconds is 6 and fps is 8. However, since videos can be saved at any fps, the only condition that
            needs to be satisfied is that of divisibility mentioned above.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        timesteps (`List[int]`, *optional*):
            Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
            in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
            passed will be used. Must be in descending order.
        guidance_scale (`float`, *optional*, defaults to 7.0):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        num_videos_per_prompt (`int`, *optional*, defaults to 1):
            The number of videos to generate per prompt.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
            of a plain tuple.
        attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        callback_on_step_end (`Callable`, *optional*):
            A function that calls at the end of each denoising steps during the inference. The function is called
            with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
            callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
            `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.
        max_sequence_length (`int`, defaults to `226`):
            Maximum sequence length in encoded prompt. Must be consistent with
            `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

    Examples:

    Returns:
        [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] or `tuple`:
        [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] if `return_dict` is True, otherwise a
        `tuple`. When returning a tuple, the first element is a list with the generated images.
    """
    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
    width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
    num_frames = num_frames or self.transformer.config.sample_frames

    num_videos_per_prompt = 1

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds,
        negative_prompt_embeds,
    )
    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._interrupt = False

    # 2. Default call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt,
        negative_prompt,
        do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
    )
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
    self._num_timesteps = len(timesteps)

    # 5. Prepare latents
    latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

    # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
    patch_size_t = self.transformer.config.patch_size_t
    additional_frames = 0
    if patch_size_t is not None and latent_frames % patch_size_t != 0:
        additional_frames = patch_size_t - latent_frames % patch_size_t
        num_frames += additional_frames * self.vae_scale_factor_temporal

    latent_channels = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_videos_per_prompt,
        latent_channels,
        num_frames,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Create rotary embeds if required
    image_rotary_emb = (
        self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
        if self.transformer.config.use_rotary_positional_embeddings
        else None
    )

    latent_height = int(height/self.vae_scale_factor_spatial)
    latent_width = int(width/self.vae_scale_factor_spatial)
    latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

    # 8. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        # for DPM-solver++
        old_pred_original_sample = None
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            # predict noise model_output
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
                return_dict=False,
                ###################################################
                height=latent_height,
                width=latent_width,
                frames=latent_frames,
                ##################################################
            )[0]
            noise_pred = noise_pred.float()

            # perform guidance
            if use_dynamic_cfg:
                self._guidance_scale = 1 + guidance_scale * (
                    (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                )
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            else:
                latents, old_pred_original_sample = self.scheduler.step(
                    noise_pred,
                    old_pred_original_sample,
                    t,
                    timesteps[i - 1] if i > 0 else None,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )
            latents = latents.to(prompt_embeds.dtype)

            # call the callback, if provided
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()


    if not output_type == "latent":
        # Discard any padding frames that were added for CogVideoX 1.5
        latents = latents[:, additional_frames:]
        video = self.decode_latents(latents)
        video = self.video_processor.postprocess_video(video=video, output_type=output_type)
    else:
        video = latents

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (video,)

    return CogVideoXPipelineOutput(frames=video)


@torch.no_grad()
def CogVideoXImageToVideoPipeline_call(
        self,
        image: PipelineImageInput,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            image (`PipelineImageInput`):
                The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The width in pixels of the generated image. This is set to 720 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 8. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

        Examples:

        Returns:
            [`~pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            image=image,
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

        image = self.video_processor.preprocess(image, height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )

        latent_channels = self.transformer.config.in_channels // 2
        latents, image_latents = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Create ofs embeds if required
        ofs_emb = None if self.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)

        ###############################
        latent_height = int(height/self.vae_scale_factor_spatial)
        latent_width = int(width/self.vae_scale_factor_spatial)
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        ###############################

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                latent_image_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
                latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    ###################################################
                    height=latent_height,
                    width=latent_width,
                    frames=latent_frames,
                    ##################################################
                )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # Discard any padding frames that were added for CogVideoX 1.5
            latents = latents[:, additional_frames:]
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)


def CogVideoXTransformer3DModelForward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        height: int = None,
        width: int = None,
        frames: int = None,
    ):  
        
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape


        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks
        p = self.config.patch_size
        p_t = self.config.patch_size_t
        if p_t is None:
            frames = frames
        else:
            frames = frames // p_t

        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    ##########################################################################################
                    timestep=timestep, 
                    height= height // self.config.patch_size,
                    width = width // self.config.patch_size,
                    frames = frames
                    ##########################################################################################
                )

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


def CogVideoXBlockForward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        height: int = None,
        width: int = None,
        frames: int = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

    text_seq_length = encoder_hidden_states.size(1)

    # norm & modulate
    norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
        hidden_states, encoder_hidden_states, temb
    )

    # attention
    attn_hidden_states, attn_encoder_hidden_states = self.attn1(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
        ##################################################
        height=height,
        width=width,
        frames=frames,
        timestep=timestep,
        ##################################################
    )

    hidden_states = hidden_states + gate_msa * attn_hidden_states
    encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

    # norm & modulate
    norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
        hidden_states, encoder_hidden_states, temb
    )

    # feed-forward
    norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
    ff_output = self.ff(norm_hidden_states)

    hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
    encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

    return hidden_states, encoder_hidden_states


def attn_call(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        height: int = None,
        width: int = None,
        timestep: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        ####################################################################################################
        if hasattr(self, "store_attn_map"):
            self.attn_map = rearrange(attention_probs, 'b (h w) d -> b d h w', h=height)
            self.timestep = int(timestep.item())
        ####################################################################################################
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(attn_weight.device)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return torch.dropout(attn_weight, dropout_p, train=True) @ value, attn_weight


def attn_call2_0(
    self,
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    temb: Optional[torch.Tensor] = None,
    height: int = None,
    width: int = None,
    timestep: Optional[torch.Tensor] = None,
    *args,
    **kwargs,
) -> torch.Tensor:
    if len(args) > 0 or kwargs.get("scale", None) is not None:
        deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        deprecate("scale", "1.0.0", deprecation_message)

    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    # TODO: add support for attn.scale when we move to Torch 2.1
    ####################################################################################################
    if hasattr(self, "store_attn_map"):
        hidden_states, attention_probs = scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        self.attn_map = rearrange(
            attention_probs,
            'batch attn_head (h w) attn_dim -> batch attn_head h w attn_dim ',
            h=height
        ) # detach height*width
        self.timestep = int(timestep.item())
    else:
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
    ####################################################################################################

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim) # (b,attn_head,h*w,attn_dim) -> (b,h*w,attn_head*attn_dim)
    hidden_states = hidden_states.to(query.dtype)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states

def cogvideox_attn_call2_0(
    self,
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    ############################################################
    height: int = None,
    width: int = None,
    frames: int= None,
    timestep: Optional[torch.Tensor] = None,
    ############################################################
) -> torch.Tensor:

    text_seq_length = encoder_hidden_states.size(1)

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
    
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads
    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # Apply RoPE if needed
    if image_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb
        # .embeddings import apply_rotary_emb

        query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
        if not attn.is_cross_attention:
            key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)


    if hasattr(self, "store_attn_map"):
        # TODO:基于https://github.com/THUDM/CogVideo/issues/109
        # 来实现简单的attention map计算
        
        # # identity matrix在计算attention_map * value的时候 cuda out of memeory
        # guidance scale选择取出只包括文本prompt的query然后分别对不同的head求attention然后求mean。

        attention_probs_list = []
        # for i in range(query.shape[1]):
        #     query_attention= query[-1][i]
        #     key_attention = key[-1][i]
        #     identity_matrix = torch.eye(query_attention.shape[-2], device=query_attention.device, dtype=query_attention.dtype)
        #     attention_probs_temp=  F.scaled_dot_product_attention(
        #         query_attention, key_attention, identity_matrix, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        #     )
        #     attention_probs_list.append(attention_probs_temp.cpu())
        #     del query_attention,key_attention,identity_matrix, attention_probs_temp

        # attention_probs = torch.stack(attention_probs_list)    
        # attention_probs = torch.mean(attention_probs, dim=0)


        for i in range(0, query.shape[1], 16):
            query_attention= query[-1][i:i+16]
            key_attention = key[-1][i:i+16]
            identity_matrix = torch.eye(query_attention.shape[-2], device=query_attention.device, dtype=query_attention.dtype)
            attention_probs_temp=  F.scaled_dot_product_attention(
                query_attention, key_attention, identity_matrix, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            attention_probs_list.append(attention_probs_temp.cpu())
            del query_attention,key_attention,identity_matrix, attention_probs_temp
        
        attention_probs = torch.mean(torch.cat(attention_probs_list), dim=0)
        
        hidden_states= F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        
        # TODO:
        # 计算 video-to-video attention 并重排
        video_to_video = attention_probs[text_seq_length:, text_seq_length:]

        self.attn_map_v2v = rearrange(
            video_to_video,
            '(frames1 height1 width1) (frames2 height2 width2) -> frames1 height1 width1 frames2 height2 width2',
            frames1=frames,
            height1=height,
            width1=width,
            frames2=frames,
            height2=height,
            width2=width,
        )
        self.attn_map_v2v = self.attn_map_v2v.mean(dim=(1, 2, 4, 5))  # 在空间维度求平均
        
        # 清理不需要的变量避免OOM
        del video_to_video, attention_probs

        # 保存时间步长
        self.timestep = timestep[0].cpu().item()

    else:
        hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    encoder_hidden_states, hidden_states = hidden_states.split(
        [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
    )
    return hidden_states, encoder_hidden_states
