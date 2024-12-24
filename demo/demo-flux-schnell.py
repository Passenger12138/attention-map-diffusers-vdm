import torch
from diffusers import FluxPipeline
from attention_map_diffusers import (
    attn_maps,
    init_pipeline,
    save_attention_maps
)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
)
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
pipe.to('cuda')

##### 1. Replace modules and Register hook #####
pipe = init_pipeline(pipe)
################################################

# recommend not using batch operations for sd3, as cpu memory could be exceeded.
prompts = [
    # "A photo of a puppy wearing a hat.",
    "A capybara holding a sign that reads Hello World.",
]

images = pipe(
    prompts,
    num_inference_steps=15,
    guidance_scale=4.5,
).images

for batch, image in enumerate(images):
    image.save(f'{batch}-flux-schnell.png')

##### 2. Process and Save attention map #####
# set unconditional to False for flux
save_attention_maps(attn_maps, pipe.tokenizer, prompts, base_dir='attn_maps-flux-schnell', unconditional=False)
#############################################
