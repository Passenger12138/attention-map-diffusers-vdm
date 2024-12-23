import torch
from diffusers import SanaPipeline
from attention_map_diffusers import (
    attn_maps,
    init_pipeline,
    save_attention_maps
)

pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
    variant="fp16",
    torch_dtype=torch.float16,
)
pipe.to("cuda")

pipe.vae.to(torch.bfloat16)
pipe.text_encoder.to(torch.bfloat16)

##### 1. Replace modules and Register hook #####
# TODO: not implemented yet.
pipe = init_pipeline(pipe)
################################################

prompts = [
    "a cyberpunk cat with a neon sign that says 'Sana'",
    # "A capybara holding a sign that reads Hello World.",
]
images = pipe(
    prompt=prompts,
    height=1024,
    width=1024,
    guidance_scale=5.0,
    num_inference_steps=20,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images

for batch, image in enumerate(images):
    image.save(f'{batch}-sana.png')

##### 2. Process and Save attention map #####
save_attention_maps(attn_maps, pipe.tokenizer, prompts, base_dir='attn_maps-sana', unconditional=True)
#############################################