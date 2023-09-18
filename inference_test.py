from perfusion_pytorch.embedding import OpenClipEmbedWrapper, EmbeddingWrapper
from perfusion_pytorch.save_load import load
from perfusion_pytorch import Rank1EditModule
from train_perfusion import PerfusionModel
import torch
from diffusers import StableDiffusionPipeline

device = "cuda:0"
generator = torch.Generator(device="cuda").manual_seed(12345)
prompts = ["photo of a dog, high quality"]
superclass_string = 'dog'

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", requires_safety_checker=False)
pipe.to(device)
image = pipe(prompts[0], num_inference_steps=30, guidance_scale=7.5,generator=generator).images[0]
image.save("test-before.png")

pipe.to("cpu")
perfusion_model = PerfusionModel(pipe.unet, pipe.text_encoder, pipe.tokenizer, superclass_string).to(device).requires_grad_(False)
load(perfusion_model, 'dog_concept.pt')
perfusion_model.eval()

pipe.text_encoder.to(device)
# wrapped_clip_with_new_concept = OpenClipEmbedWrapper(
#     pipe.text_encoder,
#     tokenizer = pipe.tokenizer,
#     superclass_string = superclass_string
# )

text_enc, superclass_enc, mask, indices = perfusion_model.wrapped_embeds(prompts)

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", requires_safety_checker=False)
pipe.unet = perfusion_model.unet
pipe.text_encoder = perfusion_model.clip_model
pipe.to(device)

image = pipe(prompts[0], num_inference_steps=30, guidance_scale=7.5,generator=generator, cross_attention_kwargs={"concept_indices": indices, "text_enc_with_superclass": superclass_enc}).images[0]
image.save("test-after.png")