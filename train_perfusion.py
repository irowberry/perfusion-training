import torch
from perfusion_pytorch import Rank1EditModule, save_load, EmbeddingWrapper
from perfusion_pytorch.optimizer import get_finetune_optimizer, get_finetune_parameters
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from PIL import Image
from torch import nn
from diffusers.models.attention_processor import Attention
from transformers import CLIPTextModel, CLIPTokenizer
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from diffusers.image_processor import VaeImageProcessor

device = "mps"

class PerfusionAttnProcessor(nn.Module):
    r"""
    Processor for implementing attention for the Perfusion method.

    Args:
        train_kv (`bool`, defaults to `True`):
            Whether to newly train the key and value matrices corresponding to the text features.
        train_q_out (`bool`, defaults to `False`):
            Whether to newly train query matrices corresponding to the latent image features.
        hidden_size (`int`, *optional*, defaults to `None`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*, defaults to `None`):
            The number of channels in the `encoder_hidden_states`.
        out_bias (`bool`, defaults to `True`):
            Whether to include the bias parameter in `train_q_out`.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
    """

    def __init__(
        self,
        train_kv=True,
        hidden_size=None,
        cross_attention_dim=None,
        out_bias=True,
        dropout=0.0,
    ):
        super().__init__()
        self.train_kv = train_kv
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        # `_custom_diffusion` id for easy serialization and loading.
        if self.train_kv: # Use Rank1EditModule
            self.to_k_custom_diffusion = Rank1EditModule(nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False), is_key_proj=True)
            self.to_v_custom_diffusion = Rank1EditModule(nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False))

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        # accept additional parameters for Rank1EditModule during forward pass
        concept_indices = cross_attention_kwargs['concept_indices']
        text_enc_with_superclass = cross_attention_kwargs['text_enc_with_superclass']
        
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states.to(attn.to_q.weight.dtype))

        if encoder_hidden_states is None:
            crossattn = False
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if self.train_kv:
            key = self.to_k_custom_diffusion(encoder_hidden_states.to(self.to_k_custom_diffusion.weight.dtype), 
                                             concept_indices=concept_indices,
                                             text_enc_with_superclass=text_enc_with_superclass)
            value = self.to_v_custom_diffusion(encoder_hidden_states.to(self.to_v_custom_diffusion.weight.dtype), 
                                               concept_indices=concept_indices,
                                               text_enc_with_superclass=text_enc_with_superclass)
            key = key.to(attn.to_q.weight.dtype)
            value = value.to(attn.to_q.weight.dtype)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        if crossattn:
            detach = torch.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :] * 0.0
            key = detach * key + (1 - detach) * key.detach()
            value = detach * value + (1 - detach) * value.detach()

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class PerfusionModel(nn.Module):
    def __init__(self, unet, clip_model, tokenizer, superclass_string):
        super(PerfusionModel, self).__init__()
        self.unet = unet
        self.clip_model = clip_model
        train_kv = True
        self.superclass_string = superclass_string
        self.custom_diffusion_attn_procs = {}
        
        # self.unet.requires_grad_(False)
        # self.clip_model.requires_grad_(False)
        self.l_tokenizer = lambda x: tokenizer(x, 
                                               padding="max_length", 
                                               max_length=tokenizer.model_max_length, 
                                               truncation=True,
                                               return_tensors='pt')['input_ids']
        self.l_encoder = lambda x: self.clip_model.text_model.encoder(x)[0]
        self.wrapped_embeds = EmbeddingWrapper(self.clip_model.get_input_embeddings(), 
                                          superclass_string=self.superclass_string, 
                                          tokenize=self.l_tokenizer,
                                          tokenizer_pad_id=49407,
                                          )        
        
        # self.wrapped_clip = OpenClipEmbedWrapper(self.clip_model, superclass_string=self.superclass_string)
        attention_class = PerfusionAttnProcessor

        # Here we replace all the K,V matricies found in stable diffusion u-net with the ones wrapped in the Rank1EditModule
        st = self.unet.state_dict()
        for name, _ in self.unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_custom_diffusion.weight": st[layer_name + ".to_k.weight"],
                "to_v_custom_diffusion.weight": st[layer_name + ".to_v.weight"],
            }
            if cross_attention_dim is not None:
                self.custom_diffusion_attn_procs[name] = attention_class(
                    train_kv=train_kv,
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                ).to(self.unet.device)
                self.custom_diffusion_attn_procs[name].load_state_dict(weights, strict=False)
            else:
                self.custom_diffusion_attn_procs[name] = attention_class(
                    train_kv=False,
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                )
        del st
        self.unet.set_attn_processor(self.custom_diffusion_attn_procs)
        
    def forward(self, noisy_latents, timesteps, text):
        # embeds_with_new_concept, embeds_with_superclass, embed_mask, concept_indices = self.wrapped_embeds(text, clip_transformer_fn=self.l_encoder)
        # enc_with_new_concept = self.clip_model.text_model.encoder(embeds_with_new_concept)[0]
        # if embeds_with_superclass is not None:
        #     enc_with_superclass = self.clip_model.text_model.encoder(embeds_with_superclass)[0]
        # else:
        #     enc_with_superclass = embeds_with_superclass
        
        enc_with_new_concept, enc_with_superclass, embed_mask, concept_indices = self.wrapped_embeds(text, clip_transformer_fn=self.l_encoder)
            
        if self.training:
            out = self.unet(noisy_latents, 
                            timesteps, 
                            enc_with_new_concept,
                            cross_attention_kwargs={
                                'text_enc_with_superclass': enc_with_superclass,
                                'concept_indices': concept_indices,
                                },
                            attention_mask=embed_mask,
                            )
        else:
            uncond_ids = self.l_tokenizer([""]*enc_with_new_concept.shape[0])
            uncond_enc = self.clip_model(uncond_ids.to(noisy_latents.device))[0]
            out = self.unet(noisy_latents, 
                            timesteps, 
                            torch.cat([uncond_enc, enc_with_new_concept]),
                            cross_attention_kwargs={
                                'text_enc_with_superclass': enc_with_superclass,
                                'concept_indices': concept_indices,
                                },
                            attention_mask=torch.cat([embed_mask, torch.ones_like(embed_mask, dtype=embed_mask.dtype)])
                            )
        return out.sample

def open_and_prepare_images(dir):
    image_paths = os.listdir(dir)
    image_paths = [dir + "/" + f for f in image_paths]
    images = []
    img_proc = VaeImageProcessor()
    for path in image_paths:
        image = Image.open(path)
        image = img_proc.resize(image, 512, 512)
        # # image = image.convert("RGB")
        # image = image.resize((512, 512))
        # image = np.array(image).astype(np.uint8)
        # image = (image / 127.5 - 1.0).astype(np.float32)
        # images.append(torch.from_numpy(image).permute(2, 0, 1))
        images.append(image)
    images = img_proc.pil_to_numpy(images)
    images = img_proc.numpy_to_pt(images)
    return images.to(device)

def train(dataloader, model, num_steps, vae, noise_scheduler, opt):
    i = 0
    pbar = tqdm(total=num_steps)
    while i <= num_steps:
        for batch in dataloader:
            opt.zero_grad()
            images, text = batch
            text = list(text)
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            noise = torch.rand_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            out = model(noisy_latents, timesteps, text)
            loss = F.mse_loss(out, noise, reduction="mean")
            loss.backward()
            opt.step()
            pbar.update(len(images))
            pbar.set_description(f"Loss: {loss.item()/len(images)}")
            i += len(images)
            
# def inference(prompt, model, scheduler, num_inference_steps, num_images=4, guidance_scale = 7.5):
#     scheduler.set_timesteps(num_inference_steps)
#     latents = torch.randn((num_images, model.unet.config.in_channels, 512 // 8, 512 // 8))
#     latents = latents * scheduler.init_noise_sigma
#     latents = latents.to(device)
#     for t in tqdm(scheduler.timesteps):
#         latent_model_input = torch.cat([latents]*2)
#         latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
#         with torch.no_grad():
#             t = t.to(device)
#             noise_pred = model(latent_model_input, t, [prompt]*num_images)
        
#         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#         noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
#         latents = scheduler.step(noise_pred, t, latents).prev_sample
        
#     latents = 1 / 0.18215 * latents
#     with torch.no_grad():
#         image = vae.decode(latents).sample

#     image = (image / 2 + 0.5).clamp(0, 1)
#     image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
#     images = (image * 255).round().astype("uint8")
#     pil_images = [Image.fromarray(image) for image in images]
    
#     return pil_images

def inference(prompts, pipe, model):
    # embeds_with_new_concept, embeds_with_superclass, embed_mask, concept_indices = model.wrapped_embeds(prompts, clip_transformer_fn=model.l_encoder)
    # enc_with_new_concept = model.clip_model.text_model.encoder(embeds_with_new_concept)[0]
    # if embeds_with_superclass is not None:
    #     enc_with_superclass = model.clip_model.text_model.encoder(embeds_with_superclass)[0]
    # else:
    #     enc_with_superclass = embeds_with_superclass
    enc_with_new_concept, enc_with_superclass, embed_mask, concept_indices = model.wrapped_embeds(prompts, clip_transformer_fn=model.l_encoder)

    pipe.unet = model.unet
    pipe.text_encoder = model.clip_model
    pipe.to(device)
    images = pipe(prompt_embeds=enc_with_new_concept,
                 num_inference_steps=30, 
                 guidance_scale=7.5, 
                 cross_attention_kwargs={"concept_indices": concept_indices, "text_enc_with_superclass": enc_with_superclass}).images
    return images
    
    
if __name__ == "__main__":
    model_name = "runwayml/stable-diffusion-v1-5"
    
    vae = AutoencoderKL.from_pretrained(model_name, subfolder='vae').to(device)
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
    # noise_scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    # clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained='laion2B-s32B-b82K')
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer')
    clip_model = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder')
    images = open_and_prepare_images("./dog")
    
    dl = DataLoader(list(zip(images, ["a photo of a dog sitting"]*len(images))), 2, True)
    superclass_string = "dog"
    
    pipe = StableDiffusionPipeline.from_pretrained(model_name, requires_safety_checker=False)
    pipe.scheduler = noise_scheduler
    perfusion_model = PerfusionModel(unet, clip_model, tokenizer, superclass_string).to(device).requires_grad_(False)
    # for param in get_finetune_parameters(perfusion_model):
    #     param.requires_grad = True
    # opt = get_finetune_optimizer(perfusion_model)
    
    # perfusion_model.train()
    # train(dl, perfusion_model, 1200, vae, noise_scheduler, opt)
    # save_load.save(perfusion_model, 'dog_concept.pt')
    save_load.load(perfusion_model, 'dog_concept.pt')
    perfusion_model.eval()
    with torch.no_grad():
        # images = inference('A photo of a dog', perfusion_model, noise_scheduler, 30, 1)
        images = inference(['a photo of a dog wearing sunglasses, high quality'], pipe, perfusion_model)
    # images = inference("a photo of a dog with sunglasses on", perfusion_model, noise_scheduler, 100)
    
    
    for i, img in enumerate(images):
        img.save(f"image{i}.jpg")
    