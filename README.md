# perfusion-training
Rough code for training Stable Diffusion using the Perfusion method. 
The goal is for it to be usable with diffusion models found on Hugging Face. 
An important note is that the diffusers package has a TODO that is important for this script. 
Clone diffusers and install using `pip install -e .`, in diffusers/models/attention_processor.py in the `Attention` class, and `prepare_attention_mask` method, there is a TODO that says something about stable-diffusion-pipelines (line 408). You just need to change attention_mask to be  `F.pad(attention_mask, (0, target_length - current_length), value=0.0)` in that if branch