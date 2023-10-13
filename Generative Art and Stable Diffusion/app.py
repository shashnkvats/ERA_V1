from base64 import b64encode

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from huggingface_hub import notebook_login

# For video display:
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image
from torch import autocast
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging
import os

from config import RADIO_OPTIONS, MAPPING

import streamlit as st


torch.manual_seed(1)
if not (Path.home()/'.cache/huggingface'/'token').exists(): notebook_login()

# Supress some unnecessary warnings when loading the CLIPTextModel
logging.set_verbosity_error()

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
# if "mps" == torch_device: os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"


# Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

# The noise scheduler
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

# To the GPU we go!
vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device)



import streamlit as st

st.markdown('<h1 style="text-align: center;">Dreamstream</h1>', unsafe_allow_html=True)

col1, col2 = st.columns([3,1])
prompt = col1.text_input("Imagine...")
dropdown_value = col2.selectbox("Style", RADIO_OPTIONS, index=0)

prompt += prompt + f" in style of {dropdown_value}"
prompt = prompt.lower()

generate = st.button("Generate")

if generate:

    def pil_to_latent(input_im):
        # Single image -> single latent in a batch (so size 1, 4, 64, 64)
        with torch.no_grad():
            latent = vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(torch_device)*2-1) # Note scaling
        return 0.18215 * latent.latent_dist.sample()


    def latents_to_pil(latents):
        # bath of latents -> list of images
        latents = (1 / 0.18215) * latents
        with torch.no_grad():
            image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images


    def set_timesteps(scheduler, num_inference_steps):
        scheduler.set_timesteps(num_inference_steps)
        scheduler.timesteps = scheduler.timesteps.to(torch.float32)

    def get_output_embeds(input_embeddings):
        # CLIP's text model uses causal mask, so we prepare it here:
        bsz, seq_len = input_embeddings.shape[:2]
        causal_attention_mask = text_encoder.text_model._build_causal_attention_mask(bsz, seq_len, dtype=input_embeddings.dtype)

        # Getting the output embeddings involves calling the model with passing output_hidden_states=True
        # so that it doesn't just return the pooled final predictions:
        encoder_outputs = text_encoder.text_model.encoder(
            inputs_embeds=input_embeddings,
            attention_mask=None, # We aren't using an attention mask so that can be None
            causal_attention_mask=causal_attention_mask.to(torch_device),
            output_attentions=None,
            output_hidden_states=True, # We want the output embs not the final output
            return_dict=None,
        )

        # We're interested in the output hidden state only
        output = encoder_outputs[0]

        # There is a final layer norm we need to pass these through
        output = text_encoder.text_model.final_layer_norm(output)

        # And now they're ready!
        return output
    
    def saturation_loss(images):
        red_variance = images[:, 0].var()
        green_variance = images[:, 1].var()
        blue_variance = images[:, 2].var()
        return -(red_variance + green_variance + blue_variance)  # Negative because we want to maximize variance
    

    def generate_with_embs(text_embeddings, use_saturation_loss=True):
        height = 512                        # default height of Stable Diffusion
        width = 512                         # default width of Stable Diffusion
        num_inference_steps = 50           # Number of denoising steps
        guidance_scale = 8               # Scale for classifier-free guidance
        generator = torch.manual_seed(32)   # Seed generator to create the inital latent noise
        batch_size = 1
        saturation_loss_scale = 200 

        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Prep Scheduler
        set_timesteps(scheduler, num_inference_steps)

        # Prep latents
        latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
        )
        latents = latents.to(torch_device)
        latents = latents * scheduler.init_noise_sigma

        # Loop
        for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = scheduler.sigmas[i]
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            print("Shape of noise_pred:", noise_pred.shape)

            # perform CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            #### ADDITIONAL GUIDANCE ###
            # if i%5 == 0:
            if use_saturation_loss and i%5 == 0:
                # Requires grad on the latents
                latents = latents.detach().requires_grad_()

                # Get the predicted x0:
                latents_x0 = latents - sigma * noise_pred
                # latents_x0 = scheduler.step(noise_pred, t, latents).pred_original_sample

                # Decode to image space
                denoised_images = vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5 # range (0, 1)

                # Calculate loss
                loss = saturation_loss(denoised_images) * saturation_loss_scale

                # Occasionally print it out
                if i%10==0:
                    print(i, 'loss:', loss.item())

                # Get gradient
                cond_grad = torch.autograd.grad(loss, latents)[0]

                # Modify the latents based on this gradient
                latents = latents.detach() - cond_grad * sigma**2

            # Now step with scheduler
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        return latents_to_pil(latents)[0]


        
    illustration_embed = torch.load(list(MAPPING[dropdown_value].values())[0])
    
    # Tokenize
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    input_ids = text_input.input_ids.to(torch_device)

    # Get token embeddings
    token_emb_layer = text_encoder.text_model.embeddings.token_embedding
    token_embeddings = token_emb_layer(input_ids)

    # The new embedding. Which is now a mixture of the token embeddings for 'puppy' and 'skunk'
    replacement_token_embedding = illustration_embed[list(MAPPING[dropdown_value].keys())[0]].to(torch_device)

    # Insert this into the token embeddings
    token_embeddings[0, torch.where(input_ids[0]==6829)[0]] = replacement_token_embedding.to(torch_device)

    # Combine with pos embs
    pos_emb_layer = text_encoder.text_model.embeddings.position_embedding
    position_ids = text_encoder.text_model.embeddings.position_ids[:, :77]
    position_embeddings = pos_emb_layer(position_ids)
    input_embeddings = token_embeddings + position_embeddings

    #  Feed through to get final output embs
    modified_output_embeddings = get_output_embeds(input_embeddings)

    col7, col8 = st.columns([1,1])
    # Generate an image with saturation_loss
    with_loss_image = generate_with_embs(modified_output_embeddings, use_saturation_loss=True)
    col7.image(with_loss_image, caption="With Saturation Loss", use_column_width=True, channels="RGB")

    # Generate an image without saturation_loss
    without_loss_image = generate_with_embs(modified_output_embeddings, use_saturation_loss=False)
    col8.image(without_loss_image, caption="Without Saturation Loss", use_column_width=True, channels="RGB")

