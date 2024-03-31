import argparse
import datetime
import logging
import inspect
import math
import os
import random
import gc
import copy

from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import diffusers
import transformers

from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from models.unet.unet_3d_condition import UNet3DConditionModel
from diffusers.models import AutoencoderKL
from diffusers import DDIMScheduler, TextToVideoSDPipeline
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from diffusers.models.attention import BasicTransformerBlock

from transformers import CLIPTextModel, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPEncoder
from einops import rearrange, repeat
from utils.ddim_utils import inverse_video

import imageio
import numpy as np

from dataset import *
from loss import *
from noise_init import *

from utils.func_utils import *


logger = get_logger(__name__, log_level="INFO")

def log_validation(accelerator, config, batch, global_step, text_prompt, unet, text_encoder, vae, output_dir):
    with accelerator.autocast():
        unet.eval()
        text_encoder.eval()
        unet_and_text_g_c(unet, text_encoder, False, False)

        # handle spatial lora
        if 'spatial_scale' in config.val.keys():
            loras = extract_lora_child_module(unet, target_replace_module=["Transformer2DModel"])
            for lora_i in loras:
                lora_i.scale = config.val.spatial_scale
        
        # preset_noise = batch['inversion_noise']

        pipeline = TextToVideoSDPipeline.from_pretrained(
            config.model.pretrained_model_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet
        )

        diffusion_scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.scheduler = diffusion_scheduler

        prompt_list = text_prompt if len(config.val.prompt) <= 0 else config.val.prompt
        for seed in range(config.val.seed_range[0], config.val.seed_range[1]):

            config.noise_init.seed = seed
            # handle different noise initialization strategy
            init_func_name = f'initialize_noise_with_{config.noise_init.type}'
            # Assuming config.dataset is a DictConfig object
            init_params_dict = OmegaConf.to_container(config.noise_init, resolve=True)
            # Remove the 'type' key
            init_params_dict.pop('type', None)  # 'None' ensures no error if 'type' key doesn't exist

            init_func_to_call = globals().get(init_func_name)
            init_noise = init_func_to_call(batch['inversion_noise'], **init_params_dict)

            for prompt in prompt_list:
                file_name = f"{prompt.replace(' ', '_')}_seed_{seed}.mp4"
                file_path = f"{output_dir}/samples_{global_step}/"
                if not os.path.exists(file_path):
                    os.makedirs(file_path)

                with torch.no_grad():
                    video_frames = pipeline(
                        prompt=config.val.prompt_prefix + prompt,
                        negative_prompt=config.val.negative_prompt,
                        width=config.val.width,
                        height=config.val.height,
                        num_frames=config.val.num_frames,
                        num_inference_steps=config.val.num_inference_steps,
                        guidance_scale=config.val.guidance_scale,
                        latents=init_noise,
                    ).frames[0]
                export_to_video(video_frames, os.path.join(file_path, file_name), config.dataset.fps)
                logger.info(f"Saved a new sample to {os.path.join(file_path, file_name)}")
        del pipeline
        torch.cuda.empty_cache()

def create_logging(logging, logger, accelerator):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

def accelerate_set_verbose(accelerator):
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

def export_to_video(video_frames, output_video_path, fps):
    video_writer = imageio.get_writer(output_video_path, fps=fps)
    for img in video_frames:
        video_writer.append_data(np.array(img))
    video_writer.close()

def create_output_folders(output_dir, config):
    # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = os.path.join(output_dir)

    os.makedirs(out_dir, exist_ok=True)
    # os.makedirs(f"{out_dir}/samples", exist_ok=True)
    OmegaConf.save(config, os.path.join(out_dir, 'config.yaml'))

    return out_dir

def load_primary_models(pretrained_model_path):
    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    return noise_scheduler, tokenizer, text_encoder, vae, unet

def unet_and_text_g_c(unet, text_encoder, unet_enable, text_enable):
    unet._set_gradient_checkpointing(value=unet_enable)
    text_encoder._set_gradient_checkpointing(CLIPEncoder, value=text_enable)

def freeze_models(models_to_freeze):
    for model in models_to_freeze:
        if model is not None: model.requires_grad_(False)

def is_attn(name):
    return ('attn1' or 'attn2' == name.split('.')[-1])

def set_processors(attentions):
    for attn in attentions: attn.set_processor(AttnProcessor2_0())

def set_torch_2_attn(unet):
    optim_count = 0

    for name, module in unet.named_modules():
        if is_attn(name):
            if isinstance(module, torch.nn.ModuleList):
                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        set_processors([m.attn1, m.attn2])
                        optim_count += 1
    if optim_count > 0:
        print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")

def handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet):
    try:
        is_torch_2 = hasattr(F, 'scaled_dot_product_attention')
        enable_torch_2 = is_torch_2 and enable_torch_2_attn

        if enable_xformers_memory_efficient_attention and not enable_torch_2:
            if is_xformers_available():
                from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
                unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if enable_torch_2:
            set_torch_2_attn(unet)

    except:
        print("Could not enable memory efficient attention for xformers or Torch 2.0.")

def negate_params(name, negation):
    # We have to do this if we are co-training with LoRA.
    # This ensures that parameter groups aren't duplicated.
    if negation is None: return False
    for n in negation:
        if n in name and 'temp' not in name:
            return True
    return False

def is_mixed_precision(accelerator):
    weight_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16

    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    return weight_dtype

def cast_to_gpu_and_type(model_list, accelerator, weight_dtype):
    for model in model_list:
        if model is not None: model.to(accelerator.device, dtype=weight_dtype)

def handle_cache_latents(
        should_cache,
        output_dir,
        train_dataloader,
        train_batch_size,
        vae,
        unet,
        pretrained_model_path,
        cached_latent_dir=None,
):
    # Cache latents by storing them in VRAM.
    # Speeds up training and saves memory by not encoding during the train loop.
    if not should_cache: return None
    vae.to('cuda', dtype=torch.float16)
    vae.enable_slicing()

    pipe = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_path,
        vae=vae,
        unet=copy.deepcopy(unet).to('cuda', dtype=torch.float16)
    )
    pipe.text_encoder.to('cuda', dtype=torch.float16)

    cached_latent_dir = (
        os.path.abspath(cached_latent_dir) if cached_latent_dir is not None else None
    )

    if cached_latent_dir is None:
        cache_save_dir = f"{output_dir}/cached_latents"
        os.makedirs(cache_save_dir, exist_ok=True)

        for i, batch in enumerate(tqdm(train_dataloader, desc="Caching Latents.")):
            save_name = f"cached_{i}"
            full_out_path = f"{cache_save_dir}/{save_name}.pt"

            pixel_values = batch['pixel_values'].to('cuda', dtype=torch.float16)
            batch['latents'] = tensor_to_vae_latent(pixel_values, vae)

            batch['inversion_noise'] = inverse_video(pipe, batch['latents'], 50)
            for k, v in batch.items(): batch[k] = v[0]

            torch.save(batch, full_out_path)
            del pixel_values
            del batch

            # We do this to avoid fragmentation from casting latents between devices.
            torch.cuda.empty_cache()
    else:
        cache_save_dir = cached_latent_dir

    return torch.utils.data.DataLoader(
        CachedDataset(cache_dir=cache_save_dir),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0
    )

def enforce_zero_terminal_snr(betas):
    """
    Corrects noise in diffusion schedulers.
    From: Common Diffusion Noise Schedules and Sample Steps are Flawed
    https://arxiv.org/pdf/2305.08891.pdf
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (
            alphas_bar_sqrt_0 - alphas_bar_sqrt_T
    )

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas

def should_sample(global_step, validation_steps, validation_data):
    return global_step % validation_steps == 0 and validation_data.sample_preview

def save_pipe(
        path,
        global_step,
        accelerator,
        unet,
        text_encoder,
        vae,
        output_dir,
        is_checkpoint=False,
        save_pretrained_model=True,
        **extra_params
):
    if is_checkpoint:
        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = output_dir

    # Save the dtypes so we can continue training at the same precision.
    u_dtype, t_dtype, v_dtype = unet.dtype, text_encoder.dtype, vae.dtype

    # Copy the model without creating a reference to it. This allows keeping the state of our lora training if enabled.
    unet_out = copy.deepcopy(accelerator.unwrap_model(unet.cpu(), keep_fp32_wrapper=False))
    text_encoder_out = copy.deepcopy(accelerator.unwrap_model(text_encoder.cpu(), keep_fp32_wrapper=False))

    pipeline = TextToVideoSDPipeline.from_pretrained(
        path,
        unet=unet_out,
        text_encoder=text_encoder_out,
        vae=vae,
    ).to(torch_dtype=torch.float32)

    lora_managers_spatial = extra_params.get('lora_managers_spatial', [None])
    lora_manager_spatial = lora_managers_spatial[-1]
    if lora_manager_spatial is not None:
        lora_manager_spatial.save_lora_weights(model=copy.deepcopy(pipeline), save_path=save_path+'/spatial', step=global_step)

    save_motion_embeddings(unet_out, os.path.join(save_path, 'motion_embed.pt'))

    if save_pretrained_model:
        pipeline.save_pretrained(save_path)

    if is_checkpoint:
        unet, text_encoder = accelerator.prepare(unet, text_encoder)
        models_to_cast_back = [(unet, u_dtype), (text_encoder, t_dtype), (vae, v_dtype)]
        [x[0].to(accelerator.device, dtype=x[1]) for x in models_to_cast_back]

    logger.info(f"Saved model at {save_path} on step {global_step}")

    del pipeline
    del unet_out
    del text_encoder_out
    torch.cuda.empty_cache()
    gc.collect()

def main(config):
    # Initialize the Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        mixed_precision=config.train.mixed_precision,
        log_with=config.train.logger_type,
        project_dir=config.train.output_dir
    )

    # Create output directories and set up logging
    if accelerator.is_main_process:
        output_dir = create_output_folders(config.train.output_dir, config)
    create_logging(logging, logger, accelerator)
    accelerate_set_verbose(accelerator)

    # Load primary models
    noise_scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(config.model.pretrained_model_path)
    freeze_models([vae, text_encoder])
    handle_memory_attention(config.train.enable_xformers_memory_efficient_attention, config.train.enable_torch_2_attn, unet)

    train_dataloader, train_dataset = prepare_data(config, tokenizer)

    # Handle latents caching
    cached_data_loader = handle_cache_latents(
        config.train.cache_latents,
        output_dir,
        train_dataloader,
        config.train.train_batch_size,
        vae,
        unet,
        config.model.pretrained_model_path,
        config.train.cached_latent_dir,
    )
    if cached_data_loader is not None:
        train_dataloader = cached_data_loader

    # Prepare parameters and optimization
    params, extra_params = prepare_params(unet, config, train_dataset)
    optimizers, lr_schedulers = prepare_optimizers(params, config, **extra_params)

    
    # Prepare models and data for training
    unet, optimizers, train_dataloader, lr_schedulers, text_encoder = accelerator.prepare(
        unet, optimizers, train_dataloader, lr_schedulers, text_encoder
    )

    # Additional model setups
    unet_and_text_g_c(unet, text_encoder, config.train.gradient_checkpointing, config.train.text_encoder_gradient_checkpointing)
    vae.enable_slicing()

    # Setup for mixed precision training
    weight_dtype = is_mixed_precision(accelerator)
    cast_to_gpu_and_type([text_encoder, vae], accelerator, weight_dtype)

    # Recalculate training steps and epochs
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.train.gradient_accumulation_steps)
    num_train_epochs = math.ceil(config.train.max_train_steps / num_update_steps_per_epoch)

    # Initialize trackers and store configuration
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    # Train!
    total_batch_size = config.train.train_batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.train.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, config.train.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    
    for epoch in range(first_epoch, num_train_epochs):
        train_loss_temporal = 0.0

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if config.train.resume_from_checkpoint and epoch == first_epoch and step < config.train.resume_step:
                if step % config.train.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet), accelerator.accumulate(text_encoder):

                text_prompt = batch['text_prompt'][0]

                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                    
                with accelerator.autocast():
                    if global_step == 0:
                        unet.train()
                        
                    loss_func_name = f'{config.loss.type}Loss'
                    loss_func_to_call = globals().get(loss_func_name)

                    loss_temporal, train_loss_temporal = loss_func_to_call(
                        train_loss_temporal,
                        accelerator,
                        optimizers,
                        lr_schedulers,
                        unet,
                        vae,
                        text_encoder,
                        noise_scheduler,
                        batch,
                        step,
                        config
                    )
                                    
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss_temporal}, step=global_step)
                train_loss_temporal = 0.0
                if global_step % config.train.checkpointing_steps == 0 and global_step > 0:
                    save_pipe(
                        config.model.pretrained_model_path,
                        global_step,
                        accelerator,
                        unet,
                        text_encoder,
                        vae,
                        output_dir,
                        is_checkpoint=True,
                        save_pretrained_model=config.train.save_pretrained_model,
                        **extra_params
                    )

                if should_sample(global_step, config.train.validation_steps, config.val):
                    if accelerator.is_main_process:
                        log_validation(
                            accelerator, 
                            config,
                            batch,
                            global_step,
                            text_prompt,
                            unet,
                            text_encoder,
                            vae,
                            output_dir
                        )


                    unet_and_text_g_c(
                        unet,
                        text_encoder,
                        config.train.gradient_checkpointing,
                        config.train.text_encoder_gradient_checkpointing
                    )

            if loss_temporal is not None:
                accelerator.log({"loss_temporal": loss_temporal.detach().item()}, step=step)

            if global_step >= config.train.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_pipe(
            config.model.pretrained_model_path,
            global_step,
            accelerator,
            unet,
            text_encoder,
            vae,
            output_dir,
            is_checkpoint=False,
            save_pretrained_model=config.train.save_pretrained_model,
            **extra_params
        )
    accelerator.end_training()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, default='/remote-home/lzwang/projects/MotionInversion/configs/config.yaml')
#     args = parser.parse_args()

#     # Load and merge configurations
#     config = OmegaConf.load(args.config)
#     main(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='/remote-home/lzwang/projects/MotionInversion/configs/config.yaml')
    parser.add_argument("--single_video_path", type=str)
    parser.add_argument("--prompts", type=str, help="JSON string of prompts")
    args = parser.parse_args()

    # Load and merge configurations
    config = OmegaConf.load(args.config)
    
    # Update the config with the command-line arguments
    if args.single_video_path:
        config.dataset.single_video_path = args.single_video_path
        # Set the output dir
        config.train.output_dir = os.path.join(config.train.output_dir, os.path.basename(args.single_video_path).split('.')[0])
    
    if args.prompts:
        config.val.prompt = json.loads(args.prompts)
        


    main(config)
