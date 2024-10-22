import torch
import random
import cv2
import fnmatch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from diffusers.optimization import get_scheduler
from einops import rearrange, repeat
from omegaconf import OmegaConf
from dataset import *
from models.unet.motion_embeddings import *
from .lora import *
from .lora_handler import *

def find_videos(directory, extensions=('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.gif')):
    video_files = []
    for root, dirs, files in os.walk(directory):
        for extension in extensions:
            for filename in fnmatch.filter(files, '*' + extension):
                video_files.append(os.path.join(root, filename))
    return video_files

def param_optim(model, condition, extra_params=None, is_lora=False, negation=None):
    extra_params = extra_params if len(extra_params.keys()) > 0 else None
    return {
        "model": model,
        "condition": condition,
        'extra_params': extra_params,
        'is_lora': is_lora,
        "negation": negation
    }

def create_optim_params(name='param', params=None, lr=5e-6, extra_params=None):
    params = {
        "name": name,
        "params": params,
        "lr": lr
    }
    if extra_params is not None:
        for k, v in extra_params.items():
            params[k] = v

    return params

def create_optimizer_params(model_list, lr):
    import itertools
    optimizer_params = []

    for optim in model_list:
        model, condition, extra_params, is_lora, negation = optim.values()
        # Check if we are doing LoRA training.
        if is_lora and condition and isinstance(model, list):
            params = create_optim_params(
                params=itertools.chain(*model),
                extra_params=extra_params
            )
            optimizer_params.append(params)
            continue

        if is_lora and condition and not isinstance(model, list):
            for n, p in model.named_parameters():
                if 'lora' in n:
                    params = create_optim_params(n, p, lr, extra_params)
                    optimizer_params.append(params)
            continue

        # If this is true, we can train it.
        if condition:
            for n, p in model.named_parameters():
                should_negate = 'lora' in n and not is_lora
                if should_negate: continue

                params = create_optim_params(n, p, lr, extra_params)
                optimizer_params.append(params)

    return optimizer_params

def get_optimizer(use_8bit_adam):
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        return bnb.optim.AdamW8bit
    else:
        return torch.optim.AdamW
    
# Initialize the optimizer
def prepare_optimizers(params, config, **extra_params):   
    optimizer_cls = get_optimizer(config.train.use_8bit_adam)

    optimizer_temporal = optimizer_cls(
        params,
        lr=config.loss.learning_rate
    )

    lr_scheduler_temporal = get_scheduler(
        config.loss.lr_scheduler,
        optimizer=optimizer_temporal,
        num_warmup_steps=config.loss.lr_warmup_steps * config.train.gradient_accumulation_steps,
        num_training_steps=config.train.max_train_steps * config.train.gradient_accumulation_steps,
    )

    # Insert Spatial LoRAs
    if config.loss.type == 'DebiasedHybrid':
        unet_lora_params_spatial_list = extra_params.get('unet_lora_params_spatial_list', [])
        spatial_lora_num = extra_params.get('spatial_lora_num', 1)

        optimizer_spatial_list = []
        lr_scheduler_spatial_list = []
        for i in range(spatial_lora_num):
            unet_lora_params_spatial = unet_lora_params_spatial_list[i]

            optimizer_spatial = optimizer_cls(
                create_optimizer_params(
                    [
                        param_optim(
                            unet_lora_params_spatial, 
                            config.loss.use_unet_lora, 
                            is_lora=True,
                            extra_params={**{"lr": config.loss.learning_rate_spatial}}
                        )
                    ], 
                    config.loss.learning_rate_spatial
                ),
                lr=config.loss.learning_rate_spatial
            )
            optimizer_spatial_list.append(optimizer_spatial)

            # Scheduler
            lr_scheduler_spatial = get_scheduler(
                config.loss.lr_scheduler,
                optimizer=optimizer_spatial,
                num_warmup_steps=config.loss.lr_warmup_steps * config.train.gradient_accumulation_steps,
                num_training_steps=config.train.max_train_steps * config.train.gradient_accumulation_steps,
            )
            lr_scheduler_spatial_list.append(lr_scheduler_spatial)

    else:
        optimizer_spatial_list = []
        lr_scheduler_spatial_list = []
    

    
    return [optimizer_temporal] + optimizer_spatial_list, [lr_scheduler_temporal] + lr_scheduler_spatial_list

def sample_noise(latents, noise_strength, use_offset_noise=False):
    b, c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)

    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents

@torch.no_grad()
def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * 0.18215

    return latents

def prepare_data(config, tokenizer):
    # Get the training dataset based on types (json, single_video, image)

    # Assuming config.dataset is a DictConfig object
    dataset_params_dict = OmegaConf.to_container(config.dataset, resolve=True)

    # Remove the 'type' key
    dataset_params_dict.pop('type', None)  # 'None' ensures no error if 'type' key doesn't exist

    train_datasets = []

    # Loop through all available datasets, get the name, then add to list of data to process.
    for DataSet in [VideoJsonDataset, SingleVideoDataset, ImageDataset, VideoFolderDataset]:
        for dataset in config.dataset.type:
            if dataset == DataSet.__getname__():
                train_datasets.append(DataSet(**dataset_params_dict, tokenizer=tokenizer))

    if len(train_datasets) < 0:
        raise ValueError("Dataset type not found: 'json', 'single_video', 'folder', 'image'")
        
    train_dataset = train_datasets[0]


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.train_batch_size,
        shuffle=True
    )

    return train_dataloader, train_dataset

# create parameters for optimziation
def prepare_params(unet, config, train_dataset):
    extra_params = {}

    params,embedding_layers = inject_motion_embeddings(
        unet, 
        combinations=config.model.motion_embeddings.combinations,
        config=config
    )

    config.model.embedding_layers = embedding_layers
    if config.loss.type == "DebiasedHybrid":
        if config.loss.spatial_lora_num == -1:
            config.loss.spatial_lora_num = train_dataset.__len__()

        lora_managers_spatial, unet_lora_params_spatial_list, unet_negation_all = inject_spatial_loras(
            unet=unet, 
            use_unet_lora=True,
            lora_unet_dropout=0.1,
            lora_path='',
            lora_rank=32,
            spatial_lora_num=1,
        )
        
        extra_params['lora_managers_spatial'] = lora_managers_spatial
        extra_params['unet_lora_params_spatial_list'] = unet_lora_params_spatial_list
        extra_params['unet_negation_all'] = unet_negation_all

    return params, extra_params