import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from train import export_to_video
from models.unet.motion_embeddings import load_motion_embeddings
from noise_init.blend_init import BlendInit
from noise_init.blend_freq_init import BlendFreqInit
from noise_init.fft_init import FFTInit
from noise_init.freq_init import FreqInit
from attn_ctrl import register_attention_control
import numpy as np
import os
from omegaconf import OmegaConf

def get_pipe(embedding_dir='baseline',config=None,noisy_latent=None, video_round=None):

    # load video generation model
    pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w",torch_dtype=torch.float16)

    # use videocrafterv2 unet
    if config.model.unet == 'videoCrafter2':
        from models.unet.unet_3d_condition import UNet3DConditionModel
        # unet = UNet3DConditionModel.from_pretrained("adamdad/videocrafterv2_diffusers",subfolder='unet',torch_dtype=torch.float16)
        unet = UNet3DConditionModel.from_pretrained("adamdad/videocrafterv2_diffusers",torch_dtype=torch.float16)
        
        pipe.unet = unet

    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # memory optimization
    pipe.enable_vae_slicing()

    # if 'vanilla' not in embedding_dir:

    noisy_latent = torch.load(f'{embedding_dir}/cached_latents/cached_0.pt')['inversion_noise'][None,]
    if video_round is None:
        motion_embed = torch.load(f'{embedding_dir}/motion_embed.pt')
    else:
        motion_embed = torch.load(f'{embedding_dir}/{video_round}/motion_embed.pt')
    load_motion_embeddings(
        pipe.unet, 
        motion_embed, 
    )
    config.model['embedding_layers'] = list(motion_embed.keys())

    return pipe, config, noisy_latent

def inference(embedding_dir='vanilla',
              video_round=None, 
              prompts=None,
              save_dir=None,
              seed=None,
              motion_type=None,
              ):
    
    # check motion type is valid
    if motion_type != 'camera' and \
        motion_type != 'object' and \
            motion_type != 'hybrid':
        raise ValueError('Invalid motion type')

    if seed is None:
        seed = 0

    # load motion embedding
    noisy_latent = None

    config = OmegaConf.load(f'{embedding_dir}/config.yaml') 
    

    # different motion type assigns different strategy
    if motion_type == 'camera':
        config['strategy']['removeMFromV'] = True
        
    elif motion_type == 'object' or motion_type == 'hybrid':
        config['strategy']['vSpatial_frameSubtraction'] = True

    
    pipe, config, noisy_latent = get_pipe(embedding_dir=embedding_dir,config=config,noisy_latent=noisy_latent,video_round=video_round)
    n_frames = config.val.num_frames

    all_video_frames = []
    shape = (config.val.height,config.val.width)
    os.makedirs(save_dir,exist_ok=True)
    for prompt in prompts:

        cur_save_dir = f'{save_dir}/{"_".join(prompt.split())}.mp4'

        
        register_attention_control(pipe.unet, None,config=config)

        if noisy_latent is not None:
            torch.manual_seed(seed)
            noise = torch.randn_like(noisy_latent)
            init_noise = BlendInit(noisy_latent, noise, noise_prior=0.5)
        else:
            init_noise = None

        input_init_noise = init_noise.clone() if not init_noise is None else None
        video_frames = pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=12,
            height=shape[0],
            width=shape[1],
            num_frames=n_frames,
            generator=torch.Generator("cuda").manual_seed(seed),
            latents=input_init_noise,
        ).frames[0]

        video_path = export_to_video(video_frames,output_video_path=cur_save_dir,fps=8)
        print(video_path)
        all_video_frames.append(video_frames)
    return all_video_frames


if __name__ =="__main__":
       
    prompts = ["A skateboard slides along a city lane",
                "A tank is running in the desert.",
                "A toy train chugs around a roundabout tree"]

    
    embedding_dir = './results'
    video_round = 'checkpoint-250'
    save_dir = f'outputs'

    inference(
        embedding_dir=embedding_dir,
        prompts=prompts, 
        video_round=video_round,
        save_dir=save_dir,
        motion_type='hybrid',
        seed=100
        )
    
