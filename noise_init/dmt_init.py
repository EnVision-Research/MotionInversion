import math

import torch
import torch.fft as fft
import torch.nn.functional as F

from einops import rearrange

from utils.ddim_utils import inverse_video

def initialize_noise_with_dmt(noisy_latent, noise=None, seed=0, downsample_factor=4, num_frames=24):

    # noisy_latent = inverse_video(pipe, latents, 50)

    shape = noisy_latent.shape
    if noise is None:
        noise = torch.randn(
            shape, 
            device=noisy_latent.device, 
            generator=torch.Generator(noisy_latent.device).manual_seed(seed)
        ).to(noisy_latent.dtype)

    new_h, new_w = (
        noisy_latent.shape[-2] // downsample_factor,
        noisy_latent.shape[-1] // downsample_factor,
    )
    noise = rearrange(noise, "b c f h w -> (b f) c h w")
    noise_down = F.interpolate(noise, size=(new_h, new_w), mode="bilinear", align_corners=True, antialias=True)
    noise_up = F.interpolate(
        noise_down, size=(noise.shape[-2], noise.shape[-1]), mode="bilinear", align_corners=True, antialias=True
    )
    high_freqs = noise - noise_up
    noisy_latent = rearrange(noisy_latent, "b c f h w -> (b f) c h w")
    noisy_latent_down = F.interpolate(
        noisy_latent, size=(new_h, new_w), mode="bilinear", align_corners=True, antialias=True
    )
    low_freqs = F.interpolate(
        noisy_latent_down,
        size=(noisy_latent.shape[-2], noisy_latent.shape[-1]),
        mode="bilinear",
        align_corners=True,
        antialias=True,
    )
    noisy_latent = low_freqs + high_freqs
    noisy_latent = rearrange(noisy_latent, "(b f) c h w -> b c f h w", f=num_frames)

    return noisy_latent