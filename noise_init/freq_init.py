""" 
https://arxiv.org/abs/2311.17009
"""

import math

import torch
import torch.fft as fft
import torch.nn.functional as F

from einops import rearrange


def FreqInit(noisy_latent, noise, downsample_factor=4, num_frames=24):

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