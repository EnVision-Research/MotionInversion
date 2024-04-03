import math

import torch
import torch.fft as fft
import torch.nn.functional as F

from einops import rearrange


def BlendFreqInit(noisy_latent, noise, noise_prior=0.5, downsample_factor=4):
    f = noisy_latent.shape[2]
    new_h, new_w = (
        noisy_latent.shape[-2] // downsample_factor,
        noisy_latent.shape[-1] // downsample_factor,
    )

    noise = rearrange(noise, "b c f h w -> (b f) c h w")
    noise_down = F.interpolate(noise, size=(new_h, new_w), mode="bilinear", align_corners=True, antialias=True)
    noise_up = F.interpolate(
        noise_down, size=(noise.shape[-2], noise.shape[-1]), mode="bilinear", align_corners=True, antialias=True
    )
    noise_high_freqs = noise - noise_up


    noisy_latent = rearrange(noisy_latent, "b c f h w -> (b f) c h w")
    noisy_latent_down = F.interpolate(
        noisy_latent, size=(new_h, new_w), mode="bilinear", align_corners=True, antialias=True
    )
    latents_low_freqs = F.interpolate(
        noisy_latent_down,
        size=(noisy_latent.shape[-2], noisy_latent.shape[-1]),
        mode="bilinear",
        align_corners=True,
        antialias=True,
    )

    latent_high_freqs = noisy_latent - latents_low_freqs

    noisy_latent = latents_low_freqs + (noise_prior) ** 0.5 * latent_high_freqs + (
        1-noise_prior) ** 0.5 * noise_high_freqs


    noisy_latent = rearrange(noisy_latent, "(b f) c h w -> b c f h w", f=f)

    return noisy_latent