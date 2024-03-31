import torch

from utils.ddim_utils import inverse_video

def initialize_noise_with_blend(noisy_latent, noise=None, seed=0, noise_prior=0.5):

    # noisy_latent = inverse_video(pipe, latents, 50)
    shape = noisy_latent.shape
    if noise is None:
        noise = torch.randn(
            shape, 
            device=noisy_latent.device, 
            generator=torch.Generator(noisy_latent.device).manual_seed(seed)
        ).to(noisy_latent.dtype)


    latents = (noise_prior) ** 0.5 * noisy_latent + (
        1-noise_prior) ** 0.5 * noise

    return latents