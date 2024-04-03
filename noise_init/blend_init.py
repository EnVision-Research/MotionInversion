""" 
https://arxiv.org/abs/2310.08465
"""

def BlendInit(noisy_latent, noise, noise_prior=0.5):

    latents = (noise_prior) ** 0.5 * noisy_latent + (
        1-noise_prior) ** 0.5 * noise

    return latents