import torch
from torchvision import transforms
import torch.nn.functional as F
import random

from utils.lora import extract_lora_child_module
from utils.func_utils import tensor_to_vae_latent, sample_noise

def DebiasedHybridLoss(
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
        config,
        random_hflip_img=False,
        spatial_lora_num=1
    ):
    mask_spatial_lora = random.uniform(0, 1) < 0.2
    cache_latents = config.train.cache_latents



    if not cache_latents:
        latents = tensor_to_vae_latent(batch["pixel_values"], vae)
    else:
        latents = batch["latents"]

    # Sample noise that we'll add to the latents
    # use_offset_noise = use_offset_noise and not rescale_schedule
        
    noise = sample_noise(latents, 0.1, False)
    bsz = latents.shape[0]

    # Sample a random timestep for each video
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
    timesteps = timesteps.long()

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # *Potentially* Fixes gradient checkpointing training.
    # See: https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
    # if kwargs.get('eval_train', False):
    #     unet.eval()
    #     text_encoder.eval()

    # Encode text embeddings
    token_ids = batch['prompt_ids']
    encoder_hidden_states = text_encoder(token_ids)[0]
    detached_encoder_state = encoder_hidden_states.clone().detach()

    # Get the target for loss depending on the prediction type
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise

    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)

    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    encoder_hidden_states = detached_encoder_state


    # optimization
    if mask_spatial_lora:
        loras = extract_lora_child_module(unet, target_replace_module=["Transformer2DModel"])
        for lora_i in loras:
            lora_i.scale = 0.
        loss_spatial = None
    else:
        loras = extract_lora_child_module(unet, target_replace_module=["Transformer2DModel"])

        if spatial_lora_num == 1:
            for lora_i in loras:
                lora_i.scale = 1.
        else:
            for lora_i in loras:
                lora_i.scale = 0.

            for lora_idx in range(0, len(loras), spatial_lora_num):
                loras[lora_idx + step].scale = 1.

        loras = extract_lora_child_module(unet, target_replace_module=["TransformerTemporalModel"])
        if len(loras) > 0:
            for lora_i in loras:
                lora_i.scale = 0.

        ran_idx = torch.randint(0, noisy_latents.shape[2], (1,)).item()

        if random.uniform(0, 1) < random_hflip_img:
            pixel_values_spatial = transforms.functional.hflip(
                batch["pixel_values"][:, ran_idx, :, :, :]).unsqueeze(1)
            latents_spatial = tensor_to_vae_latent(pixel_values_spatial, vae)
            noise_spatial = sample_noise(latents_spatial, 0.1, False)
            noisy_latents_input = noise_scheduler.add_noise(latents_spatial, noise_spatial, timesteps)
            target_spatial = noise_spatial
            model_pred_spatial = unet(noisy_latents_input, timesteps,
                                    encoder_hidden_states=encoder_hidden_states).sample
            loss_spatial = F.mse_loss(model_pred_spatial[:, :, 0, :, :].float(),
                                    target_spatial[:, :, 0, :, :].float(), reduction="mean")
        else:
            noisy_latents_input = noisy_latents[:, :, ran_idx, :, :]
            target_spatial = target[:, :, ran_idx, :, :]
            model_pred_spatial = unet(noisy_latents_input.unsqueeze(2), timesteps,
                                    encoder_hidden_states=encoder_hidden_states).sample
            loss_spatial = F.mse_loss(model_pred_spatial[:, :, 0, :, :].float(),
                                    target_spatial.float(), reduction="mean")


    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
    loss_temporal = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    beta = 1
    alpha = (beta ** 2 + 1) ** 0.5
    ran_idx = torch.randint(0, model_pred.shape[2], (1,)).item()
    model_pred_decent = alpha * model_pred - beta * model_pred[:, :, ran_idx, :, :].unsqueeze(2)
    target_decent = alpha * target - beta * target[:, :, ran_idx, :, :].unsqueeze(2)
    loss_ad_temporal = F.mse_loss(model_pred_decent.float(), target_decent.float(), reduction="mean")
    loss_temporal = loss_temporal + loss_ad_temporal

    avg_loss_temporal = accelerator.gather(loss_temporal.repeat(config.train.train_batch_size)).mean()
    train_loss_temporal += avg_loss_temporal.item() / config.train.gradient_accumulation_steps

    if not mask_spatial_lora:
        accelerator.backward(loss_spatial, retain_graph=True)
        if spatial_lora_num == 1:
            optimizers[1].step()
        else:
            optimizers[step+1].step()

    accelerator.backward(loss_temporal)
    optimizers[0].step()

    if spatial_lora_num == 1:
        lr_schedulers[1].step()
    else:
        lr_schedulers[1 + step].step()

    lr_schedulers[0].step()

    return loss_temporal, train_loss_temporal