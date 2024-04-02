import torch
import torch.nn.functional as F
from utils.func_utils import tensor_to_vae_latent, sample_noise

def MotionDistillationLoss(
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
    ):
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
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
    
    loss_temporal = 0
    model_pred_reidual = torch.abs(model_pred[:,:,1:,:,:] - model_pred[:,:,:-1,:,:])
    target_residual = torch.abs(target[:, :, 1:, :, :] - target[:, :, :-1, :, :])
    loss_temporal = loss_temporal + (1 - F.cosine_similarity(model_pred_reidual, target_residual, dim=2).mean)

    avg_loss_temporal = accelerator.gather(loss_temporal.repeat(config.train.train_batch_size)).mean()
    train_loss_temporal += avg_loss_temporal.item() / config.train.gradient_accumulation_steps

    accelerator.backward(loss_temporal)
    optimizers[0].step()
    lr_schedulers[0].step()

    return loss_temporal, train_loss_temporal

