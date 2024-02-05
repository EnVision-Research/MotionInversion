CUDA_VISIBLE_DEVICES=1 accelerate launch pe_inversion_unet3d.py \
    --pretrained_model_name_or_path='/home/wangluozhou/pretrained_models/zeroscope_v2_576w' \
    --per_gpu_batch_size=1 --gradient_accumulation_steps=1 \
    --max_train_steps=600 \
    --width=288 \
    --height=160 \
    --num_frames=24 \
    --checkpointing_steps=100 --checkpoints_total_limit=3 \
    --learning_rate=1e-1 --lr_warmup_steps=0 \
    --seed=0 \
    --validation_steps=100 \
    --output_dir='/home/wangluozhou/projects/MotionInversion/outputs/0205/05' \
    --validation_file='/home/wangluozhou/projects/MotionInversion/resources/05.txt' \
    --video_path='/home/wangluozhou/projects/MotionInversion/resources/05_cats_play_24.mp4' \
    --pe_size 1280 \
    --pe_module down mid up \
    --mixed_precision="fp16" \
    --enable_xformers_memory_efficient_attention \
    --num_validation_videos 3
    
