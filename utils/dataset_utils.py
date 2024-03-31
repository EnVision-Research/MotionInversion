import os
import json
import decord
decord.bridge.set_bridge('torch')

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T

from itertools import islice

from glob import glob
from PIL import Image
from einops import rearrange, repeat


def read_caption_file(caption_file):
        with open(caption_file, 'r', encoding="utf8") as t:
            return t.read()

def get_text_prompt(
        text_prompt: str = '', 
        fallback_prompt: str= '',
        file_path:str = '', 
        ext_types=['.mp4'],
        use_caption=False
    ):
    try:
        if use_caption:
            if len(text_prompt) > 1: return text_prompt
            caption_file = ''
            # Use caption on per-video basis (One caption PER video)
            for ext in ext_types:
                maybe_file = file_path.replace(ext, '.txt')
                if maybe_file.endswith(ext_types): continue
                if os.path.exists(maybe_file): 
                    caption_file = maybe_file
                    break

            if os.path.exists(caption_file):
                return read_caption_file(caption_file)
            
            # Return fallback prompt if no conditions are met.
            return fallback_prompt

        return text_prompt
    except:
        print(f"Couldn't read prompt caption for {file_path}. Using fallback.")
        return fallback_prompt

def get_video_frames(vr, start_idx, sample_rate=1, max_frames=24):
    max_range = len(vr)
    frame_number = sorted((0, start_idx, max_range))[1]

    frame_range = range(frame_number, max_range, sample_rate)
    frame_range_indices = list(frame_range)[:max_frames]

    return frame_range_indices

def get_prompt_ids(prompt, tokenizer):
    prompt_ids = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
    ).input_ids

    return prompt_ids

def process_video(vid_path, use_bucketing, w, h, get_frame_buckets, get_frame_batch):
    if use_bucketing:
        vr = decord.VideoReader(vid_path)
        resize = get_frame_buckets(vr)
        video = get_frame_batch(vr, resize=resize)

    else:
        vr = decord.VideoReader(vid_path, width=w, height=h)
        video = get_frame_batch(vr)

    return video, vr

def min_res(size, min_size): return 192 if size < 192 else size

def up_down_bucket(m_size, in_size, direction):
    if direction == 'down': return abs(int(m_size - in_size))
    if direction == 'up': return abs(int(m_size + in_size))

def get_bucket_sizes(size, direction: 'down', min_size):
    multipliers = [64, 128]
    for i, m in enumerate(multipliers):
        res =  up_down_bucket(m, size, direction)
        multipliers[i] = min_res(res, min_size=min_size)
    return multipliers

def closest_bucket(m_size, size, direction, min_size):
    lst = get_bucket_sizes(m_size, direction, min_size)
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i]-size))]

def resolve_bucket(i,h,w): return  (i / (h / w))

def sensible_buckets(m_width, m_height, w, h, min_size=192):
    if h > w:
        w = resolve_bucket(m_width, h, w)
        w = closest_bucket(m_width, w, 'down', min_size=min_size)
        return w, m_height
    if h < w:
        h = resolve_bucket(m_height, w, h)
        h = closest_bucket(m_height, h, 'down', min_size=min_size)
        return m_width, h

    return m_width, m_height