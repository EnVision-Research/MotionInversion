import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.attention import BasicTransformerBlock
from diffusers.utils.import_utils import is_xformers_available
from transformers.models.clip.modeling_clip import CLIPEncoder

GRADIENT_CHECKPOINTING =  True
TEXT_ENCODER_GRADIENT_CHECKPOINTING = True
ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION = True
ENABLE_TORCH_2_ATTN = True

def is_attn(name):
    return ('attn1' or 'attn2' == name.split('.')[-1])

def unet_and_text_g_c(unet, text_encoder, unet_enable=GRADIENT_CHECKPOINTING, text_enable=TEXT_ENCODER_GRADIENT_CHECKPOINTING):
    unet._set_gradient_checkpointing(value=unet_enable)
    text_encoder._set_gradient_checkpointing(CLIPEncoder)

def set_processors(attentions):
    for attn in attentions: attn.set_processor(AttnProcessor2_0())

def set_torch_2_attn(unet):
    optim_count = 0

    for name, module in unet.named_modules():
        if is_attn(name):
            if isinstance(module, torch.nn.ModuleList):
                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        set_processors([m.attn1, m.attn2])
                        optim_count += 1
    if optim_count > 0:
        print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")

def handle_memory_attention(
        unet,
        enable_xformers_memory_efficient_attention=ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION, 
        enable_torch_2_attn=ENABLE_TORCH_2_ATTN
    ):
    try:
        is_torch_2 = hasattr(F, 'scaled_dot_product_attention')
        enable_torch_2 = is_torch_2 and enable_torch_2_attn

        if enable_xformers_memory_efficient_attention and not enable_torch_2:
            if is_xformers_available():
                from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
                unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if enable_torch_2:
            set_torch_2_attn(unet)

    except:
        print("Could not enable memory efficient attention for xformers or Torch 2.0.")