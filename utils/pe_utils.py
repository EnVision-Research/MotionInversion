import re
import torch
from torch import nn
from diffusers.models.embeddings import SinusoidalPositionalEmbedding

class SinusoidalPositionalEmbeddingForInversion(nn.Module):
    """Apply positional information to a sequence of embeddings.

    Takes in a sequence of embeddings with shape (batch_size, seq_length, embed_dim) and adds positional embeddings to
    them

    Args:
        embed_dim: (int): Dimension of the positional embedding.
        max_seq_length: Maximum sequence length to apply positional embeddings

    """

    def __init__(self, pe=None, embed_dim: int = None, max_seq_length: int = 32, dtype=torch.float16):
        super().__init__()
        if pe is not None:
            self.pe = nn.Parameter(pe.to(dtype))
        else:
            self.pe = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim).to(dtype))

    def forward(self, x):
        batch_size, seq_length, _ = x.shape
        # if seq_length != 16:

        #     pe_interpolated = self.pe[:, :16].permute(0, 2, 1)

        #     pe_interpolated = F.interpolate(pe_interpolated, size=seq_length, mode='linear', align_corners=False)

        #     pe_interpolated = pe_interpolated.permute(0, 2, 1)

        #     pe_interpolated = pe_interpolated.expand(batch_size, -1, -1)

        #     x = x + pe_interpolated
        # else:
        x = x + self.pe[:, :seq_length]

        return x

def replace_positional_embedding(model,target_size=[320, 640, 1280], target_module=['up','down','mid']):
    replacement_dict = {}

    # First, identify all modules that need to be replaced
    for name, module in model.named_modules():
        if isinstance(module, SinusoidalPositionalEmbedding):
            replacement_dict[name] = SinusoidalPositionalEmbeddingForInversion(pe=module.pe, dtype=model.dtype)

    # Now, replace the identified modules
    for name, new_module in replacement_dict.items():
        parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
        module_name = name.rsplit('.', 1)[-1]
        parent_module = model
        if parent_name:
            parent_module = dict(model.named_modules())[parent_name]

        if new_module.pe.shape[-1] in target_size and parent_name.split('_')[0] in target_module:
            setattr(parent_module, module_name, new_module)

def replace_positional_embedding_unet3d(model,target_size=[320, 640, 1280], target_module=['up','down','mid']):
    replacement_dict = {}

    # First, identify all modules that need to be replaced
    for name, module in model.named_modules():
        if 'temp_attention' in name and re.search(r'transformer_blocks\.\d+$', name):
            replacement_dict[f'{name}.pos_embed'] = SinusoidalPositionalEmbeddingForInversion(embed_dim=module.norm1.normalized_shape[0], dtype=model.dtype)

    # Now, replace the identified modules
    for name, new_module in replacement_dict.items():
        parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
        module_name = name.rsplit('.', 1)[-1]
        parent_module = model
        if parent_name:
            parent_module = dict(model.named_modules())[parent_name]

        if new_module.pe.shape[-1] in target_size and parent_name.split('_')[0] in target_module:
            setattr(parent_module, module_name, new_module)

def save_positional_embeddings(model, file_path):
    # Extract positional embeddings from all instances of SinusoidalPositionalEmbeddingForInversion
    positional_embeddings = {
        name: module.pe
        for name, module in model.named_modules()
        if isinstance(module, SinusoidalPositionalEmbeddingForInversion)
    }
    # Save the positional embeddings to the specified file path
    torch.save(positional_embeddings, file_path)

# def load_positional_embeddings(model, file_path):
#     # Load the positional embeddings from the file
#     saved_embeddings = torch.load(file_path)
#     # Assign the loaded embeddings back to the corresponding modules in the model
#     for name, module in model.named_modules():
#         if isinstance(module, SinusoidalPositionalEmbeddingForInversion):
#             module.pe.data.copy_(saved_embeddings[name].data)


# def load_positional_embeddings(model, file_path):
#     # Load the positional embeddings from the file
#     saved_embeddings = torch.load(file_path)
#     # Assign the loaded embeddings back to the corresponding modules in the model
#     for name, module in model.named_modules():
#         if isinstance(module, SinusoidalPositionalEmbeddingForInversion):
#             module.pe.data.copy_(saved_embeddings[name].data)



def load_positional_embedding(model,file_path):
    replacement_dict = {}
    saved_embeddings = torch.load(file_path)

    # First, identify all modules that need to be replaced
    # for name, module in model.named_modules():
    #     if 'temp_attention' in name and re.search(r'transformer_blocks\.\d+$', name):
    #         replacement_dict[f'{name}.pos_embed'] = SinusoidalPositionalEmbeddingForInversion(pe=saved_embeddings[f'{name}.pos_embed'].data, dtype=model.dtype)

    for key in saved_embeddings.keys():
        replacement_dict[key] = SinusoidalPositionalEmbeddingForInversion(pe=saved_embeddings[key].data, dtype=model.dtype)


    # Now, replace the identified modules
    for name, new_module in replacement_dict.items():
        parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
        module_name = name.rsplit('.', 1)[-1]
        parent_module = model
        if parent_name:
            parent_module = dict(model.named_modules())[parent_name]
        # if new_module.pe.shape[-1] in target_size and parent_name.split('_')[0] in target_module:
        setattr(parent_module, module_name, new_module)