import re
import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionEmbedding(nn.Module):

    def __init__(self, embed_dim: int = None, max_seq_length: int = 32):
        super().__init__()
        self.embed = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))
        self.scale = 1.0
        self.trained_length = -1

    def set_scale(self, scale: float):
        self.scale = scale

    def set_lengths(self, trained_length: int):
        if trained_length > self.embed.shape[1] or trained_length <= 0:
            raise ValueError("Trained length is out of bounds")
        self.trained_length = trained_length

    def forward(self, x):
        _, seq_length, _ = x.shape  # seq_length here is the target sequence length for x

        # Assuming self.embed is [batch, frames, dim]
        embeddings = self.embed[:, :seq_length]  # Initial slice, may not be necessary depending on the interpolation logic

        # Check if interpolation is needed
        if self.trained_length != -1 and seq_length != self.trained_length:
            # Interpolate embeddings to match x's sequence length
            # Ensure embeddings is [batch, dim, frames] for 1D interpolation across frames
            embeddings = embeddings.permute(0, 2, 1)  # Now [batch, dim, frames]
            embeddings = F.interpolate(embeddings, size=(seq_length,), mode='linear', align_corners=False)
            embeddings = embeddings.permute(0, 2, 1)  # Revert to [batch, frames, dim]

        # Ensure the interpolated embeddings match the sequence length of x
        if embeddings.shape[1] != seq_length:
            raise ValueError(f"Interpolated embeddings sequence length {embeddings.shape[1]} does not match x's sequence length {seq_length}")

        # Now embeddings should have the shape [batch, seq_length, dim] matching x
        x = x + embeddings * self.scale  # Assuming broadcasting is desired over the batch and dim dimensions

        return x

def inject_motion_embeddings(model, sizes=[320, 640, 1280], modules=['up','down','mid']):
    replacement_dict = {}

    for name, module in model.named_modules():
        if 'temp_attention' in name and re.search(r'transformer_blocks\.\d+$', name):
            replacement_dict[f'{name}.pos_embed'] = MotionEmbedding(embed_dim=module.norm1.normalized_shape[0]).to(dtype=model.dtype, device=model.device)
    
    for name, new_module in replacement_dict.items():
        parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
        module_name = name.rsplit('.', 1)[-1]
        parent_module = model
        if parent_name:
            parent_module = dict(model.named_modules())[parent_name]

        if new_module.embed.shape[-1] in sizes and parent_name.split('_')[0] in modules:
            setattr(parent_module, module_name, new_module)
    
    parameters_list = []
    for name, para in model.named_parameters():
        if 'pos_embed' in name:
            parameters_list.append(para)
            para.requires_grad = True
        else:
            para.requires_grad = False
    
    return parameters_list

def save_motion_embeddings(model, file_path):
    # Extract motion embedding from all instances of MotionEmbedding
    motion_embeddings = {
        name: module.embed
        for name, module in model.named_modules()
        if isinstance(module, MotionEmbedding)
    }
    # Save the motion embeddings to the specified file path
    torch.save(motion_embeddings, file_path)

def load_motion_embeddings(model, saved_embeddings):
    for key, embedding in saved_embeddings.items():
        # Extract parent module and module name from the key
        parent_name = key.rsplit('.', 1)[0] if '.' in key else ''
        module_name = key.rsplit('.', 1)[-1]

        # Retrieve the parent module
        parent_module = model
        if parent_name:
            parent_module = dict(model.named_modules())[parent_name]

        # Create a new MotionEmbedding instance with the correct dimensions
        new_module = MotionEmbedding(embed_dim=embedding.shape[-1], max_seq_length=embedding.shape[-2])

        # Properly assign the loaded embeddings to the 'embed' parameter wrapped in nn.Parameter
        # Ensure the embedding is on the correct device and has the correct dtype
        new_module.embed = nn.Parameter(embedding.to(dtype=model.dtype, device=model.device))

        # Replace the corresponding module in the model with the new MotionEmbedding instance
        setattr(parent_module, module_name, new_module)

def set_motion_embedding_scale(model, scale_value):
    # Iterate over all modules in the model
    for _, module in model.named_modules():
        # Check if the module is an instance of MotionEmbedding
        if isinstance(module, MotionEmbedding):
            # Set the scale attribute to the specified value
            module.scale = scale_value

def set_motion_embedding_length(model, trained_length):
    # Iterate over all modules in the model
    for _, module in model.named_modules():
        # Check if the module is an instance of MotionEmbedding
        if isinstance(module, MotionEmbedding):
            # Set the length to the specified value
            module.trained_length = trained_length





