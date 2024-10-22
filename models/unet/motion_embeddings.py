import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class MotionEmbedding(nn.Module):

    def __init__(self, embed_dim: int = None, max_seq_length: int = 32, wh: int = 1):
        super().__init__()
        self.embed = nn.Parameter(torch.zeros(wh, max_seq_length, embed_dim))
        print('register spatial motion embedding with', wh)

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
        # print('seq_length',seq_length)
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

        if x.shape[0] != embeddings.shape[0]:
            x = x + embeddings.repeat(x.shape[0]//embeddings.shape[0],1,1) * self.scale
        else:
            # Now embeddings should have the shape [batch, seq_length, dim] matching x
            x = x + embeddings * self.scale  # Assuming broadcasting is desired over the batch and dim dimensions

        return x


    def forward_average(self, x):
        _, seq_length, _ = x.shape  # seq_length here is the target sequence length for x
        # print('seq_length',seq_length)
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

        embeddings_mean = embeddings.mean(dim=1, keepdim=True)
        embeddings = embeddings - embeddings_mean
        if x.shape[0] != embeddings.shape[0]:
            x = x + embeddings.repeat(x.shape[0]//embeddings.shape[0],1,1) * self.scale
        else:
            # Now embeddings should have the shape [batch, seq_length, dim] matching x
            x = x + embeddings * self.scale  # Assuming broadcasting is desired over the batch and dim dimensions

        return x

    def forward_frameSubtraction(self, x):
        _, seq_length, _ = x.shape  # seq_length here is the target sequence length for x
        # print('seq_length',seq_length)
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

        embeddings_subtraction = embeddings[:,1:] - embeddings[:,:-1]
        
        embeddings = embeddings.clone().detach()
        embeddings[:,1:] = embeddings_subtraction
        
        # first frame minus mean
        # embeddings[:,0:1] = embeddings[:,0:1] - embeddings.mean(dim=1, keepdim=True)

        if x.shape[0] != embeddings.shape[0]:
            x = x + embeddings.repeat(x.shape[0]//embeddings.shape[0],1,1) * self.scale
        else:
            # Now embeddings should have the shape [batch, seq_length, dim] matching x
            x = x + embeddings * self.scale  # Assuming broadcasting is desired over the batch and dim dimensions

        return x

class MotionEmbeddingSpatial(nn.Module):

    def __init__(self, h: int = None, w: int = None, embed_dim: int = None, max_seq_length: int = 32):
        super().__init__()
        self.embed = nn.Parameter(torch.zeros(h*w, max_seq_length, embed_dim))
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

        if x.shape[0] != embeddings.shape[0]:
            x = x + embeddings.repeat(x.shape[0]//embeddings.shape[0],1,1) * self.scale
        else:
            # Now embeddings should have the shape [batch, seq_length, dim] matching x
            x = x + embeddings * self.scale  # Assuming broadcasting is desired over the batch and dim dimensions

        return x


def inject_motion_embeddings(model, combinations=None, config=None):
    spatial_shape=np.array([config.dataset.height,config.dataset.width])
    shape32 = np.ceil(spatial_shape/32).astype(int)
    shape16 = np.ceil(spatial_shape/16).astype(int)
    spatial_name = 'vSpatial'
    replacement_dict = {}
    # support for 32 frames
    max_seq_length = 32
    inject_layers = []
    for name, module in model.named_modules():
        
        # check if the module is temp_attention
        PETemporal = '.temp_attentions.' in name

        if not(PETemporal and re.search(r'transformer_blocks\.\d+$', name)):
            continue

        if not ([name.split('_')[0], module.norm1.normalized_shape[0]] in combinations):
            continue
        
        replacement_dict[f'{name}.pos_embed'] = MotionEmbedding(max_seq_length=max_seq_length, embed_dim=module.norm1.normalized_shape[0]).to(dtype=model.dtype, device=model.device)
         
    replacement_keys = list(set(replacement_dict.keys()))
    temp_attn_list =    [name.replace('pos_embed','attn1') for name in replacement_keys] + \
                        [name.replace('pos_embed','attn2') for name in replacement_keys]
    embed_dims = [replacement_dict[replacement_keys[i]].embed.shape[2] for i in range(len(replacement_keys))]
    
    for temp_attn_index,temp_attn in enumerate(temp_attn_list):
        place_in_net = temp_attn.split('_')[0]
        pattern = r'(\d+)\.temp_attentions'
        match = re.search(pattern, temp_attn)
        place_in_net = temp_attn.split('_')[0]
        index_in_net = match.group(1)
        h,w = None,None
        if place_in_net == 'up':
            if index_in_net == "1":
                h, w = shape32
            elif index_in_net == "2":
                h, w = shape16
        elif place_in_net == 'down':
            if index_in_net == "1":
                h, w = shape16
            elif index_in_net == "2":
                h, w = shape32
        
        replacement_dict[temp_attn+'.'+spatial_name] = \
            MotionEmbedding(
                wh=h*w,
                embed_dim=embed_dims[temp_attn_index%len(replacement_keys)]
                ).to(dtype=model.dtype, device=model.device)

    for name, new_module in replacement_dict.items():
        parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
        module_name = name.rsplit('.', 1)[-1]
        parent_module = model
        if parent_name:
            parent_module = dict(model.named_modules())[parent_name]

        if [parent_name.split('_')[0], new_module.embed.shape[-1]] in combinations:
            inject_layers.append(name)
            setattr(parent_module, module_name, new_module)

    inject_layers = list(set(inject_layers))
    for name in inject_layers:
        print(f"Injecting motion embedding at {name}")

    parameters_list = []
    for name, para in model.named_parameters():
        if 'pos_embed' in name or spatial_name in name:
            parameters_list.append(para)
            para.requires_grad = True
        else:
            para.requires_grad = False

    return parameters_list, inject_layers

def save_motion_embeddings(model, file_path):
    # Extract motion embedding from all instances of MotionEmbedding
    motion_embeddings = {
        name: module.embed
        for name, module in model.named_modules()
        if isinstance(module, MotionEmbedding) or isinstance(module, MotionEmbeddingSpatial)
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

        new_module = MotionEmbedding(wh = embedding.shape[0],embed_dim=embedding.shape[-1], max_seq_length=embedding.shape[-2])

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





