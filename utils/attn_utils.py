import abc
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def reduce_batch_size_and_convert_to_image(attention_map, pixel_size=10, frames=16):
    # Reducing the batch size dimension by averaging
    reduced_map = attention_map.mean(dim=0)
    
    # Normalizing the attention map to be between 0 and 1
    normalized_map = (reduced_map - reduced_map.min()) / (reduced_map.max() - reduced_map.min())
    
    # Converting to 24x24 image
    image = normalized_map.view(frames, frames).cpu().detach().numpy()
    
    # Scaling the normalized image to be between 0 and 255 for uint8
    image_scaled = np.uint8(image * 255)
    
    # Convert to PIL Image in 'L' mode (grayscale) and resize
    pil_image = Image.fromarray(image_scaled, 'L').resize((frames * pixel_size, frames * pixel_size), resample=Image.NEAREST)
    
    return pil_image

def build_image_grid(attention_maps_list, pixel_size=10, frames=16):
    images = [reduce_batch_size_and_convert_to_image(attention_map, frames=frames) for attention_map in attention_maps_list]
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(len(images))))
    
    # Resize each image to make each pixel a larger square
    resized_images = [image.resize((frames * pixel_size, frames * pixel_size), resample=Image.NEAREST) for image in images]
    width, height = resized_images[0].size
    
    # Create a new image with white background for the grid
    grid_img = Image.new('RGB', size=(grid_size * width, grid_size * height), color=(255, 255, 255))
    
    for i, image in enumerate(resized_images):
        grid_x = (i % grid_size) * width
        grid_y = (i // grid_size) * height
        grid_img.paste(image, (grid_x, grid_y))
    
    return grid_img

def compute_average_map(attention_maps_list, pixel_size=20, frames=16, reduction='mean', batch_size=2, height=16, width=16):

    dtype = attention_maps_list[0].dtype
    device = attention_maps_list[0].device
    if reduction == 'temporal':
        # Initialize an empty tensor for averaging
        average_map = torch.zeros(batch_size, height, width, frames, frames, dtype=dtype, device=device)

        for attention_map in attention_maps_list:
            # Restore each attention map back to [batch_size, height, width, num_frames, num_frames]
            reshaped_map = attention_map.reshape(batch_size, height, width, frames, frames)
            average_map += reshaped_map

        # Compute the average
        average_map /= len(attention_maps_list)

        image_batch = []
        for b in range(batch_size):
            # Create a grid for each batch
            grid = torch.zeros(height * frames, width * frames).to(device, dtype)

            for h in range(height):
                for w in range(width):
                    # Extract each num_frames * num_frames image
                    img = average_map[b, h, w, :, :]
                    grid[h*frames:(h+1)*frames, w*frames:(w+1)*frames] = img

            # Normalize and convert to PIL image
            grid_normalized = grid.cpu().detach().numpy()
            grid_normalized = (grid_normalized - grid_normalized.min()) / (grid_normalized.max() - grid_normalized.min()) * 255
            grid_image = Image.fromarray(grid_normalized.astype(np.uint8), 'L')
            resized_image = grid_image.resize((width * frames * pixel_size, height * frames * pixel_size), resample=Image.NEAREST)
            image_batch.append(resized_image)
        return image_batch
    
    elif reduction =='spatial':
        average_map = torch.zeros(batch_size, frames, frames, height, width, dtype=dtype, device=device)

        for attention_map in attention_maps_list:
            # Restore each attention map back to [batch_size, height, width, num_frames, num_frames]
            reshaped_map = attention_map.reshape(batch_size, height, width, frames, frames)
            average_map += reshaped_map

        # Compute the average
        average_map /= len(attention_maps_list)

        # Process the average map to create a batch of frame grid images
        image_batch = []
        for b in range(batch_size):
            # Create a grid for each batch
            grid = torch.zeros(frames * height, frames * width, dtype=dtype, device=device)

            for f1 in range(frames):
                for f2 in range(frames):
                    # Extract each height * width image
                    img = average_map[b, :, :, f1, f2]
                    grid[f1*height:(f1+1)*height, f2*width:(f2+1)*width] = img

            # Normalize and convert to PIL image
            grid_normalized = grid.cpu().numpy()
            grid_normalized = (grid_normalized - grid_normalized.min()) / (grid_normalized.max() - grid_normalized.min()) * 255
            grid_image = Image.fromarray(grid_normalized.astype(np.uint8), 'L')
            resized_image = grid_image.resize((width * frames * pixel_size, height * frames * pixel_size), resample=Image.NEAREST)

            image_batch.append(resized_image)

        return image_batch

    elif reduction =='mean':
        # Initialize an empty tensor for averaging
        average_map = torch.zeros(frames, frames).to(device, dtype)
        
        for attention_map in attention_maps_list:
            # Reduce each attention map and add to the average
            reduced_map = attention_map.mean(dim=0)
            average_map += reduced_map.view(frames, frames)
        
        # Compute the average
        average_map /= len(attention_maps_list)
        
        # Convert the average tensor to a numpy array
        average_array = average_map.cpu().detach().numpy()

        # Normalize the array to be in the range [0, 255]
        average_array_normalized = (average_array - average_array.min()) / (average_array.max() - average_array.min()) * 255
        average_array_normalized = average_array_normalized.astype(np.uint8)

        # Convert to a PIL image in 'L' mode (grayscale)
        average_image = Image.fromarray(average_array_normalized, 'L')

        # Resize the image to make each pixel a larger square
        new_size = (frames * pixel_size, frames * pixel_size)
        resized_image = average_image.resize(new_size, resample=Image.NEAREST)

        return resized_image

def register_attention_control(self, controller):

    attn_procs = {}
    temp_attn_count = 0

    for name in self.unet.attn_processors.keys():
        if 'temp_attentions' in name or 'motion_modules' in name:
            if name.endswith("fuser.attn.processor"):
                attn_procs[name] = DummyAttnProcessor()
                continue

            if name.startswith("mid_block"):
                place_in_unet = "mid"

            elif name.startswith("up_blocks"):
                place_in_unet = "up"

            elif name.startswith("down_blocks"):
                place_in_unet = "down"

            else:
                continue

            temp_attn_count += 1
            attn_procs[name] = MyAttnProcessor(
                attnstore=controller, place_in_unet=place_in_unet
            )
        else:
            attn_procs[name] = AttnProcessor2_0()


    self.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = temp_attn_count

class MyAttnProcessor:

    def __init__(self, attnstore, place_in_unet, hidden_states_store=None):
        super().__init__()
        self.attnstore = attnstore
        self.hidden_states_store = hidden_states_store
        self.place_in_unet = place_in_unet

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size=batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        attention_probs = self.attnstore(attention_probs, is_cross, self.place_in_unet) # 

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0 # compute in parrallel

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # if attn.shape[1] <= 64 ** 2:  # avoid memory overhead
        resolution = int((attn.shape[0] // (self.batch_size * 2)) ** (0.5))
        if key in self.target_keys and resolution in self.target_resolutions:
            self.step_store[key].append(attn)
        return attn
    
    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self, type=None):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

    def __init__(self):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.target_keys = ['down_self', 'mid_self', 'up_self']
        self.target_resolutions = [16, 32, 64, 128]
        self.batch_size = 1




class AttentionReplacement(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # if attn.shape[1] <= 64 ** 2:  # avoid memory overhead
        # self.step_store[key].append(attn)
        resolution = int((attn.shape[0] // (self.batch_size * 2)) ** (0.5))
        if key in self.target_keys and resolution in self.target_resolutions:
            h = attn.shape[0] // 2
            attn[h:] = attn[:h]

        return attn
    
    # def between_steps(self):
    #     if len(self.attention_store) == 0:
    #         self.attention_store = self.step_store
    #     else:
    #         for key in self.attention_store:
    #             for i in range(len(self.attention_store[key])):
    #                 self.attention_store[key][i] += self.step_store[key][i]
    #     self.step_store = self.get_empty_store()

    # def get_average_attention(self):
    #     average_attention = self.attention_store
    #     return average_attention

    # def get_average_global_attention(self, type=None):
    #     average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
    #                          self.attention_store}
    #     return average_attention

    def reset(self):
        super(AttentionReplacement, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

    def __init__(self):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super(AttentionReplacement, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.target_keys = ['down_self', 'mid_self', 'up_self']
        self.target_resolutions = [16, 32, 64, 128]
        self.batch_size = 2