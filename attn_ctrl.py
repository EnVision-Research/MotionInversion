import abc

LOW_RESOURCE = False
import torch 
import cv2 
import torch
import os
import numpy as np
from collections import defaultdict





class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str, heads):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str, heads):
        if self.record_this_cycle == False:
            return attn
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet, heads)
        
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
        self.record_this_cycle=True

    def disable(self):
        self.record_this_cycle = False
    
    def enable(self):
        self.record_this_cycle = True

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str, heads=None):
        return attn
    
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    @staticmethod
    def get_empty_store_motion_embds():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str, heads=None):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
            if heads is not None:
                self.heads.append(heads)
        return attn

    def forward_motion_embds(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        self.step_store_motion_embds[key].append(attn)

    def forward_hidden_states_F(self, attn, is_cross: bool, place_in_unet: str):
        save_to = 'step_store_hidden_states_F'
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        h = attn.shape[0]
        step_store_hidden_states = getattr(self, save_to)
        step_store_hidden_states[key].append(attn[h//2:])

    def forward_hidden_states_Fm(self, attn, is_cross: bool, place_in_unet: str):
        save_to = 'step_store_hidden_states_Fm'
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        h = attn.shape[0]
        step_store_hidden_states = getattr(self, save_to)
        step_store_hidden_states[key].append(attn[h//2:])

    def between_steps(self):
        if len(getattr(self, self.attention_store_lists[0])) == 0:
            for i in range(len(self.step_store_lists)):
                setattr(self, self.attention_store_lists[i], getattr(self,self.step_store_lists[i]))
        else:
            # self.vis_up_attn_maps(attention_store=self.step_store,round=self.cur_step)

            for attention_store, step_store in zip(self.attention_store_lists, self.step_store_lists):
                attention_store_list = getattr(self, attention_store)
                step_store_list = getattr(self, step_store)
                for key in attention_store_list:
                    for i in range(len(attention_store_list[key])):
                        if not step_store_list[key][i].shape[0] == attention_store_list[key][i].shape[0]:
                            print(f"Discard: {key} {i} step: {step_store_list[key][i].shape[0]} store: {attention_store_list[key][i].shape[0]}")
                            continue
                        attention_store_list[key][i] += step_store_list[key][i]
                setattr(self, attention_store, attention_store_list)
        for i in range(len(self.step_store_lists)):
            setattr(self, self.step_store_lists[i], self.get_empty_store())

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def get_empty(self):
        for i in range(len(self.step_store_lists)):
            setattr(self, self.step_store_lists[i], self.get_empty_store())

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.step_store_motion_embds = self.get_empty_store_motion_embds()
        self.motion_embds_store = {}
        self.get_empty()

    def __init__(self,prompt='origin',seed='42',save_dir='inference_output', shape=(320,576)):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.heads = []
        self.prompt = prompt
        self.seed = seed
        self.attn_names = defaultdict(list)
        self.save_dir = save_dir

        self.step_store_motion_embds = self.get_empty_store_motion_embds()
        self.motion_embds_store = {}

        self.step_store_hidden_states = self.get_empty_store()
        self.hidden_states_store = {}
        
        self.step_store_lists = ['step_store', 'step_store_motion_embds', 
                                 'step_store_hidden_states_F', 'step_store_hidden_states_Fm']
        self.attention_store_lists = ['attention_store', 'motion_embds_store', 
                                      'hidden_states_store_F', 'hidden_states_store_Fm']

        shape = np.array(shape)
        shape8 = np.ceil(shape / 8).astype(int)
        shape16 = np.ceil(shape / 16).astype(int)
        shape32 = np.ceil(shape / 32).astype(int)
        shape64 = np.ceil(shape / 64).astype(int)
        self.attention_shapes_h_w = {
            'down':[shape8,shape16,shape32],
            'mid':[shape64],
            'up':[shape32,shape16,shape8]
        }
        print(self.attention_shapes_h_w)
        self.get_empty()
    
    def vis_motion_embds_maps(self,round=None):
        attention_store = self.motion_embds_store
        keys = ['down_self','up_self']
        merge_nums = [4,6]
        orders = [
            [1,2],
            [1,2]
        ]
        save_dir = f'{self.save_dir}/{"_".join(self.prompt.split())}_seed_{self.seed}/position_embs'
        os.makedirs(save_dir, exist_ok=True)

        for key, merge_num, order in zip(keys,merge_nums, orders):
            attention_maps = attention_store[key]
            attention_maps = [sum(attention_maps[i:i+merge_num]) for i in range(0,len(attention_maps),merge_num)]
            for attn_idx, attention_map in enumerate(attention_maps):
                attention_map = attention_map.detach().cpu().numpy()
                attention_map = attention_map.sum(axis=0)
                attention_map = (attention_map - attention_map.min())/(attention_map.max()-attention_map.min())
                attention_map = cv2.applyColorMap((attention_map*255).astype(np.uint8), cv2.COLORMAP_JET)

                gap = 8
                expand_scale = 8
                f = attention_map.shape[0]
                larger_map = np.zeros(((expand_scale+gap)  * f ,(expand_scale+ gap)  * f,3),dtype='uint8') + 255
                for i in range(f):
                    for j in range(f):
                        larger_map[i*expand_scale+i*gap:i*expand_scale+i*gap + expand_scale , j*expand_scale+j*gap:j*expand_scale+j*gap+expand_scale] = attention_map[i,j]
                cv2.imwrite(f'{save_dir}/{key}_{order[attn_idx]}.png',larger_map)

        return None

    def vis_up_attn_maps(self,attention_store=None,round=None):
        attention_shapes_h_w = self.attention_shapes_h_w
        if attention_store is None:
            attention_store = self.attention_store
        up_self = attention_store['up_self']

        if len(up_self) == 18:
            up_self = [sum(up_self[0:6]),sum(up_self[6:12]),sum(up_self[12:18])]
        elif len(up_self) == 12:
            up_self = [sum(up_self[0:6]),sum(up_self[6:12])]
        elif len(up_self) == 6:
            up_self = [sum(up_self[0:6])]
        elif len(up_self) == 0:
            return
        else:
            raise ValueError(f'up_self length {len(up_self)} is not supported')
        save_dir = f'{self.save_dir}/{"_".join(self.prompt.split())}_seed_{self.seed}'
        if round is not None:
            save_dir = f'{save_dir}/round_{round}'
        os.makedirs(save_dir, exist_ok=True)

        for i in range(len(up_self)):
            attention_map = up_self[i].detach().cpu().numpy()
            h, w = attention_shapes_h_w['up'][i]
            f = attention_map.shape[-2]
            gap = h // 5
            canvas = np.zeros((h*f + gap*f, w*f + gap*f,3), dtype='uint8') + 255
            attention_map = attention_map.reshape(h,w,-1,f,f)
            attention_map = attention_map.sum(axis=2) / attention_map.shape[2]

            # normalize along q axis
            # attention_map = (attention_map - attention_map.min(axis=-1,keepdims=True))/(attention_map.max(axis=-1,keepdims=True)-attention_map.min(axis=-1,keepdims=True))

            for q in range(f):
                for k in range(f):
                    # heatmap 
                    attn_map = attention_map[...,q,k]
                    # normalize_heatmap
                    attn_map = (attn_map - attn_map.min())/(attn_map.max()-attn_map.min())
                    # make the attn_map to be heatmap
                    attn_map = cv2.applyColorMap((attn_map*255).astype(np.uint8), cv2.COLORMAP_JET)
                    # apply attn_map to canvas
                    canvas[q*h+gap*q:q*h+h+gap*q,k*w+gap*k:k*w+w+gap*k,:] = attn_map

            cv2.imwrite(f'{save_dir}/up_attnmap_{i}.png',canvas)


    def vis_down_attn_maps(self,attention_store=None,round=None):
        attention_shapes_h_w = self.attention_shapes_h_w
        if attention_store is None:
            attention_store = self.attention_store
        down_self = attention_store['down_self']


        if len(down_self) == 4:
            down_self = [sum(down_self[0:4])]
        elif len(down_self) == 0:
            return
        else:
            raise ValueError(f'down_self length {len(down_self)} is not supported')
        save_dir = f'{self.save_dir}/{"_".join(self.prompt.split())}_seed_{self.seed}'
        if round is not None:
            save_dir = f'{save_dir}/round_{round}'
        os.makedirs(save_dir, exist_ok=True)

        for i in range(len(down_self)):
            attention_map = down_self[i].detach().cpu().numpy()
            h, w = attention_shapes_h_w['down'][2-i]
            f = attention_map.shape[-2]
            gap = h // 5
            canvas = np.zeros((h*f + gap*f, w*f + gap*f,3), dtype='uint8') + 255
            attention_map = attention_map.reshape(h,w,-1,f,f)
            attention_map = attention_map.sum(axis=2) / attention_map.shape[2]

            # normalize along q axis
            # attention_map = (attention_map - attention_map.min(axis=-1,keepdims=True))/(attention_map.max(axis=-1,keepdims=True)-attention_map.min(axis=-1,keepdims=True))

            for q in range(f):
                for k in range(f):
                    # heatmap 
                    attn_map = attention_map[...,q,k]
                    # normalize_heatmap
                    attn_map = (attn_map - attn_map.min())/(attn_map.max()-attn_map.min())
                    # make the attn_map to be heatmap
                    attn_map = cv2.applyColorMap((attn_map*255).astype(np.uint8), cv2.COLORMAP_JET)
                    # apply attn_map to canvas
                    canvas[q*h+gap*q:q*h+h+gap*q,k*w+gap*k:k*w+w+gap*k,:] = attn_map

            cv2.imwrite(f'{save_dir}/down_attnmap_{i}.png',canvas)

    
    def save_attr(self,attr):
        save_dir = f'{self.save_dir}/{"_".join(self.prompt.split())}_seed_{self.seed}'
        save_tensor = getattr(self,attr)
        # torch.save(save_tensor,f'{save_dir}/{attr}.pt')

from typing import Any, Dict, Optional

def register_attention_control(unet, controller,  config=None):

    def BasicTransformerBlock_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            norm_hidden_states = norm_hidden_states.squeeze(1)
        else:
            raise ValueError("Incorrect norm used")

        # save the origin_hidden_states w/o pos_embed, for the use of motion v embedding
        origin_hidden_states = None
        if self.pos_embed is not None or hasattr(self.attn1,'vSpatial'):
            origin_hidden_states = norm_hidden_states.clone()
            if cross_attention_kwargs is None:
                cross_attention_kwargs = {}
            cross_attention_kwargs["origin_hidden_states"] = origin_hidden_states

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

            
        # 1. Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 2.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                # ziyang
                origin_hidden_states = norm_hidden_states.clone()
                norm_hidden_states = self.pos_embed(norm_hidden_states)
                cross_attention_kwargs["origin_hidden_states"] = origin_hidden_states

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states
            # ziyang
            if cross_attention_kwargs is not None and "origin_hidden_states" in cross_attention_kwargs:
                cross_attention_kwargs.pop("origin_hidden_states")

        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(
                self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size, lora_scale=lora_scale
            )
        else:
            ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


    def temp_attn_forward(self, place_in_unet, addition_info=None):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        
        def forward_until_attn_map(hidden_states, attr):
            batch_size, sequence_length, _ = hidden_states.shape
            attention_mask = self.prepare_attention_mask(None, sequence_length, batch_size)
            hidden_states = hidden_states
            query = self.to_q(hidden_states)
            encoder_hidden_states = hidden_states
            key = self.to_k(encoder_hidden_states)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)

            attention_probs = self.get_attention_scores(query, key, attention_mask)
            getattr(controller, attr)(attention_probs, False, place_in_unet)
            


        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None,temb=None,origin_hidden_states=None):
            is_cross = encoder_hidden_states is not None
            
            residual = hidden_states

            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            query = self.to_q(hidden_states)
            key = self.to_k(encoder_hidden_states)
            
            if addition_info is not None:
                if addition_info['removeMFromV']:
                    value = self.to_v(origin_hidden_states)
                elif hasattr(self,'vSpatial'):
                    if addition_info['vSpatial_frameSubtraction']:
                        value = self.to_v(self.vSpatial.forward_frameSubtraction(origin_hidden_states))
                    else:
                        value = self.to_v(self.vSpatial(origin_hidden_states))
                else:
                    value = self.to_v(origin_hidden_states)
            else:
                value = self.to_v(encoder_hidden_states)


            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)
            attention_probs = controller(attention_probs, is_cross, place_in_unet, self.heads)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = to_out(hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states
        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()
    
    from functools import partial
    def register_recr(net_, count, place_in_unet, name, config=None):

        print(name)

        if net_.__class__.__name__ == 'BasicTransformerBlock':
            BasicTransformerBlock_forward_ = partial(BasicTransformerBlock_forward, net_)
            net_.forward = BasicTransformerBlock_forward_

        if net_.__class__.__name__ == 'Attention':
            block_name = name.split('.attn')[0] 
            if config is not None and block_name in set([l.split('.attn')[0].split('.pos_embed')[0] for l in config.model.embedding_layers]):
                addition_info = {}
                addition_info['layer_name'] = name
                addition_info['removeMFromV'] = config.strategy.get('removeMFromV', False)
                addition_info['vSpatial_frameSubtraction'] = config.strategy.get('vSpatial_frameSubtraction', False)
                net_.forward = temp_attn_forward(net_, place_in_unet, addition_info)
                print('register Motion V embedding at ', block_name)
                return count + 1
            else:
                return count

        elif hasattr(net_, 'children'):
            for net_name, net__ in dict(net_.named_children()).items():
                count = register_recr(net__, count, place_in_unet, name = name + '.' + net_name, config=config)
        return count

    cross_att_count = 0
    sub_nets = unet.named_children()
    
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down", name = net[0], config=config)
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up", name = net[0], config=config)
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid",name = net[0], config=config)

    controller.num_att_layers = cross_att_count

