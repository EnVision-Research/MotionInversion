import abc

LOW_RESOURCE = False
import torch 
import cv2 
import torch
import os
import numpy as np
from collections import defaultdict
from functools import partial
from typing import Any, Dict, Optional

def register_attention_control(unet, config=None):

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
                # save the origin_hidden_states
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
            # delete the origin_hidden_states
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


    def temp_attn_forward(self, additional_info=None):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None,temb=None,origin_hidden_states=None):
            
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
            
            # strategies to manipulate the motion value embedding
            if additional_info is not None:
                # empirically, in the inference stage of camera motion
                # discarding the motion value embedding improves the text similarity of the generated video
                if additional_info['removeMFromV']:
                    value = self.to_v(origin_hidden_states)
                elif hasattr(self,'vSpatial'):
                    # during inference, the debiasing operation helps to generate more diverse videos
                    # refer to the 'Figure.3 Right' in the paper for more details
                    if additional_info['vSpatial_frameSubtraction']:
                        value = self.to_v(self.vSpatial.forward_frameSubtraction(origin_hidden_states))
                    # during training, do not apply debias operation for motion learning
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

    def register_recr(net_, count, name, config=None):

        if net_.__class__.__name__ == 'BasicTransformerBlock':
            BasicTransformerBlock_forward_ = partial(BasicTransformerBlock_forward, net_)
            net_.forward = BasicTransformerBlock_forward_

        if net_.__class__.__name__ == 'Attention':
            block_name = name.split('.attn')[0] 
            if config is not None and block_name in set([l.split('.attn')[0].split('.pos_embed')[0] for l in config.model.embedding_layers]):
                additional_info = {}
                additional_info['layer_name'] = name
                additional_info['removeMFromV'] = config.strategy.get('removeMFromV', False)
                additional_info['vSpatial_frameSubtraction'] = config.strategy.get('vSpatial_frameSubtraction', False)
                net_.forward = temp_attn_forward(net_, additional_info)
                print('register Motion V embedding at ', block_name)
                return count + 1
            else:
                return count

        elif hasattr(net_, 'children'):
            for net_name, net__ in dict(net_.named_children()).items():
                count = register_recr(net__, count, name = name + '.' + net_name, config=config)
        return count

    sub_nets = unet.named_children()
    
    for net in sub_nets:
        register_recr(net[1], 0,name = net[0], config=config)

    

