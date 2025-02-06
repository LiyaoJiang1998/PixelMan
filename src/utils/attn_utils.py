import os
import sys
import cv2
import abc
import math
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Optional, Union

from diffusers.models.attention_processor import Attention
from diffusers.models import UNet2DConditionModel
import torchvision.transforms as T
from torchvision.transforms import ToTensor
from torchvision.utils import save_image


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, attn_name: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, attn_name: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            attn = self.forward(attn, is_cross, attn_name)
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
    def get_empty_store(self):
        return {attn_name: [] for attn_name in self.attention_to_use}

    def forward(self, attn, is_cross: bool, attn_name: str):
        if attn_name in self.attention_to_use:
            if self.keep_grad and "attn2" in attn_name:
                self.step_store[attn_name].append(attn)
            else:
                self.step_store[attn_name].append(attn.detach())
        return attn

    def between_steps(self):
        with torch.no_grad():
            if len(self.attention_store) == 0:
                self.attention_store = self.step_store
            else:
                for key in self.attention_store:
                    for i in range(len(self.attention_store[key])):
                        self.attention_store[key][i] += self.step_store[key][i].detach()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]][0] for key in
                             self.attention_store}
                    
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        torch.cuda.empty_cache()

    def __init__(self, attention_to_use, device=None, keep_grad=False):
        super(AttentionStore, self).__init__()
        self.attention_to_use = attention_to_use
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.device = device
        self.keep_grad = keep_grad
        
    def get_attention_map(
        self,
        output_size=(16,16),
        token_ids=(0,),
        average_layers=True,
        cfg=True,
    ):
        cross_attention_maps = {}
        self_attention_maps = {}
        raw_attention_maps = self.get_average_attention()
        
        for layer in raw_attention_maps:
            if layer.endswith("attn2"):
                split_attention_maps = torch.stack(
                    raw_attention_maps[layer].chunk(2), dim=0)
                batch, channel, img_embed_len, text_embed_len = split_attention_maps.shape
                reshaped_split_attention_maps = (
                    split_attention_maps[
                        :, :, :, torch.tensor(list(token_ids))
                    ]
                    .reshape(
                        batch,
                        channel,
                        round(math.sqrt(img_embed_len/(output_size[1]/output_size[0]))), # h
                        round(math.sqrt(img_embed_len*(output_size[1]/output_size[0]))), # w
                        len(token_ids),
                    )
                    .permute(0, 1, 4, 2, 3)
                )

                resized_reshaped_split_attention_maps = (
                    torch.nn.functional.interpolate(
                        reshaped_split_attention_maps[0],
                        size=output_size,
                        mode="bilinear",
                    ).mean(dim=0)
                )
                if cfg:
                    resized_reshaped_split_attention_maps_1 = (
                        torch.nn.functional.interpolate(
                            reshaped_split_attention_maps[1],
                            size=output_size,
                            mode="bilinear",
                        ).mean(dim=0)
                    )
                    resized_reshaped_split_attention_maps = torch.stack(
                        [
                            resized_reshaped_split_attention_maps,
                            resized_reshaped_split_attention_maps_1,
                        ],
                        dim=0,
                    )
                cross_attention_maps[layer] = resized_reshaped_split_attention_maps

            elif layer.endswith("attn1"):  # self attentions
                channel, img_embed_len1, img_embed_len2 = raw_attention_maps[layer].shape
                if cfg:
                    split_attention_maps = raw_attention_maps[layer][channel // 2 :] # .to(torch.float32)
                else:
                    split_attention_maps = raw_attention_maps[layer] # .to(torch.float32)
                channel, img_embed_len1, img_embed_len2 = split_attention_maps.shape
                reshaped_split_attention_maps = (
                    split_attention_maps.reshape(
                        channel,
                        img_embed_len1,
                        round(math.sqrt(img_embed_len2/(output_size[1]/output_size[0]))), # h
                        round(math.sqrt(img_embed_len2*(output_size[1]/output_size[0]))), # w
                    )
                )
                resized_reshaped_split_attention_maps = torch.nn.functional.interpolate(
                    reshaped_split_attention_maps, size=output_size, mode="bilinear"
                )
                resized_reshaped_split_attention_maps = (
                    resized_reshaped_split_attention_maps.mean(dim=0)
                )
                self_attention_maps[layer] = resized_reshaped_split_attention_maps
            
        sd_cross_attention_maps = [None, None]
        sd_self_attention_maps = None
        
        if len(cross_attention_maps.values()) > 0:
            if average_layers:
                sd_cross_attention_maps = (
                    torch.stack(list(cross_attention_maps.values()), dim=0)
                    .mean(dim=0)
                    .to(self.device)
                )
            else:
                sd_cross_attention_maps = torch.stack(
                    list(cross_attention_maps.values()), dim=1
                ).to(self.device)
        if len(self_attention_maps.values()) > 0:
            sd_self_attention_maps = torch.stack(
                list(self_attention_maps.values()), dim=0
            ).mean(dim=0)

        if cfg:
            return (
                sd_cross_attention_maps[0],
                sd_cross_attention_maps[1],
                sd_self_attention_maps,
            )
        else:
            return (
                None,
                sd_cross_attention_maps[0],
                sd_self_attention_maps,
            )

