
from diffusers import StableDiffusionPipeline
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch.nn.functional as F
import torch
from basicsr.utils import img2tensor
from tqdm import tqdm
import torch.nn as nn
import copy
import numpy as np

from collections import defaultdict
import diffusers
import torchvision.transforms.functional as TF
import einops
import math

class Sampler(StableDiffusionPipeline):

    @torch.no_grad()
    def edit(
        self,
        prompt:  List[str],
        mode,
        emb_im,
        emb_im_uncond,
        edit_kwargs,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.5,
        latent: Optional[torch.FloatTensor] = None,
        start_time=50,
        energy_scale = 0,
        SDE_strength = 0.4,
        SDE_strength_un = 0,
        latent_noise_ref = None,
        alg='D+',
        words=[],
        shift = (0.0, 0.0),
    ):  

        print('Start Editing:')
        self.alg=alg
        NFE = 0
        
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
                [""], padding="max_length", max_length=max_length, return_tensors="pt"
            )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        # image prompt
        if emb_im is not None and emb_im_uncond is not None:
            uncond_embeddings = torch.cat([uncond_embeddings, emb_im_uncond],dim=1)
            text_embeddings_org = text_embeddings
            text_embeddings = torch.cat([text_embeddings, emb_im],dim=1)
            context = torch.cat([uncond_embeddings.expand(*text_embeddings.shape), text_embeddings])
        else:
            # No IP Adapter
            text_embeddings_org = text_embeddings
            context = torch.cat([uncond_embeddings, text_embeddings])

        self.scheduler.set_timesteps(num_inference_steps) 
        dict_mask = edit_kwargs['dict_mask'] if 'dict_mask' in edit_kwargs else None
        h, w = latent.shape[-2],latent.shape[-1]
        
        self.attention_store.reset() # Reset the attention_store before running denoising loop
        
        # Self-Guidance on CA of provided words
        if len(words) > 0:
            prompt_text_ids = self.tokenizer(prompt, return_tensors='np')['input_ids'][0]
            token_ids = []
            for word in words:
                word_ids = self.tokenizer(word, return_tensors='np')['input_ids']
                word_ids = word_ids[word_ids < 49406]
                token_ids.append(search_sequence_numpy(prompt_text_ids, word_ids))
            token_ids = tuple(np.concatenate(token_ids).tolist())
            print("Prompt: %s"%(prompt))
            print("Editing words: %s, Token ids: %s"%(words, token_ids))
        else:
            token_ids = (0,)
        
        for i, t in enumerate(tqdm(self.scheduler.timesteps[-start_time:])):
            next_timestep = min(t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
            next_timestep = max(next_timestep, 0)
            
            repeat = 1
            stack = []
            do_self_guidance = energy_scale!=0 and i<30 and len(words) > 0
            for ri in range(repeat):
                with torch.set_grad_enabled(do_self_guidance):
                    latent = latent.clone().detach().requires_grad_(do_self_guidance)
                    latent_in = torch.cat([latent.unsqueeze(2)] * 2)
                    noise_pred = self.unet(latent_in, t, encoder_hidden_states=context, mask=dict_mask, save_kv=False, mode=mode, iter_cur=i)["sample"].squeeze(2)
                    self.unet.zero_grad()
                    NFE += 1
                    
                    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
                    
                    map_output_size = (h//4, w//4)
                    ca_map_uncond, ca_map_cond, sa_map_cond = self.attention_store.get_attention_map(output_size=map_output_size, token_ids=token_ids, average_layers=True, cfg=True) # 768 res -> 768/8=96 latent_res
                    self.attention_store.reset()
                                                                            
                    if do_self_guidance:
                        # editing guidance
                        noise_pred_org = noise_pred
                        if mode == 'move':
                            guidance = self.guidance_move_sg(latent=latent, ca_map=ca_map_cond, shift=shift, energy_scale=energy_scale)
                        noise_pred = noise_pred + guidance
                    else:
                        noise_pred_org=None
                    
                # zt->zt-1
                prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
                beta_prod_t = 1 - alpha_prod_t
                pred_original_sample = (latent - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

                eta = 0.0
                try:
                    variance = self.scheduler._get_variance(t, prev_timestep)
                except:
                    variance = 0.0
                std_dev_t = eta * variance ** (0.5)
                if noise_pred_org is not None:
                    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred_org
                else:
                    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred

                latent_prev = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
            
            latent = latent_prev
            latent = latent.detach()
                    
        # print("SelfGuidance NFE:", NFE+num_inference_steps)
        return latent


    def guidance_move_sg(self, latent, ca_map, shift,
                        energy_scale=None, sg_loss_rescale = 1000.):                                                                                                                                                                                                                       
        sg_loss = SelfGuidanceEdits.centroid(ca_map=ca_map, shift=shift)
        # sg_loss_rescale avoid underflow
        sg_grad = torch.autograd.grad(sg_loss_rescale * sg_loss, latent, retain_graph=False)[0] / sg_loss_rescale
        guidance = energy_scale * sg_grad.detach()
                
        self.unet.zero_grad()
        return guidance
    

# From: Self-Guidance Demo https://colab.research.google.com/drive/1SEM1R9mI9cF-aFpqg3NqHP8gN8irHuJi?usp=sharing
class SelfGuidanceEdits:
    @staticmethod
    def _centroid(a):
        x = torch.linspace(0, 1, a.shape[2]).to(a.device)
        y = torch.linspace(0, 1, a.shape[1]).to(a.device)
        # a is (n, h, w)
        attn_x = a.sum(1)  # (n, w)
        attn_y = a.sum(2)  # (n, h)

        def f(_attn, _linspace):
            _attn = _attn / (_attn.sum(1, keepdim=True) + 1e-4)  # (n, 1)
            _weighted_attn = _linspace[None, ...] * _attn  # (n, h or w)
            return _weighted_attn.sum(1)  # (n)

        centroid_x = f(attn_x, x)
        centroid_y = f(attn_y, y)
        centroid = torch.stack((centroid_x, centroid_y), -1)  # (n, 2)
        return centroid

    @staticmethod
    def centroid(ca_map, shift=(0.0, 0.0)):
        obs_centroid = SelfGuidanceEdits._centroid(ca_map)
        
        shift = torch.tensor(shift).to(ca_map.device)
        shift = shift.reshape((1,) * (obs_centroid.ndim - shift.ndim) + shift.shape)
        
        tgt_centroid = obs_centroid.detach() + shift
                
        return (obs_centroid - tgt_centroid).abs().mean()
    
def search_sequence_numpy(arr, seq):
    Na, Nseq = arr.size, seq.size

    r_seq = np.arange(Nseq)

    M = (arr[np.arange(Na - Nseq + 1)[:, None] + r_seq] == seq).all(1)

    if M.any() > 0:
        return np.where(np.convolve(M, np.ones((Nseq), dtype=int)) > 0)[0]
    else:
        return []  # No match found
    
