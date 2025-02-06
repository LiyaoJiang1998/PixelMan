
from diffusers import StableDiffusionPipeline
from typing import Any, Callable, Dict, List, Optional, Union
import torch.nn.functional as F
import torch
from basicsr.utils import img2tensor
from tqdm import tqdm
import torch.nn as nn
import copy
import numpy as np
import math
from basicsr.utils import tensor2img
from PIL import Image
import torchvision.transforms as T


def call_unet(self, latent_input, context, t, dict_mask, save_kv, mode, iter_cur, guidance_scale, sa_masking_ipt):
    if latent_input.shape[0] != 1:
        concat_latent_model_input = torch.cat(
            [
                latent_input[0].unsqueeze(0),
                latent_input[1].unsqueeze(0),
            ],
            dim=0,
        )
        concat_prompt_embeds = torch.cat(
            [
                context[0].unsqueeze(0),
                context[1].unsqueeze(0),
            ],
            dim=0,
        )
        
        with torch.no_grad():
            concat_noise_pred = self.unet(concat_latent_model_input, t, encoder_hidden_states=concat_prompt_embeds, 
                                          mask=dict_mask, save_kv=save_kv, mode=mode, iter_cur=iter_cur, sa_masking_ipt=sa_masking_ipt)["sample"].squeeze(2)
        (noise_pred_uncond, noise_pred_text) = concat_noise_pred.chunk(2, dim=0)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    else:
        # No CFG, null prompt
        with torch.no_grad():
            noise_pred = self.unet(latent_input, t, encoder_hidden_states=context[0].unsqueeze(0), 
                                   mask=dict_mask, save_kv=save_kv, mode=mode, iter_cur=iter_cur, sa_masking_ipt=sa_masking_ipt)["sample"].squeeze(2)
        
    return noise_pred


class Sampler(StableDiffusionPipeline):
    def edit_eg_ddim(
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
        sa_masking_ipt=False,
    ):  
        
        print('Start Editing:')
        self.alg=alg
        NFE = 0
        use_cfg = prompt != ""
        # generate source text embedding
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
        
        for i, t in enumerate(tqdm(self.scheduler.timesteps[-num_inference_steps:])):
            next_timestep = min(t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
            next_timestep = max(next_timestep, 0)
            if energy_scale==0 or alg=='D':
                repeat=1
            elif 20<i<30 and i%2==0 : 
                repeat = 3
            else:
                repeat = 1
            stack = []
            for ri in range(repeat):
                self.attention_store.reset() # Reset the attention_store before running denoising loop
                                
                if use_cfg:
                    latent_in = torch.cat([latent.unsqueeze(2)] * 2)
                    NFE += 1
                else:
                    latent_in = latent.unsqueeze(2)
                    NFE += 1
                latent_in = self.scheduler.scale_model_input(latent_in, t)
                noise_pred = call_unet(self, latent_in, context, t, dict_mask, False, mode, i, guidance_scale, sa_masking_ipt=sa_masking_ipt)
                
                with torch.no_grad():
                    map_output_size = (h//4, w//4)
                    ca_map_uncond, ca_map_cond, sa_map_cond = self.attention_store.get_attention_map(output_size=map_output_size, token_ids=(5,), average_layers=True, cfg=True)
                    self.attention_store.reset()
                    
                    # Save the m^sim region (similar object and shadow) from current QK^T SA map of m^gen pixels, 
                    img = sa_map_cond.float()
                    # Interpolate m_gen mask to match latent_in
                    m_gen = edit_kwargs["dict_mask"]['m_gen']
                    attn_res_h =  round(math.sqrt(img.shape[0]/(w/h)))
                    attn_res_w =  round(math.sqrt(img.shape[0]*(w/h)))
                    m_gen = (F.interpolate(m_gen[None,None].float(), (attn_res_h, attn_res_w))>0).reshape(-1)
                    # Process SA map to average SA map for m_gen pixels
                    img = img[m_gen,:,:].mean(dim=0)
                    img = (img - torch.min(img)) / (torch.max(img) - torch.min(img)) # scale to 0-1
                    m_sim = img >= 0.1 # threshold at 0.1 to binarize
                    edit_kwargs["dict_mask"]['m_sim'] = m_sim # update m^sim, use it for masking in next timestep
                                        
                if energy_scale!=0 and i<30 and (alg=='D' or i%2==0 or i<10):
                    # editing guidance
                    noise_pred_org = noise_pred
                    if mode == 'move':
                        guidance = self.guidance_move(latent=latent, latent_noise_ref=latent_noise_ref[-(i+1)], t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)
                        NFE += 2
                    noise_pred = noise_pred + guidance
                else:
                    noise_pred_org=None
                    
                # zt->zt-1
                prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
                beta_prod_t = 1 - alpha_prod_t
                pred_original_sample = (latent - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

                if 10<i<20:
                    eta, eta_rd = SDE_strength_un, SDE_strength
                else:
                    eta, eta_rd = 0., 0.
                
                try:
                    variance = self.scheduler._get_variance(t, prev_timestep)
                except:
                    variance = 0.0
                std_dev_t = eta * variance ** (0.5)
                std_dev_t_rd = eta_rd * variance ** (0.5)
                if noise_pred_org is not None:
                    pred_sample_direction_rd = (1 - alpha_prod_t_prev - std_dev_t_rd**2) ** (0.5) * noise_pred_org
                    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred_org
                else:
                    pred_sample_direction_rd = (1 - alpha_prod_t_prev - std_dev_t_rd**2) ** (0.5) * noise_pred
                    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred

                latent_prev = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
                latent_prev_rd = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction_rd

                # Regional SDE
                if (eta_rd > 0 or eta>0) and alg=='D+':
                    variance_noise = torch.randn_like(latent_prev)
                    variance_rd = std_dev_t_rd * variance_noise
                    variance = std_dev_t * variance_noise
                    
                    if mode == 'move':
                        mask = (F.interpolate(edit_kwargs["mask_x0"][None,None], (edit_kwargs["mask_cur"].shape[-2], edit_kwargs["mask_cur"].shape[-1]))>0.5).float()
                        mask = ((edit_kwargs["mask_cur"]+mask)>0.5).float()
                        mask = (F.interpolate(mask, (latent_prev.shape[-2], latent_prev.shape[-1]))>0.5).to(dtype=latent.dtype)
                    latent_prev = (latent_prev+variance)*(1-mask) + (latent_prev_rd+variance_rd)*mask

                if repeat>1:
                    with torch.no_grad():
                        alpha_prod_t = self.scheduler.alphas_cumprod[next_timestep]
                        alpha_prod_t_next = self.scheduler.alphas_cumprod[t]
                        beta_prod_t = 1 - alpha_prod_t
                        model_output = self.unet(latent_prev.unsqueeze(2), next_timestep, encoder_hidden_states=text_embeddings, 
                                                 mask=dict_mask, save_kv=False, mode=mode, iter_cur=-2, sa_masking_ipt=False)["sample"].squeeze(2)
                        NFE += 1
                        next_original_sample = (latent_prev - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
                        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
                        latent = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
            
            latent = latent_prev

        return latent

    def edit_gsn_ddim(
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
        sa_masking_ipt=False,
    ):  
        print('Start Editing:')
        self.alg=alg
        NFE = 0
        use_cfg = prompt != ""
        # generate source text embedding
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
        
        for i, t in enumerate(tqdm(self.scheduler.timesteps[-num_inference_steps:])):
            next_timestep = min(t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
            next_timestep = max(next_timestep, 0)
            if energy_scale==0 or alg=='D':
                repeat=1
            elif 20<i<30 and i%2==0 : 
                repeat = 3
            else:
                repeat = 1
            stack = []
            for ri in range(repeat):
                self.attention_store.reset() # Reset the attention_store before running denoising loop
                                
                if energy_scale!=0 and i<30 and (alg=='D' or i%2==0 or i<10):
                    # editing guidance
                    if mode == 'move':
                        guidance = self.guidance_move(latent=latent, latent_noise_ref=latent_noise_ref[-(i+1)], t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)
                        NFE += 2
                    latent = latent - 1e0*guidance
                    noise_pred_org=None
                else:
                    noise_pred_org=None
                                    
                if use_cfg:
                    latent_in = torch.cat([latent.unsqueeze(2)] * 2)
                    NFE += 1
                else:
                    latent_in = latent.unsqueeze(2)
                    NFE += 1
                latent_in = self.scheduler.scale_model_input(latent_in, t)
                noise_pred = call_unet(self, latent_in, context, t, dict_mask, False, mode, i, guidance_scale, sa_masking_ipt=sa_masking_ipt)

                with torch.no_grad():
                    map_output_size = (h//4, w//4)
                    ca_map_uncond, ca_map_cond, sa_map_cond = self.attention_store.get_attention_map(output_size=map_output_size, token_ids=(5,), average_layers=True, cfg=True) # 768 res -> 768/8=96 latent_res
                    self.attention_store.reset()
                    
                    # Save the m^sim region (similar object and shadow) from current QK^T SA map of m^gen pixels, 
                    img = sa_map_cond.float()
                    # Interpolate m_gen mask to match latent_in
                    m_gen = edit_kwargs["dict_mask"]['m_gen']
                    attn_res_h =  round(math.sqrt(img.shape[0]/(w/h)))
                    attn_res_w =  round(math.sqrt(img.shape[0]*(w/h)))
                    m_gen = (F.interpolate(m_gen[None,None].float(), (attn_res_h, attn_res_w))>0).reshape(-1)
                    # Process SA map to average SA map for m_gen pixels
                    img = img[m_gen,:,:].mean(dim=0)
                    img = (img - torch.min(img)) / (torch.max(img) - torch.min(img)) # scale to 0-1
                    m_sim = img >= 0.1 # threshold at 0.1 to binarize
                    edit_kwargs["dict_mask"]['m_sim'] = m_sim # update m^sim, use it for masking in next timestep

                # zt->zt-1
                prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
                beta_prod_t = 1 - alpha_prod_t

                if 10<i<20:
                    eta, eta_rd = SDE_strength_un, SDE_strength
                else:
                    eta, eta_rd = 0., 0.
                
                try:
                    variance = self.scheduler._get_variance(t, prev_timestep)
                except:
                    variance = 0.0
                std_dev_t = eta * variance ** (0.5)
                std_dev_t_rd = eta_rd * variance ** (0.5)

                pred_original_sample = (latent - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
                
                pred_sample_direction_rd = (1 - alpha_prod_t_prev - std_dev_t_rd**2) ** (0.5) * noise_pred
                pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred
                
                latent_prev = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
                latent_prev_rd = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction_rd

                # Regional SDE
                if (eta_rd > 0 or eta>0) and alg=='D+':
                    variance_noise = torch.randn_like(latent_prev)
                    variance_rd = std_dev_t_rd * variance_noise
                    variance = std_dev_t * variance_noise
                    
                    if mode == 'move':
                        mask = (F.interpolate(edit_kwargs["mask_x0"][None,None], (edit_kwargs["mask_cur"].shape[-2], edit_kwargs["mask_cur"].shape[-1]))>0.5).float()
                        mask = ((edit_kwargs["mask_cur"]+mask)>0.5).float()
                        mask = (F.interpolate(mask, (latent_prev.shape[-2], latent_prev.shape[-1]))>0.5).to(dtype=latent.dtype)
            latent = latent_prev
            
        return latent
        
    def edit_eg_pixelman(
        self,
        prompt:  List[str],
        mode,
        emb_im,
        emb_im_uncond,
        edit_kwargs,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.5,
        clean_latents: Optional[torch.FloatTensor] = None,
        clean_latents_manipulated: Optional[torch.FloatTensor] = None,
        start_time=50,
        energy_scale = 0,
        SDE_strength = 0.4,
        SDE_strength_un = 0,
        alg='D+',
        sa_masking_ipt=False,
        use_copy_paste=True,
    ):          
        print('Start Editing:')
        self.alg=alg
        NFE = 0
        use_cfg = prompt != ""
        # generate source text embedding
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
        h, w = clean_latents.shape[-2],clean_latents.shape[-1]
        
        pred_original_sample = clean_latents_manipulated        
        for i, t in enumerate(tqdm(self.scheduler.timesteps[-num_inference_steps:])):
            eps_con = torch.randn(clean_latents.shape, dtype=clean_latents.dtype, device=clean_latents.device)
            timesteps = torch.tensor([t])
            zt = self.scheduler.add_noise(pred_original_sample, eps_con, timesteps)
            zt_src = self.scheduler.add_noise(clean_latents, eps_con, timesteps)
            zt_copy = self.scheduler.add_noise(clean_latents_manipulated, eps_con, timesteps)
        
            next_timestep = min(t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
            next_timestep = max(next_timestep, 0)
            if energy_scale==0 or alg=='D':
                repeat=1
            elif int(num_inference_steps*0.4)<i<int(num_inference_steps*0.6) and i%2==0:
                repeat = 3
            else:
                repeat = 1
            stack = []
            for ri in range(repeat):
                self.attention_store.reset() # Reset the attention_store before running denoising loop
                
                if use_cfg:
                    zt_input = torch.cat([zt.unsqueeze(2)] * 2)
                    zt_src_input = torch.cat([zt_src.unsqueeze(2)] * 2)
                    zt_copy_input = torch.cat([zt_copy.unsqueeze(2)] * 2)
                    NFE += 3
                else:
                    zt_input = zt.unsqueeze(2)
                    zt_src_input = zt_src.unsqueeze(2)
                    zt_copy_input = zt_copy.unsqueeze(2)
                    NFE += 3
                zt_input = self.scheduler.scale_model_input(zt_input, t)
                zt_src_input = self.scheduler.scale_model_input(zt_src_input, t)
                zt_copy_input = self.scheduler.scale_model_input(zt_copy_input, t)
                
                noise_pred_copy = call_unet(self, zt_copy_input, context, t, dict_mask, True, mode, i, guidance_scale, sa_masking_ipt=False)
                self.attention_store.reset()
                noise_pred_src = call_unet(self, zt_src_input, context, t, dict_mask, True, mode, i, guidance_scale, sa_masking_ipt=False) 
                self.attention_store.reset()
                noise_pred = call_unet(self, zt_input, context, t, dict_mask, False, mode, i, guidance_scale, sa_masking_ipt=sa_masking_ipt) # inject K,V for z_tgt

                with torch.no_grad():
                    map_output_size = (h//4, w//4)
                    ca_map_uncond, ca_map_cond, sa_map_cond = self.attention_store.get_attention_map(output_size=map_output_size, token_ids=(5,), average_layers=True, cfg=use_cfg)
                    self.attention_store.reset()
                    
                    # Save the m^sim region (similar object and shadow) from current QK^T SA map of m^gen pixels, 
                    img = sa_map_cond.float()
                    # Interpolate m_gen mask to match latent_in
                    m_gen = edit_kwargs["dict_mask"]['m_gen']
                    attn_res_h =  round(math.sqrt(img.shape[0]/(w/h)))
                    attn_res_w =  round(math.sqrt(img.shape[0]*(w/h)))
                    m_gen = (F.interpolate(m_gen[None,None].float(), (attn_res_h, attn_res_w), mode="bicubic", antialias=True)>0).reshape(-1)
                    # Process SA map to average SA map for m_gen pixels
                    img = img[m_gen,:,:].mean(dim=0)
                    img = (img - torch.min(img)) / (torch.max(img) - torch.min(img)) # scale to 0-1
                    m_sim = img >= 0.1 # threshold at 0.1 to binarize
                    edit_kwargs["dict_mask"]['m_sim'] = m_sim # update m^sim, use it for masking in next timestep

                if energy_scale!=0 and i<int(num_inference_steps*0.6) and (alg=='D' or i%2==0 or i<int(num_inference_steps*0.2)):
                    # editing guidance
                    noise_pred_org = noise_pred
                    if mode == 'move':
                        guidance = self.guidance_move(latent=zt, latent_noise_ref=zt_copy, t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)   
                        NFE += 2
                    noise_pred = noise_pred + guidance
                else:
                    noise_pred_org=None
                    guidance = noise_pred - noise_pred

                if int(num_inference_steps*0.2)<i<int(num_inference_steps*0.4):
                    eta, eta_rd = SDE_strength_un, SDE_strength
                else:
                    eta, eta_rd = 0., 0.
                
                if i < num_inference_steps-2 and use_copy_paste:
                    mask_tmp = 1-(F.interpolate(edit_kwargs["dict_mask"]['m_gen'][None,None].float(), (int(noise_pred.shape[-2]), int(noise_pred.shape[-1])), 
                                            mode="bicubic", antialias=True)).to('cuda', dtype=noise_pred.dtype)

                    mask_tmp = T.functional.gaussian_blur(mask_tmp, kernel_size=9)
                else:
                    mask_tmp = torch.ones_like(noise_pred).to('cuda', dtype=noise_pred.dtype)
                    
                pred_z0 = self.scheduler.step(noise_pred, t, zt)[1]
                pred_z0_copy = self.scheduler.step(noise_pred_copy, t, zt_copy)[1]
                delta_zt = pred_z0 - pred_z0_copy
                
                pred_original_sample = clean_latents_manipulated + delta_zt * mask_tmp

        return pred_original_sample


    def edit_gsn_pixelman(
        self,
        prompt:  List[str],
        mode,
        emb_im,
        emb_im_uncond,
        edit_kwargs,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.5,
        clean_latents: Optional[torch.FloatTensor] = None,
        clean_latents_manipulated: Optional[torch.FloatTensor] = None,
        start_time=50,
        energy_scale = 0,
        SDE_strength = 0.4,
        SDE_strength_un = 0,
        alg='D+',
        sa_masking_ipt=False,
        use_copy_paste=True,
    ):          
        print('Start Editing:')
        self.alg=alg
        NFE = 0
        use_cfg = prompt != ""
        # generate source text embedding
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
        h, w = clean_latents.shape[-2],clean_latents.shape[-1]
        
        pred_original_sample = clean_latents_manipulated
        for i, t in enumerate(tqdm(self.scheduler.timesteps[-num_inference_steps:])):
            eps_con = torch.randn(clean_latents.shape, dtype=clean_latents.dtype, device=clean_latents.device)
            timesteps = torch.tensor([t])
            zt = self.scheduler.add_noise(pred_original_sample, eps_con, timesteps)
            zt_src = self.scheduler.add_noise(clean_latents, eps_con, timesteps)
            zt_copy = self.scheduler.add_noise(clean_latents_manipulated, eps_con, timesteps)
                
            next_timestep = min(t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
            next_timestep = max(next_timestep, 0)
            
            if energy_scale==0 or alg=='D':
                repeat=1
            elif int(num_inference_steps*0.4)<i<int(num_inference_steps*0.6) and i%2==0:
                repeat = 3
            else:
                repeat = 1
            
            for ri in range(repeat):
                self.attention_store.reset() # Reset the attention_store before running denoising loop

                if energy_scale!=0 and i<int(num_inference_steps*0.6) and (alg=='D' or i%2==0 or i<int(num_inference_steps*0.2)):
                    # editing guidance
                    if mode == 'move':
                        guidance = self.guidance_move(latent=zt, latent_noise_ref=zt_copy, t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)
                        NFE += 2
                    zt = zt - 1e0*guidance
                else:
                    guidance = zt - zt

            if use_cfg:
                zt_input = torch.cat([zt.unsqueeze(2)] * 2)
                zt_src_input = torch.cat([zt_src.unsqueeze(2)] * 2)
                zt_copy_input = torch.cat([zt_copy.unsqueeze(2)] * 2)
                NFE += 3
            else:
                zt_input = zt.unsqueeze(2)
                zt_src_input = zt_src.unsqueeze(2)
                zt_copy_input = zt_copy.unsqueeze(2)
                NFE += 3
            zt_input = self.scheduler.scale_model_input(zt_input, t)
            zt_src_input = self.scheduler.scale_model_input(zt_src_input, t)
            zt_copy_input = self.scheduler.scale_model_input(zt_copy_input, t)
            
            noise_pred_copy = call_unet(self, zt_copy_input, context, t, dict_mask, True, mode, i, guidance_scale, sa_masking_ipt=False)
            self.attention_store.reset()
            noise_pred_src = call_unet(self, zt_src_input, context, t, dict_mask, True, mode, i, guidance_scale, sa_masking_ipt=False) 
            self.attention_store.reset()
            noise_pred = call_unet(self, zt_input, context, t, dict_mask, False, mode, i, guidance_scale, sa_masking_ipt=sa_masking_ipt)

            with torch.no_grad():
                map_output_size = (h//4, w//4)
                ca_map_uncond, ca_map_cond, sa_map_cond = self.attention_store.get_attention_map(output_size=map_output_size, token_ids=(5,), average_layers=True, cfg=use_cfg) # 768 res -> 768/8=96 latent_res
                self.attention_store.reset()
                
                # Save the m^sim region (similar object and shadow) from current QK^T SA map of m^gen pixels, 
                img = sa_map_cond.float()
                # Interpolate m_gen mask to match latent_in
                m_gen = edit_kwargs["dict_mask"]['m_gen']
                attn_res_h =  round(math.sqrt(img.shape[0]/(w/h)))
                attn_res_w =  round(math.sqrt(img.shape[0]*(w/h)))
                m_gen = (F.interpolate(m_gen[None,None].float(), (attn_res_h, attn_res_w), mode="bicubic", antialias=True)>0).reshape(-1)
                # Process SA map to average SA map for m_gen pixels
                img = img[m_gen,:,:].mean(dim=0)
                img = (img - torch.min(img)) / (torch.max(img) - torch.min(img)) # scale to 0-1
                m_sim = img >= 0.1 # threshold at 0.1 to binarize
                edit_kwargs["dict_mask"]['m_sim'] = m_sim # update m^sim, use it for masking in next timestep
            
            if int(num_inference_steps*0.2)<i<int(num_inference_steps*0.4):
                eta, eta_rd = SDE_strength_un, SDE_strength
            else:
                eta, eta_rd = 0., 0.

            if i < num_inference_steps-2 and use_copy_paste: # only last 2 steps generates naturally (to avoid aliased edges, but keep it small (e.g. 4) to preseve object details)
                mask_tmp = 1-(F.interpolate(edit_kwargs["dict_mask"]['m_gen'][None,None].float(), (int(noise_pred.shape[-2]), int(noise_pred.shape[-1])), 
                                        mode="bicubic", antialias=True)).to('cuda', dtype=noise_pred.dtype)
                mask_tmp = T.functional.gaussian_blur(mask_tmp, kernel_size=9)
            else:
                mask_tmp = torch.ones_like(noise_pred).to('cuda', dtype=noise_pred.dtype)
                
            pred_z0 = self.scheduler.step(noise_pred, t, zt)[1]
            pred_z0_copy = self.scheduler.step(noise_pred_copy, t, zt_copy)[1]
            delta_zt = pred_z0 - pred_z0_copy

            pred_original_sample = clean_latents_manipulated + delta_zt * mask_tmp
            
        return pred_original_sample
        
    def guidance_move(
        self, 
        mask_x0, 
        mask_x0_ref, 
        mask_tar, 
        mask_cur, 
        mask_other, 
        mask_overlap, 
        mask_non_overlap,
        latent, 
        latent_noise_ref, 
        t, 
        up_ft_index, 
        text_embeddings, 
        up_scale, 
        resize_scale, 
        energy_scale,
        w_edit,
        w_content,
        w_contrast,
        w_inpaint, 
        dict_mask
    ):
        cos = nn.CosineSimilarity(dim=1)
        loss_scale = [0.5, 0.5]
        with torch.no_grad():
            up_ft_tar = self.estimator(
                        sample=latent_noise_ref.squeeze(2),
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings)['up_ft']
            up_ft_tar_org = copy.deepcopy(up_ft_tar)
            for f_id in range(len(up_ft_tar_org)):
                up_ft_tar_org[f_id] = F.interpolate(up_ft_tar_org[f_id], (up_ft_tar_org[-1].shape[-2]*up_scale, up_ft_tar_org[-1].shape[-1]*up_scale))

        latent = latent.detach().requires_grad_(True)
        for f_id in range(len(up_ft_tar)):
            up_ft_tar[f_id] = F.interpolate(up_ft_tar[f_id], (int(up_ft_tar[-1].shape[-2]*resize_scale*up_scale), int(up_ft_tar[-1].shape[-1]*resize_scale*up_scale)))

        up_ft_cur = self.estimator(
                    sample=latent,
                    timestep=t,
                    up_ft_indices=up_ft_index,
                    encoder_hidden_states=text_embeddings)['up_ft']
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(up_ft_cur[f_id], (up_ft_cur[-1].shape[-2]*up_scale, up_ft_cur[-1].shape[-1]*up_scale))
        # editing energy
        loss_edit = 0
        for f_id in range(len(up_ft_tar)):
            up_ft_cur_vec = up_ft_cur[f_id][mask_cur.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
            up_ft_tar_vec = up_ft_tar[f_id][mask_tar.repeat(1,up_ft_tar[f_id].shape[1],1,1)].view(up_ft_tar[f_id].shape[1], -1).permute(1,0)
            sim = cos(up_ft_cur_vec, up_ft_tar_vec)
            sim_global = cos(up_ft_cur_vec.mean(0, keepdim=True), up_ft_tar_vec.mean(0, keepdim=True))
            loss_edit = loss_edit + (w_edit/(1+4*sim.mean()))*loss_scale[f_id] 

        # content energy
        loss_con = 0
        if mask_x0_ref is not None:
            mask_x0_ref_cur = F.interpolate(mask_x0_ref[None,None], (mask_other.shape[-2], mask_other.shape[-1]))>0.5
        else:
            mask_x0_ref_cur = mask_other
        for f_id in range(len(up_ft_tar_org)):
            sim_other = cos(up_ft_tar_org[f_id], up_ft_cur[f_id])[0][mask_other[0,0]]
            loss_con = loss_con+w_content/(1+4*sim_other.mean())*loss_scale[f_id]

        for f_id in range(len(up_ft_tar)):
            # w_contrast
            up_ft_cur_non_overlap = up_ft_cur[f_id][mask_non_overlap.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
            up_ft_tar_non_overlap = up_ft_tar_org[f_id][mask_non_overlap.repeat(1,up_ft_tar_org[f_id].shape[1],1,1)].view(up_ft_tar_org[f_id].shape[1], -1).permute(1,0)
            sim_non_overlap = (cos(up_ft_cur_non_overlap, up_ft_tar_non_overlap)+1.)/2.
            loss_con = loss_con + w_contrast*sim_non_overlap.mean()*loss_scale[f_id]

            # w_inpaint
            up_ft_cur_non_overlap = up_ft_cur[f_id][mask_non_overlap.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1],-1).permute(1,0).mean(0, keepdim=True)
            up_ft_tar_non_overlap = up_ft_tar_org[f_id][mask_x0_ref_cur.repeat(1,up_ft_tar_org[f_id].shape[1],1,1)].view(up_ft_tar_org[f_id].shape[1],-1).permute(1,0).mean(0, keepdim=True)
            sim_inpaint = ((cos(up_ft_cur_non_overlap, up_ft_tar_non_overlap)+1.)/2.)
            loss_con = loss_con + w_inpaint/(1+4*sim_inpaint.mean())

        cond_grad_edit = torch.autograd.grad(loss_edit*energy_scale, latent, retain_graph=True)[0]
        cond_grad_con = torch.autograd.grad(loss_con*energy_scale, latent)[0]
        mask_edit2 = (F.interpolate(mask_x0[None,None], (mask_cur.shape[-2], mask_cur.shape[-1]))>0.5).float()
        mask_edit1 = (mask_cur>0.5).float()
        mask = ((mask_cur+mask_edit2)>0.5).float()
        mask_edit1 = (F.interpolate(mask_edit1, (latent.shape[-2], latent.shape[-1]))>0).to(dtype=latent.dtype)
        guidance = cond_grad_edit.detach()*8e-2*mask_edit1 + cond_grad_con.detach()*8e-2*(1-mask_edit1)
        self.estimator.zero_grad()

        return guidance
