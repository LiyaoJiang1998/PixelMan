from src.models.pipeline import EditPipeline
from src.utils.utils import resize_numpy_image, process_move, roll_add_padding

import torch
from pytorch_lightning import seed_everything
from PIL import Image
from torchvision.transforms import PILToTensor, ToPILImage
import numpy as np
import torch.nn.functional as F
from basicsr.utils import img2tensor

SIZES = {
    0:4,
    1:2,
    2:1,
    3:1,
}

class EditModels():
    def __init__(self, pretrained_model_path, use_ip_adapter=False, steps=16):
        self.steps = steps
        self.ip_scale = 0.1
        self.precision = torch.float16
        self.editor = EditPipeline(sd_id=pretrained_model_path, NUM_DDIM_STEPS=self.steps, precision=self.precision, ip_scale=self.ip_scale, use_ip_adapter=use_ip_adapter)
        self.up_ft_index = [1,2] # fixed in gradio demo
        self.up_scale = 2        # fixed in gradio demo
        self.device = 'cuda'     # fixed in gradio demo

        self.use_ip_adapter = use_ip_adapter
            
    def run_move(self, 
                 original_image, 
                 mask, 
                 mask_ref, 
                 prompt, 
                 resize_scale, 
                 w_edit, 
                 w_content, 
                 w_contrast, 
                 w_inpaint, 
                 seed, 
                 selected_points, 
                 guidance_scale, 
                 energy_scale, 
                 max_resolution, 
                 SDE_strength, 
                 ip_scale=None,
                 use_gsn=True,
                 inversion_free=True,
                 sa_masking_ipt=True,
                 use_copy_paste=True,
                 ):
        seed_everything(seed)
        energy_scale = energy_scale*1e3
        img = original_image
        
        h, w = img.shape[1], img.shape[0]
        if h>w:
            max_resolution_h = h * (max_resolution/w)
            max_resolution_w = max_resolution
        elif w>h:
            max_resolution_h = max_resolution
            max_resolution_w = w * (max_resolution/h)
        else:
            max_resolution_h = max_resolution
            max_resolution_w = max_resolution
        img, input_scale = resize_numpy_image(img, max_resolution_h*max_resolution_w)
        h, w = img.shape[1], img.shape[0]
        print("Image reshaped size:", (h,w))
        
        img = Image.fromarray(img)
        img_prompt = img.resize((256, 256))
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        img_tensor = img_tensor.to(self.device, dtype=self.precision).unsqueeze(0)

        if mask_ref is not None and np.sum(mask_ref)!=0:
            mask_ref = np.repeat(mask_ref[:,:,None], 3, 2)
        else:
            mask_ref = None

        if self.use_ip_adapter:
            emb_im, emb_im_uncond = self.editor.get_image_embeds(img_prompt)
        else:
            emb_im, emb_im_uncond = None, None
        
        if ip_scale is not None and ip_scale != self.ip_scale:
            self.ip_scale = ip_scale
            self.editor.load_adapter(self.editor.ip_id, self.ip_scale)
        latent = self.editor.image2latent(img_tensor)
        clean_latents = self.editor.image2latent(img_tensor)
                
        if not inversion_free:
            ddim_latents = self.editor.ddim_inv(latent=latent, prompt=prompt)
            latent_in = ddim_latents[-1].squeeze(2)
        
        scale = 8*SIZES[max(self.up_ft_index)]/self.up_scale
        x=[]
        y=[]
        x_cur = []
        y_cur = []
        for idx, point in enumerate(selected_points):
            if idx%2 == 0:
                y.append(point[1])
                x.append(point[0])
            else:
                y_cur.append(point[1])
                x_cur.append(point[0])
        dx = x_cur[0]-x[0]
        dy = y_cur[0]-y[0]

        edit_kwargs = process_move(
            path_mask=mask, 
            h=h, 
            w=w, 
            dx=dx, 
            dy=dy, 
            scale=scale, 
            input_scale=input_scale, 
            resize_scale=resize_scale, 
            up_scale=self.up_scale, 
            up_ft_index=self.up_ft_index, 
            w_edit=w_edit, 
            w_content=w_content, 
            w_contrast=w_contrast, 
            w_inpaint=w_inpaint,  
            precision=self.precision, 
            path_mask_ref=mask_ref
        )
        
        if use_copy_paste:
            if not inversion_free:
                mask_tmp = (F.interpolate(img2tensor(mask)[0].unsqueeze(0).unsqueeze(0), (int(latent_in.shape[-2]*resize_scale), int(latent_in.shape[-1]*resize_scale)))>0).float().to('cuda', dtype=latent_in.dtype)
                latent_tmp = F.interpolate(latent_in, (int(latent_in.shape[-2]*resize_scale), int(latent_in.shape[-1]*resize_scale)))                

                p_l, p_r, p_t, p_b = roll_add_padding(padding_x=int(dx/(w/latent_in.shape[-2])*resize_scale), 
                                                      padding_y=int(dy/(w/latent_in.shape[-2])*resize_scale))
                mask_tmp = F.pad(mask_tmp, (p_l, p_r, p_t, p_b), "constant", 0.0)
                latent_tmp = F.pad(latent_tmp, (p_l, p_r, p_t, p_b), "constant", 0.0)

                mask_tmp = torch.roll(mask_tmp, (int(dy/(w/latent_in.shape[-2])*resize_scale), int(dx/(w/latent_in.shape[-2])*resize_scale)), (-2,-1))
                latent_tmp = torch.roll(latent_tmp, (int(dy/(w/latent_in.shape[-2])*resize_scale), int(dx/(w/latent_in.shape[-2])*resize_scale)), (-2,-1))

                mask_tmp = mask_tmp[:, :, p_t:p_t+int(latent_in.shape[-2]*resize_scale), p_l:p_l+int(latent_in.shape[-1]*resize_scale)]
                latent_tmp = latent_tmp[:, :, p_t:p_t+int(latent_in.shape[-2]*resize_scale), p_l:p_l+int(latent_in.shape[-1]*resize_scale)]

                pad_size_x = abs(mask_tmp.shape[-1]-latent_in.shape[-1])//2
                pad_size_y = abs(mask_tmp.shape[-2]-latent_in.shape[-2])//2
                if resize_scale>1:
                    sum_before = torch.sum(mask_tmp)
                    mask_tmp = mask_tmp[:,:,pad_size_y:pad_size_y+latent_in.shape[-2],pad_size_x:pad_size_x+latent_in.shape[-1]]
                    latent_tmp = latent_tmp[:,:,pad_size_y:pad_size_y+latent_in.shape[-2],pad_size_x:pad_size_x+latent_in.shape[-1]]
                    sum_after = torch.sum(mask_tmp)
                    if sum_after != sum_before:
                        raise ValueError('Resize out of bounds.')
                        exit(0)
                elif resize_scale<1:
                    temp = torch.zeros(1,1,latent_in.shape[-2], latent_in.shape[-1]).to(latent_in.device, dtype=latent_in.dtype)
                    temp[:,:,pad_size_y:pad_size_y+mask_tmp.shape[-2],pad_size_x:pad_size_x+mask_tmp.shape[-1]]=mask_tmp
                    mask_tmp =(temp>0.5).float()
                    temp = torch.zeros_like(latent_in)
                    temp[:,:,pad_size_y:pad_size_y+latent_tmp.shape[-2],pad_size_x:pad_size_x+latent_tmp.shape[-1]]=latent_tmp
                    latent_tmp = temp

                latent_in = (latent_in*(1-mask_tmp)+latent_tmp*mask_tmp).to(dtype=latent_in.dtype)

            else:
                # Pixel-space duplication
                latent = img_tensor
                mask_tmp = (F.interpolate((img2tensor(mask)[0].unsqueeze(0).unsqueeze(0)>0).float(), (int(latent.shape[-2]*resize_scale), int(latent.shape[-1]*resize_scale)), mode="bicubic", antialias=True)).to('cuda', dtype=latent.dtype)
                latent_tmp = F.interpolate(latent, (int(latent.shape[-2]*resize_scale), int(latent.shape[-1]*resize_scale)), mode="bicubic", antialias=True)

                p_l, p_r, p_t, p_b = roll_add_padding(padding_x=int(dx/(w/latent.shape[-2])*resize_scale), 
                                                    padding_y=int(dy/(w/latent.shape[-2])*resize_scale))
                mask_tmp = F.pad(mask_tmp, (p_l, p_r, p_t, p_b), "constant", 0.0)
                latent_tmp = F.pad(latent_tmp, (p_l, p_r, p_t, p_b), "constant", 0.0)

                mask_tmp = torch.roll(mask_tmp, (int(dy/(w/latent.shape[-2])*resize_scale), int(dx/(w/latent.shape[-2])*resize_scale)), (-2,-1))
                latent_tmp = torch.roll(latent_tmp, (int(dy/(w/latent.shape[-2])*resize_scale), int(dx/(w/latent.shape[-2])*resize_scale)), (-2,-1))

                mask_tmp = mask_tmp[:, :, p_t:p_t+int(latent.shape[-2]*resize_scale), p_l:p_l+int(latent.shape[-1]*resize_scale)]
                latent_tmp = latent_tmp[:, :, p_t:p_t+int(latent.shape[-2]*resize_scale), p_l:p_l+int(latent.shape[-1]*resize_scale)]

                pad_size_x = abs(mask_tmp.shape[-1]-latent.shape[-1])//2
                pad_size_y = abs(mask_tmp.shape[-2]-latent.shape[-2])//2
                if resize_scale>1:
                    sum_before = torch.sum(mask_tmp)
                    mask_tmp = mask_tmp[:,:,pad_size_y:pad_size_y+latent.shape[-2],pad_size_x:pad_size_x+latent.shape[-1]]
                    latent_tmp = latent_tmp[:,:,pad_size_y:pad_size_y+latent.shape[-2],pad_size_x:pad_size_x+latent.shape[-1]]
                    sum_after = torch.sum(mask_tmp)
                    if sum_after != sum_before:
                        raise ValueError('Resize out of bounds.')
                        exit(0)
                elif resize_scale<1:
                    temp = torch.zeros(1,1,latent.shape[-2], latent.shape[-1]).to(latent.device, dtype=latent.dtype)
                    temp[:,:,pad_size_y:pad_size_y+mask_tmp.shape[-2],pad_size_x:pad_size_x+mask_tmp.shape[-1]]=mask_tmp
                    mask_tmp = temp.float()
                    temp = torch.zeros_like(latent)
                    temp[:,:,pad_size_y:pad_size_y+latent_tmp.shape[-2],pad_size_x:pad_size_x+latent_tmp.shape[-1]]=latent_tmp
                    latent_tmp = temp
                
                latent = (latent*(1-mask_tmp)+latent_tmp*mask_tmp).to(dtype=latent.dtype)
                latent = self.editor.image2latent(latent)    
        
        if not inversion_free:
            if use_gsn:
                print("Method: edit_gsn_ddim")
                edit_func = self.editor.pipe.edit_gsn_ddim
            else:
                print("Method: edit_eg_ddim")
                edit_func = self.editor.pipe.edit_eg_ddim
            latent_rec = edit_func(
                mode = 'move',
                emb_im=emb_im,
                emb_im_uncond=emb_im_uncond,
                latent=latent_in, 
                prompt=prompt, 
                guidance_scale=guidance_scale, 
                energy_scale=energy_scale,  
                latent_noise_ref = ddim_latents, 
                SDE_strength=SDE_strength,
                edit_kwargs=edit_kwargs,
                num_inference_steps=self.steps,
                sa_masking_ipt=sa_masking_ipt,
            )
        else:
            if use_gsn and inversion_free:
                print("Method: edit_gsn_pixelman")
                edit_func = self.editor.pipe.edit_gsn_pixelman
            elif not use_gsn and inversion_free:
                print("Method: edit_eg_pixelman")
                edit_func = self.editor.pipe.edit_eg_pixelman
            else:
                raise ValueError("Method not implemented")
            latent_rec = edit_func(
                mode = 'move',
                emb_im=emb_im,
                emb_im_uncond=emb_im_uncond,
                clean_latents=clean_latents,
                clean_latents_manipulated=latent,
                prompt=prompt, 
                guidance_scale=guidance_scale, 
                energy_scale=energy_scale,  
                SDE_strength=SDE_strength,
                edit_kwargs=edit_kwargs,
                num_inference_steps=self.steps,
                sa_masking_ipt=sa_masking_ipt,
                use_copy_paste=use_copy_paste,
            )
        
        # Return output image
        self.editor.pipe.attention_store.reset()
        img_rec = self.editor.decode_latents(latent_rec)[:,:,::-1]
        torch.cuda.empty_cache()
        
        return [img_rec]
    