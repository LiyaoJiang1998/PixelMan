import numpy as np
import cv2
from basicsr.utils import img2tensor
import torch
import torch.nn.functional as F

def resize_numpy_image(image, max_resolution=768 * 768, resize_short_edge=None):
    h, w = image.shape[:2]
    w_org = image.shape[1]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    scale = w/w_org
    return image, scale

def split_ldm(ldm):
    x = []
    y = []
    for p in ldm:
        x.append(p[0])
        y.append(p[1])
    return x,y

def roll_add_padding(padding_x, padding_y):
    if padding_x > 0:
        p_l, p_r = 0, padding_x
    else:
        p_l, p_r = -padding_x, 0
    if padding_y > 0:
        p_t, p_b = 0, padding_y
    else:
        p_t, p_b = -padding_y, 0
                
    return p_l, p_r, p_t, p_b

def process_move(path_mask, h, w, dx, dy, scale, input_scale, resize_scale, up_scale, up_ft_index, w_edit, w_content, w_contrast, w_inpaint,  precision, path_mask_ref=None):
    dx, dy = dx*input_scale, dy*input_scale
    if isinstance(path_mask, str):
        mask_x0 = cv2.imread(path_mask)
    else:
        mask_x0 = path_mask
    mask_x0 = cv2.resize(mask_x0, (h, w))
    if len(mask_x0.shape) == 2:
        mask_x0 = np.expand_dims(mask_x0, axis=2)
    if path_mask_ref is not None:
        if isinstance(path_mask_ref, str):
            mask_x0_ref = cv2.imread(path_mask_ref)
        else:
            mask_x0_ref = path_mask_ref
        mask_x0_ref = cv2.resize(mask_x0_ref, (h, w))
    else:
        mask_x0_ref=None

    mask_x0 = img2tensor(mask_x0)[0]
    mask_x0 = (mask_x0>0.5).float().to('cuda', dtype=precision)
    if mask_x0_ref is not None:
        mask_x0_ref = img2tensor(mask_x0_ref)[0]
        mask_x0_ref = (mask_x0_ref>0.5).float().to('cuda', dtype=precision)
    mask_org = F.interpolate(mask_x0[None,None], (int(mask_x0.shape[-2]//scale), int(mask_x0.shape[-1]//scale)))>0.5

    mask_tar = F.interpolate(mask_x0[None,None], (int(mask_x0.shape[-2]//scale*resize_scale), int(mask_x0.shape[-1]//scale*resize_scale)))>0.5

    p_l, p_r, p_t, p_b = roll_add_padding(padding_x=int(dx//scale*resize_scale), padding_y=int(dy//scale*resize_scale))
    mask_cur = F.pad(mask_tar, (p_l, p_r, p_t, p_b), "constant", 0.0)
    mask_cur = torch.roll(mask_cur, (int(dy//scale*resize_scale), int(dx//scale*resize_scale)), (-2,-1))
    mask_cur = mask_cur[:, :, p_t:p_t+int(mask_x0.shape[-2]//scale*resize_scale), p_l:p_l+int(mask_x0.shape[-1]//scale*resize_scale)]

    mask_tar = F.pad(mask_tar, (p_l, p_r, p_t, p_b), "constant", 0.0)
    mask_tar = torch.roll(mask_tar, (int(dy//scale*resize_scale), int(dx//scale*resize_scale)), (-2,-1))
    mask_tar = mask_tar[:, :, p_t:p_t+int(mask_x0.shape[-2]//scale*resize_scale), p_l:p_l+int(mask_x0.shape[-1]//scale*resize_scale)]
    mask_tar = torch.roll(mask_tar, (-int(dy//scale*resize_scale), -int(dx//scale*resize_scale)), (-2,-1))

    pad_size_x = abs(mask_tar.shape[-1]-mask_org.shape[-1])//2
    pad_size_y = abs(mask_tar.shape[-2]-mask_org.shape[-2])//2
    if resize_scale>1:
        sum_before = torch.sum(mask_cur)
        mask_cur = mask_cur[:,:,pad_size_y:pad_size_y+mask_org.shape[-2],pad_size_x:pad_size_x+mask_org.shape[-1]]
        sum_after = torch.sum(mask_cur)
        if sum_after != sum_before:
            raise ValueError('Resize out of bounds, exiting.')
    else:
        temp = torch.zeros(1,1,mask_org.shape[-2], mask_org.shape[-1]).to(mask_org.device)
        temp[:,:,pad_size_y:pad_size_y+mask_cur.shape[-2],pad_size_x:pad_size_x+mask_cur.shape[-1]]=mask_cur
        mask_cur =temp>0.5

    mask_other = (1-((mask_cur+mask_org)>0.5).float())>0.5
    mask_overlap = ((mask_cur.float()+mask_org.float())>1.5).float()
    mask_non_overlap = (mask_org.float()-mask_overlap)>0.5
    
    dict_mask = {}
    dict_mask['m_gud'] = mask_org[0,0]
    dict_mask['m_gud_tar'] = mask_tar[0,0]
    dict_mask['m_gen'] = mask_cur[0,0]
    dict_mask['m_share'] = mask_other[0,0]
    if mask_x0_ref is not None:
        dict_mask['m_ref'] = mask_x0_ref[0,0]
    else:
        dict_mask['m_ref'] = None
    dict_mask['m_ipt'] = mask_non_overlap[0,0]
    
    return {
        "dict_mask":dict_mask,
        "mask_x0":mask_x0, 
        "mask_x0_ref":mask_x0_ref, 
        "mask_tar":mask_tar, 
        "mask_cur":mask_cur, 
        "mask_other":mask_other, 
        "mask_overlap":mask_overlap, 
        "mask_non_overlap":mask_non_overlap, 
        "up_scale":up_scale,
        "up_ft_index":up_ft_index,
        "resize_scale":resize_scale,
        "w_edit":w_edit,
        "w_content":w_content,
        "w_contrast":w_contrast,
        "w_inpaint":w_inpaint, 
    }

