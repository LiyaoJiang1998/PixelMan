import sys
sys.path.append(".")
sys.path.append("..")
import argparse
from tqdm import tqdm
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import PILToTensor, ToPILImage
from PIL import Image
from basicsr.utils import img2tensor, tensor2img

from run_move_static import get_data_ReS, get_data_COCOEE
from src.utils.utils import process_move, roll_add_padding
from lavis.models import load_model_and_preprocess
import clip
import pyiqa


def load_metrics(device):
    # 1. Semantic Consistency: CLIP (BLIP caption of origianl vs. edited)
    clip_model, clip_preprocess = clip.load("models/ViT-B-16.pt", device)
    clip_model.eval()
    
    # 2. Overall no-ref IQA (Harmony, image quality): TOPIQ, MUSIQ, LIQE
    metric_topiq_nr = pyiqa.create_metric('topiq_nr', as_loss=False, device=device)
    metric_musiq = pyiqa.create_metric('musiq', as_loss=False, device=device)
    metric_liqe = pyiqa.create_metric('liqe_mix', as_loss=False, device=device)

    # 3. Consistency: (full-ref: LPIPS, PSNR) 
    metric_lpips = pyiqa.create_metric('lpips', as_loss=False, device=device)
    metric_psnr = pyiqa.create_metric('psnr', device=device)
    
    metrics = {
        "clip_model" : clip_model,
        "clip_preprocess" : clip_preprocess,
        "topiq_nr": metric_topiq_nr,
        "musiq": metric_musiq,
        "liqe": metric_liqe,
        "lpips": metric_lpips,
        "psnr": metric_psnr,
    }
    
    return metrics


def load_blip_model(device):
    blip_model, blip_vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
    return blip_model, blip_vis_processors


def interpolate_mask(mask, img_tensor, resize_scale, device):
    mask = F.interpolate((img2tensor(np.expand_dims(mask.detach().cpu().numpy(), axis=2))[0].unsqueeze(0).unsqueeze(0)>0).float(),
                              (int(img_tensor.shape[-2]*resize_scale), int(img_tensor.shape[-1]*resize_scale)), mode="bicubic", antialias=True).to(device, dtype=img_tensor.dtype)
    return mask


def copy_gud_to_gen(img_tensor, mask, resize_scale, dx, dy, h, w, device):

    latent = img_tensor
    mask_tmp = (F.interpolate((mask>0).float(), (int(latent.shape[-2]*resize_scale), int(latent.shape[-1]*resize_scale)), mode="bicubic", antialias=True)).to(device, dtype=latent.dtype)
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
    
    return latent


def obtain_masked_images(source_image, edited_image, source_mask, selected_points, device):
    # Prepare arguments for process_move:
    resize_scale = 1.0
    SIZES = {0:4, 1:2, 2:1, 3:1}
    up_ft_index = [1,2]
    up_scale = 2
    input_scale = 1.0
    h, w = source_image.shape[1], source_image.shape[0]
            
    scale = 8*SIZES[max(up_ft_index)]/up_scale
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
    
    source_image = ((PILToTensor()(Image.fromarray(source_image[:, :, ::-1])) / 255.0)).to(device, dtype=torch.float16).unsqueeze(0)
    edited_image = ((PILToTensor()(Image.fromarray(edited_image[:, :, ::-1])) / 255.0)).to(device, dtype=torch.float16).unsqueeze(0)
    
    # Run process_move to get different masks
    edit_kwargs = process_move(
        path_mask=source_mask, 
        h=h, w=w, 
        dx=dx, dy=dy, 
        scale=scale, input_scale=input_scale, 
        resize_scale=resize_scale, 
        up_scale=up_scale, up_ft_index=up_ft_index, 
        w_edit=0, w_content=0, w_contrast=0, w_inpaint=0,  
        precision=torch.float16, 
        path_mask_ref=None
    )
    dict_mask = edit_kwargs["dict_mask"]
    m_gud = interpolate_mask(dict_mask["m_gud"], source_image, resize_scale, device)
    m_gen = interpolate_mask(dict_mask["m_gen"], source_image, resize_scale, device)
    m_share = interpolate_mask(dict_mask["m_share"], source_image, resize_scale, device)
    m_ipt = interpolate_mask(dict_mask["m_ipt"], source_image, resize_scale, device)
    
    # Take source image, shift the m_gud region to m_gen region
    source_image_gud_shifted = copy_gud_to_gen(source_image, m_gud, resize_scale, dx, dy, h, w, device)
    
    source_image_share = tensor2img(source_image * m_share)
    edited_image_share = tensor2img(edited_image * m_share)
    source_image_ipt = tensor2img(source_image * m_ipt)
    edited_image_ipt = tensor2img(edited_image * m_ipt)
    edited_image_gen = tensor2img(edited_image * m_gen)
    source_image_gud_shifted = tensor2img(source_image_gud_shifted * m_gen)

    return source_image_share, edited_image_share, \
           source_image_ipt, edited_image_ipt, \
           edited_image_gen, source_image_gud_shifted
    
    
def evaluate_metrics(source_image, edited_image, 
                     source_mask, selected_points,
                     metrics, blip_model, blip_vis_processors, device):
    """
    Args:
        source_image (_type_): np array with shape (C, H, W)
        edited_image (_type_): np array with shape (C, H, W)
    """
    # Resize source_image to match edited_image shape
    source_image = torch.from_numpy(source_image).unsqueeze(0)
    source_image = F.interpolate(source_image, (int(edited_image.shape[-2]), int(edited_image.shape[-1])), mode="bicubic", antialias=True)
    source_image = source_image.squeeze(0).detach().cpu().numpy()
    
    results_dict = {"source_caption": "",
                    "edited_caption": "",
                    "global_iqa": {},
                    "object_consistency": {},
                    "background_consistency": {},
                    "inpainting_similarity": {},
                    "semantic": {},}
    
    # use BLIP to obtain source and edited captions
    blip_source_image = blip_vis_processors["eval"](Image.fromarray(source_image)).unsqueeze(0).to(device)
    source_caption = blip_model.generate({"image": blip_source_image})[0]
    blip_edited_image = blip_vis_processors["eval"](Image.fromarray(edited_image)).unsqueeze(0).to(device)
    edited_caption = blip_model.generate({"image": blip_edited_image})[0]
    results_dict["source_caption"] = source_caption
    results_dict["edited_caption"] = edited_caption
    
    # Process the masks, to obtain images with specific regions masked (For regional consistency evaluation)
    source_image_share, edited_image_share, \
           source_image_ipt, edited_image_ipt, \
           edited_image_gen, source_image_gud_shifted = obtain_masked_images(source_image, edited_image, source_mask, selected_points, device)

    with torch.no_grad():
        # Global IQA Metrics
        for metric_name in ["topiq_nr", "musiq", "liqe"]:
            score = float(metrics[metric_name](Image.fromarray(edited_image)).detach().cpu().numpy().item())
            results_dict["global_iqa"][metric_name] = round(score,4)
        
        # Regional Consistency Metrics
        for metric_name in ["lpips", "psnr"]:
            score = float(metrics[metric_name](Image.fromarray(edited_image_gen), Image.fromarray(source_image_gud_shifted)).detach().cpu().numpy().item())
            results_dict["object_consistency"][metric_name] = round(score,4)
            
            score = float(metrics[metric_name](Image.fromarray(edited_image_share), Image.fromarray(source_image_share)).detach().cpu().numpy().item())
            results_dict["background_consistency"][metric_name] = round(score,4)
            
            score = float(metrics[metric_name](Image.fromarray(edited_image_ipt), Image.fromarray(source_image_ipt)).detach().cpu().numpy().item())
            results_dict["inpainting_similarity"][metric_name] = round(score,4)
                
        # Semantic Consistency Metrics
        # CLIP-t2t, CLIP-i2i
        clip_model = metrics["clip_model"]
        clip_preprocess = metrics["clip_preprocess"]
        source_caption_emb = clip_model.encode_text(clip.tokenize([source_caption]).cuda())
        edited_caption_emb = clip_model.encode_text(clip.tokenize([edited_caption]).cuda())
        source_image_emb = clip_model.encode_image(clip_preprocess(Image.fromarray(source_image)).unsqueeze(0).to(device)).cuda()
        edited_image_emb = clip_model.encode_image(clip_preprocess(Image.fromarray(edited_image)).unsqueeze(0).to(device)).cuda()
        source_caption_emb = source_caption_emb / source_caption_emb.norm(dim=-1, keepdim=True)
        edited_caption_emb = edited_caption_emb / edited_caption_emb.norm(dim=-1, keepdim=True)
        source_image_emb = source_image_emb / source_image_emb.norm(dim=-1, keepdim=True)
        edited_image_emb = edited_image_emb / edited_image_emb.norm(dim=-1, keepdim=True)
        
        # CLIP-t2t (edited caption, source caption)
        score = float((source_caption_emb @ edited_caption_emb.T).item())
        results_dict["semantic"]["clip_t2t"] = round(score,4)
        
        # CLIP-i2i (edited image, source image)
        score = float((source_image_emb @ edited_image_emb.T).item())
        results_dict["semantic"]["clip_i2i"] = round(score,4)
        
    return results_dict


def aggregate_results(results_dict):
    aggregated_results_dict = {}
    aggregated_results_dict["aggregated"] = {}
    
    # create empty list for each category-metric
    first_result = results_dict[list(results_dict.keys())[0]]
    for category in first_result:
        if isinstance(first_result[category], dict):
            aggregated_results_dict[category] = {}
            for metric, value in first_result[category].items():
                aggregated_results_dict[category][metric] = []

    for name, result in results_dict.items():
        for category in result:
            if isinstance(result[category], dict):
                for metric, value in result[category].items():
                    aggregated_results_dict[category][metric].append(value)
    
    for category in aggregated_results_dict:
        for metric, value_list in aggregated_results_dict[category].items():
            aggregated_results_dict["aggregated"]["%s-%s-mean"%(category, metric)] = round(np.average(value_list), 4)
            aggregated_results_dict["aggregated"]["%s-%s-std"%(category, metric)] = round(np.std(value_list),4)
            
    return aggregated_results_dict


def parse_args():
    '''
    Evaluation output are saved under "outputs/metrics/<dataset>/sd1p5_<steps>/" folder, named as "aggregated_<method_name>.json" and "individual_<method_name>.json"
    '''
    parser = argparse.ArgumentParser(description="Evaluate Metrics for results on a dataset & method")
    
    parser.add_argument(
        "--which_dataset", type=str, default="COCOEE", help="which dataset", choices=["ReS", "COCOEE"]
    )
    parser.add_argument(
        "--steps", type=str, default="16", help="which step was used", choices=["4", "8", "16", "50"]
    )
    parser.add_argument(
        "--input_path", type=str, default="outputs/COCOEE/null/sd1p5_16/ours_gsn-1_invfree-1_sam-1_cp-1/4-6-0.2-0.8/", help="path to the edited images of one experiment"
    )
    parser.add_argument(
        "--method_name", type=str, default="ours_gsn-1_invfree-1_sam-1_cp-1_4-6-0.2-0.8", help="method name, this decide what name the .json output will be"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    dataset_paths = {
        "ReS" : "datasets/ReS/",
        "COCOEE" : "datasets/COCOEE/",
    }
    dataset_path = dataset_paths[args.which_dataset]
    
    # Prepare dataset
    if args.which_dataset == "ReS":
        with open(os.path.join(dataset_path,"ReS_dataset.json")) as f:
            ReS_dict = json.load(f)
        images_list = sorted(list(ReS_dict.keys()))
    elif args.which_dataset == "COCOEE":
        with open(os.path.join(dataset_path, "COCOEE_dataset.json")) as f:
            COCOEE_dict = json.load(f)
        images_list = sorted(list(COCOEE_dict.keys()))
    else:
        raise ValueError("Dataset choice not exist: %s"%(args.which_dataset))
    
    # Load Evaluation Metrics/Models
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    blip_model, blip_vis_processors = load_blip_model(device)
    metrics = load_metrics(device)
    
    result_per_image_dict = {}
    # Loop over each edited image
    for idx, which_img in tqdm(enumerate(images_list), total=len(images_list)):
        print(f"Evaluating the '{which_img}' image ...")
        
        edited_image_filename = os.path.join(args.input_path, "edited_%s.png"%(which_img))
        edited_image = np.array(Image.open(edited_image_filename))
        
        if args.which_dataset == "ReS":
            source_image, mask, max_resolution, prompt, selected_points, words = get_data_ReS(which_img, True, ReS_dict)
        elif args.which_dataset == "COCOEE":
            source_image, mask, max_resolution, prompt, selected_points, words = get_data_COCOEE(which_img, True, COCOEE_dict)
        else:
            raise ValueError("Dataset choice not exist: %s"%(args.which_dataset))
        
        results_dict = evaluate_metrics(source_image=source_image, edited_image=edited_image,
                                        source_mask=mask, selected_points=selected_points,
                                        metrics=metrics, blip_model=blip_model, blip_vis_processors=blip_vis_processors, device=device)
        
        result_per_image_dict[which_img] = results_dict

    # Aggregate Results
    result_per_image_dict = dict(sorted(result_per_image_dict.items()))
    result_aggregated_dict = aggregate_results(result_per_image_dict)
    
    # Save to Files
    save_path = os.path.join("outputs", "metrics", args.which_dataset, "sd1p5_%s"%(args.steps))
    os.makedirs(save_path, exist_ok=True)
    
    with open(os.path.join(save_path, "individual_%s.json"%(args.method_name)), 'w') as f:
        json.dump(result_per_image_dict, f, sort_keys=False, indent=4)
    try:
        with open(os.path.join(args.input_path, "individual_%s.json"%(args.method_name)), 'w') as f:
            json.dump(result_per_image_dict, f, sort_keys=False, indent=4)
    except:
        pass
            
    with open(os.path.join(save_path, "aggregated_%s.json"%(args.method_name)), 'w') as f:
        json.dump(result_aggregated_dict, f, sort_keys=False, indent=4)
    try:
        with open(os.path.join(args.input_path, "aggregated_%s.json"%(args.method_name)), 'w') as f:
            json.dump(result_aggregated_dict, f, sort_keys=False, indent=4)
    except:
        pass
    