import os, sys
import argparse
import numpy as np
from PIL import Image
import json
import ast
import time
from datetime import timedelta
from tqdm import tqdm

# Dataset Path
ReS_path = "datasets/ReS/"
COCOEE_path = "datasets/COCOEE/"

model_path_dict = {
    "sd1p5": "runwayml/stable-diffusion-v1-5",
}

def exif_transpose(img):
    if not img:
        return img 
    exif_orientation_tag = 274

    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        if orientation == 1:
            pass 
        elif orientation == 2:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            img = img.rotate(180)
        elif orientation == 4:
            img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            img = img.rotate(90, expand=True)
    return img

def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def parse_args():
    parser = argparse.ArgumentParser(description="Let's Edit!")
    
    parser.add_argument(
        "--steps", type=int, default=16, choices=[4, 8, 16, 50],
    )
    parser.add_argument(
        "--which_dataset", type=str, default="COCOEE", help="which dataset to select images from?", choices=["COCOEE", "ReS"]
    )
    parser.add_argument(
        "--which_img", type=str, default="all", help="which image to edit?"
    )
    parser.add_argument(
        "--edit_algo", type=str, default="ours", help="which algorithm to edit with?", choices=["ours", "dragon", "diffeditor", "sg"]
    )
    parser.add_argument(
        "--coefficients", type=str, default="[4, 6, 0.2, 0.8]"
    )
    parser.add_argument(
        "--use_gsn", type=int, default=1
    )
    parser.add_argument(
        "--inversion_free", type=int, default=1,
    )
    parser.add_argument(
        "--sa_masking_ipt", type=int, default=1
    )
    parser.add_argument(
        "--use_copy_paste", type=int, default=1
    )
    parser.add_argument(
        "--use_prompt", action='store_true'
    )
    
    args = parser.parse_args()
    return args

def get_data_ReS(which_img, use_prompt, ReS_dict):
    data = ReS_dict[which_img]
    
    max_resolution=data["max_resolution"]
    prompt=data["prompt"]
    words=data["words"]
    selected_points=data["selected_points"]

    # load image and mask
    original_image = np.array(exif_transpose(Image.open(os.path.join(ReS_path, data["original_image_path"]))).convert('RGB'))
    mask = np.array(exif_transpose(Image.open(os.path.join(ReS_path, data["mask_path"]))).convert('L'))
    # Resize image and mask
    h, w = Image.fromarray(original_image).size
    factor = max_resolution / (min(h, w))
    if factor != 1:
        h, w = int(h * factor), int(w * factor)
        original_image = np.array(Image.fromarray(original_image).resize((h, w), Image.BICUBIC))
        mask = np.array(Image.fromarray(mask).resize((h, w), Image.NEAREST))
    mask = np.expand_dims(mask, axis=2)

    if use_prompt:
        return original_image, mask, max_resolution, prompt, selected_points, words
    else:
        return original_image, mask, max_resolution, "", selected_points, []

def get_data_COCOEE(which_img, use_prompt, COCOEE_dict):
    data = COCOEE_dict[which_img]
    
    max_resolution=data["max_resolution"]
    prompt=data["prompt"]
    words=data["words"]
    selected_points=data["selected_points"]
    
    image_path = os.path.join(COCOEE_path, "COCOEE_images", data["original_image_path"])
    mask_path = os.path.join(COCOEE_path, "COCOEE_masks",  data["mask_path"])
    
    # load image and mask
    original_image = np.array(exif_transpose(Image.open(image_path)).convert('RGB'))
    mask = np.array(exif_transpose(Image.open(mask_path)).convert('L'))
    # Resize image and mask
    h, w = Image.fromarray(original_image).size
    factor = max_resolution / (min(h, w))
    if factor != 1:
        h, w = int(h * factor), int(w * factor)
        original_image = np.array(Image.fromarray(original_image).resize((h, w), Image.BICUBIC))
        mask = np.array(Image.fromarray(mask).resize((h, w), Image.NEAREST))
    mask = np.expand_dims(mask, axis=2)
        
    if use_prompt:
        return original_image, mask, max_resolution, prompt, selected_points, words
    else:
        return original_image, mask, max_resolution, "", selected_points, []

def main():
    args = parse_args()
    
    pretrained_model_path = "runwayml/stable-diffusion-v1-5"
    model_mode = "sd1p5" + "_" + str(args.steps)

    if args.edit_algo == "ours":
        from src.demo.model import EditModels
        model = EditModels(pretrained_model_path=pretrained_model_path, steps=args.steps, use_ip_adapter=False)
        edit_algo = "ours_"+"gsn-"+str(args.use_gsn)+"_"+"invfree-"+str(args.inversion_free)+"_"+"sam-"+str(args.sa_masking_ipt)+"_"+"cp-"+str(args.use_copy_paste)      
        specific = ast.literal_eval(args.coefficients)
    elif args.edit_algo == "dragon":
        from src.demo.model_dragon import DragonModels
        model = DragonModels(pretrained_model_path=pretrained_model_path, steps=args.steps)
        edit_algo = args.edit_algo
        specific = ast.literal_eval(args.coefficients)
    elif args.edit_algo == "diffeditor":
        from src.demo.model_dragon import DragonModels
        model = DragonModels(pretrained_model_path=pretrained_model_path, steps=args.steps)
        edit_algo = args.edit_algo
        specific = ast.literal_eval(args.coefficients)
    elif args.edit_algo == "sg":
        from src.demo.model_sg import DragonModels
        model = DragonModels(pretrained_model_path=pretrained_model_path, steps=args.steps)
        edit_algo = args.edit_algo
        specific = ast.literal_eval(args.coefficients)
    else:
        raise ValueError("Algorithm choice not exist: %s"%(args.edit_algo))
    print(specific)
    
    if args.which_dataset == "ReS":
        with open(os.path.join(ReS_path,"ReS_dataset.json")) as f:
            ReS_dict = json.load(f)
        if args.which_img == "all":
            images_list = sorted(list(ReS_dict.keys()))
        else:
            images_list = [args.which_img]
    elif args.which_dataset == "COCOEE":
        with open(os.path.join(COCOEE_path, "COCOEE_dataset.json")) as f:
            COCOEE_dict = json.load(f)
        if args.which_img == "all":
            images_list = sorted(list(COCOEE_dict.keys()))
        else:
            images_list = [args.which_img]    
    else:
        raise ValueError("Dataset choice not exist: %s"%(args.which_dataset))
    
    latency_list = []
    
    for which_img in tqdm(images_list):
        
        if args.which_dataset == "ReS":
            original_image, mask, max_resolution, prompt, selected_points, words = get_data_ReS(which_img, args.use_prompt, ReS_dict)
        elif args.which_dataset == "COCOEE":
            original_image, mask, max_resolution, prompt, selected_points, words = get_data_COCOEE(which_img, args.use_prompt, COCOEE_dict)
        else:
            raise ValueError("Dataset choice not exist: %s"%(args.which_dataset))

        print(f"Editing the '{which_img}' image ...")
        
        w_edit, w_content, w_contrast, w_inpaint = specific

        dataset_str = args.which_dataset
        pr_str = 'prompt' if args.use_prompt else 'null'
        out_dir = 'outputs/' + dataset_str + "/" + pr_str + "/" + model_mode + '/' + edit_algo + '/' \
                + str(w_edit) + '-' + str(w_content) + '-' + str(w_contrast) + '-' + str(w_inpaint) + '/'
        os.makedirs(out_dir, exist_ok=True)
        
        image_name = out_dir+"edited_"+which_img+'.png'
        
        start_time = time.perf_counter()
        if args.edit_algo == "ours":
            edited_image = model.run_move(original_image, 
                            mask, 
                            mask_ref=None, 
                            prompt=prompt, 
                            resize_scale=1, 
                            w_edit=w_edit, 
                            w_content=w_content, 
                            w_contrast=w_contrast, 
                            w_inpaint=w_inpaint, 
                            seed=42, 
                            selected_points=selected_points, 
                            guidance_scale=4, 
                            energy_scale=0.5, 
                            max_resolution=max_resolution, 
                            SDE_strength=0.4, 
                            ip_scale=0.1,
                            use_gsn=bool(args.use_gsn), 
                            inversion_free=bool(args.inversion_free), 
                            sa_masking_ipt=bool(args.sa_masking_ipt), 
                            use_copy_paste=bool(args.use_copy_paste),
                            )[0]
        
        elif args.edit_algo == "dragon":
            edited_image = model.run_move(original_image, 
                            mask, 
                            mask_ref=None, 
                            prompt=prompt, 
                            resize_scale=1, 
                            w_edit=w_edit, 
                            w_content=w_content, 
                            w_contrast=w_contrast, 
                            w_inpaint=w_inpaint, 
                            seed=42, 
                            selected_points=selected_points, 
                            guidance_scale=4, 
                            energy_scale=0.5, 
                            max_resolution=max_resolution, 
                            SDE_strength=0.4, 
                            ip_scale=0.1,
                            alg="D",
                            )[0]
            
        elif args.edit_algo == "diffeditor":
            edited_image = model.run_move(original_image, 
                            mask, 
                            mask_ref=None, 
                            prompt=prompt, 
                            resize_scale=1, 
                            w_edit=w_edit, 
                            w_content=w_content, 
                            w_contrast=w_contrast, 
                            w_inpaint=w_inpaint, 
                            seed=42, 
                            selected_points=selected_points, 
                            guidance_scale=4, 
                            energy_scale=0.5, 
                            max_resolution=max_resolution, 
                            SDE_strength=0.4, 
                            ip_scale=0.1,
                            alg="D+",
                            )[0]
        
        elif args.edit_algo == "sg":
            if max_resolution > 512:
                max_resolution = 512 # avoid OOM wiht Self-Guidance
            edited_image = model.run_move(original_image, 
                            mask, 
                            mask_ref=None, 
                            prompt=prompt, 
                            resize_scale=1, 
                            w_edit=w_edit, 
                            w_content=w_content, 
                            w_contrast=w_contrast, 
                            w_inpaint=w_inpaint, 
                            seed=42, 
                            selected_points=selected_points, 
                            guidance_scale=4, 
                            energy_scale=0.1, 
                            max_resolution=max_resolution, 
                            SDE_strength=0.4, 
                            ip_scale=0.1,
                            alg="D",
                            words=words,
                            )[0]
                
        else:
            raise ValueError("Algorithm choice not exist: %s"%(args.edit_algo))
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        latency_list.append(elapsed_time)
        print("Latency: %.2fs"%elapsed_time)

        edited_image = Image.fromarray(edited_image.astype(np.uint8))
        edited_image.save(image_name)
        print(f"Image saved in: {image_name}")
    
    # Save inference latency stats
    latency_array = np.array(latency_list)
    latency_mean = latency_array.mean()
    latency_std = latency_array.std()
    latency_min = latency_array.min()
    latency_max = latency_array.max()
    latency_len = len(latency_array)
    latency_sum = latency_array.sum()
    
    latency_results = {
        'latency_mean': latency_mean,
        'latency_std': latency_std,
        'latency_min': latency_min,
        'latency_max': latency_max,
        'latency_len': latency_len,
        'latency_sum': latency_sum,
        'latency_list': latency_list
    }

    with open(os.path.join(out_dir, "a_latency.json"), 'w') as f:
        json.dump(latency_results, f, sort_keys=False, indent=4)
        
    print("Run Total Elapsed Time:", timedelta(seconds=latency_sum))


if __name__ == "__main__":
    main()