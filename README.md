# <img src="https://liyaojiang1998.github.io/projects/PixelMan/static/images/logo.png" alt="" height="30" margin="auto"> PixelMan: Consistent Object Editing with Diffusion Models via Pixel Manipulation and Generation [AAAI-25]

[📄Paper](https://arxiv.org/abs/2412.14283)
**|** 
[🌐Website](https://liyaojiang1998.github.io/projects/PixelMan)
**|** 
[📰Poster](https://liyaojiang1998.github.io/projects/PixelMan/static/pdfs/AAAI25_PixelMan_poster.pdf)
**|** 
[🪧Slides](https://liyaojiang1998.github.io/projects/PixelMan/static/pdfs/AAAI25_PixelMan_slides.pdf)
**|** 
[📹Video]()

The official implementation for "PixelMan: Consistent Object Editing with Diffusion Models via Pixel Manipulation and Generation", accepted to AAAI-25.

> [PixelMan: Consistent Object Editing with Diffusion Models via Pixel Manipulation and Generation](https://arxiv.org/abs/)\
> Liyao Jiang, Negar Hassanpour, Mohammad Salameh, Mohammadreza Samadi, Jiao He, Fengyu Sun, Di Niu\
> AAAI-25


## Description
The official implementation for "PixelMan: Consistent Object Editing with Diffusion Models via Pixel Manipulation and Generation", accepted to AAAI-25.
- In this work, we propose PixelMan, an inversion-free and training-free method for achieving consistent object editing via Pixel Manipulation and generation. 
- PixelMan maintains image consistency by directly creating a duplicate copy of the source object at target location in the pixel space, and we introduce an efficient sampling approach to iteratively harmonize the manipulated object into the target location and inpaint its original location. 
- The key to ensuring image consistency is anchoring the output image to be generated to the pixel-manipulated image as well as introducing various consistency-preserving optimization techniques during inference. 
- Moreover, we propose a leak-proof SA manipulation technique to enable cohesive inpainting by addressing the attention leakage issue which is a root cause of failed inpainting.


- If you like our project, please give us a ⭐ on Github! 

## 📰 News
- Feb 27, 2025, 12:30pm-2:30pm: Poster presentation at AAAI-25 in Philadelphia
- Feb 6, 2025: The code is now available on [GitHub](https://github.com/LiyaoJiang1998/PixelMan)
- Jan 30, 2025: The paper is now available on [arXiv](https://arxiv.org/abs/2412.14283)
- Dec 09, 2024: PixelMan is accepted to AAAI-25.

## Table of Contents
- [ PixelMan: Consistent Object Editing with Diffusion Models via Pixel Manipulation and Generation \[AAAI-25\]](#-pixelman-consistent-object-editing-with-diffusion-models-via-pixel-manipulation-and-generation-aaai-25)
  - [Description](#description)
  - [📰 News](#-news)
  - [Table of Contents](#table-of-contents)
  - [Dependencies](#dependencies)
  - [Models](#models)
  - [How to Run Demo](#how-to-run-demo)
  - [How to Run Experiments (Comparisons and Ablation)](#how-to-run-experiments-comparisons-and-ablation)
  - [Datasets](#datasets)
    - [COCOEE - from "https://github.com/Fantasy-Studio/Paint-by-Example".](#cocoee---from-httpsgithubcomfantasy-studiopaint-by-example)
    - [ReS - from "https://github.com/Yikai-Wang/ReS".](#res---from-httpsgithubcomyikai-wangres)
  - [Acknowledgements](#acknowledgements)
  - [Bibtex](#bibtex)

## Dependencies
- Python >= 3.9
- [PyTorch >= 2.0.1](https://pytorch.org/)
- [CLIP](https://github.com/openai/CLIP?tab=readme-ov-file#usage)
```bash
pip install -r requirements.txt
```

## Models
- Diffusion Model (SDv1.5) will be automatically downloaded through diffusers
- Download auxiliary models using "src/demo/download.py"
```
python src/demo/download.py
```

## How to Run Demo
We provide a demo on gradio.
```bash
python demo_pixelman.py
```
Then, go to http://0.0.0.0:7860 for the demo on object repositioning

## How to Run Experiments (Comparisons and Ablation)
```
# Edit Images with each method on two datasets
sh scripts/run_experiments.sh

# Evaluate the metrics on edited images
sh scripts/run_metrics.sh

# Note: Evaluation metrics output are saved under "outputs/metrics/<dataset>/sd1p5_<steps>/" folder, named as "aggregated_<method_name>.json" and "individual_<method_name>.json"
```

## Datasets
### COCOEE - from "https://github.com/Fantasy-Studio/Paint-by-Example".
- We include the images, labeled masks, and moving diff vectors under "datasets/COCOEE/"
### ReS - from "https://github.com/Yikai-Wang/ReS".
- We include the images, masks, and moving diff vectors under "datasets/ReS/"

## Acknowledgements
- This codebase is based on the implementation in "https://github.com/MC-E/DragonDiffusion" for [Dragondiffusion](https://arxiv.org/abs/2307.02421) and [DiffEditor](https://arxiv.org/abs/2402.02583).
- The Self-Guidance baseline method implementation follows "https://colab.research.google.com/drive/1SEM1R9mI9cF-aFpqg3NqHP8gN8irHuJi?usp=sharing".

## Bibtex 
If you find our method and paper useful, we kindly ask that you cite our paper:
```
@inproceedings{jiang2025pixelman,
    title = {PixelMan: Consistent Object Editing with Diffusion Models via Pixel Manipulation and Generation},
    author = {Liyao Jiang and Negar Hassanpour and Mohammad Salameh and Mohammadreza Samadi and Jiao He and Fengyu Sun and Di Niu},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    year={2025}
}
```