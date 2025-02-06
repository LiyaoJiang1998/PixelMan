from src.demo.download import download_all
download_all()

from src.demo.demo import create_demo_move
from src.demo.model import EditModels

import cv2
import gradio as gr

# main demo
pretrained_model_path = "runwayml/stable-diffusion-v1-5"
model = EditModels(pretrained_model_path=pretrained_model_path)

DESCRIPTION = '# PixelMan'

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.TabItem('PixelMan'):
            create_demo_move(model.run_move)

demo.queue(concurrency_count=3, max_size=20)
demo.launch(server_name="0.0.0.0")
