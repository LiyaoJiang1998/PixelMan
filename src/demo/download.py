import os
import subprocess
import shlex

def download_all():
    # download checkpoints
    os.makedirs('models', exist_ok=True)
    if not os.path.exists("models/ip_sd15_64.bin"):
        subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/model/ip_sd15_64.bin -O models/ip_sd15_64.bin'))
    if not os.path.exists("models/efficient_sam_vits.pt"):
        subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/model/efficient_sam_vits.pt -O models/efficient_sam_vits.pt'))
    print("Download Finished.")

if __name__ == "__main__":
    download_all()
    