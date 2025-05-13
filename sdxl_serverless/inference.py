import os
import sys
import subprocess

def install_requirements():
    try:
        import diffusers
        print("diffusers 이미 설치되어 있습니다.")
    except ImportError:
        print("필요한 패키지 설치 중...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                              "torch==2.2.2", "diffusers==0.27.2", 
                              "transformers==4.40.0", "accelerate==0.30.0", 
                              "xformers==0.0.25.post1", "Pillow","huggingface_hub==0.25.2","peft==0.10.0","numpy==1.24.0"])
        print("패키지 설치 완료")
install_requirements()
# 캐시 디렉토리 환경 변수 설정
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/tmp/hf_cache"
os.makedirs("/tmp/hf_cache", exist_ok=True)  # 디렉토리 생성


print("Python path:", sys.path)
print("Installed packages:")
subprocess.call(["pip", "list"])


import json
import base64
import io
import torch
import PIL.Image as Image
from diffusers import StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler
from sagemaker_inference import encoder, content_types


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_PATH = "lora_weights/pytorch_lora_weights.safetensors"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device",device)
def load_model():
    device = torch.device("cuda")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16,  # GPU에서는 float16 사용 가능
        variant="fp16", 
        cache_dir="/tmp/hf_cache"  # 명시적으로 캐시 디렉토리 지정
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(LORA_PATH, adapter_name="cloudbread")
    pipe.set_progress_bar_config(disable=True)
    return pipe

pipe = load_model()

def input_fn(body, ct):
    assert ct == content_types.JSON
    data = json.loads(body)
    img = Image.open(io.BytesIO(base64.b64decode(data["image_base64"]))).convert("RGB").resize((1024,1024))
    return {"img": img, "prompt": data.get("prompt", "cloud-bread style")}

def predict_fn(data, model):
    result = model(
        prompt=data["prompt"],
        image=data["img"],
        denoising_strength=0.55,
        adapter_names=["cloudbread"],
        adapter_weights=[0.8],
        guidance_scale=7.0,
        num_inference_steps=35,
    ).images[0]
    buf = io.BytesIO(); result.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def output_fn(pred, accept):
    return encoder.encode({"image_base64": pred}, accept or content_types.JSON)
