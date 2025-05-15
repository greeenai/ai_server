# app.py
import io, os, requests, boto3
from pathlib import Path
from urllib.parse import urlparse
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, validator

from PIL import Image, ImageOps
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline

# ----------------- 설정 -----------------
MODEL_ID  = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_PATH = "pytorch_lora_weights.safetensors"
OUTPUT_BUCKET = "green-user-images-bucket"
OUTPUT_PREFIX = "sdxl_results"

s3 = boto3.client("s3")  # IAM Role or credentials must allow GetObject & PutObject

# ----------------- Pydantic 스키마 -----------------
class GenerateRequest(BaseModel):
    image_urls: List[str]
    captions:   List[str]

    @validator("captions")
    def same_length(cls, v, values):
        if "image_urls" in values and len(v) != len(values["image_urls"]):
            raise ValueError("image_urls and captions must have the same length")
        return v

class GenerateResponse(BaseModel):
    generated_urls: List[str]

# ----------------- 유틸 -----------------
def parse_s3(uri: str):
    p = urlparse(uri)
    if p.scheme != "s3":
        raise ValueError("not s3://")
    return p.netloc, p.path.lstrip("/")

def download_image(uri: str, max_side=1024):
    if uri.startswith("s3://"):
        bucket, key = parse_s3(uri)
        buf = io.BytesIO()
        s3.download_fileobj(bucket, key, buf)
        buf.seek(0)
    else:
        r = requests.get(uri, timeout=20)
        r.raise_for_status()
        buf = io.BytesIO(r.content)

    img = Image.open(buf)
    img = ImageOps.exif_transpose(img).convert("RGB")
    # 비율 유지하여 긴 변을 max_side 로
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img

def upload_png(img: Image.Image, bucket: str, key: str):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    s3.upload_fileobj(buf, bucket, key, ExtraArgs={"ContentType": "image/png"})
    return f"s3://{bucket}/{key}"

# ----------------- 모델 -----------------
def load_pipeline():
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to("cuda")
    pipe.load_lora_weights(LORA_PATH, adapter_name="cloudbread")
    pipe.set_adapters(["cloudbread"], adapter_weights=[0.8])
    pipe.fuse_lora()
    pipe.enable_xformers_memory_efficient_attention()
    return pipe

pipe = load_pipeline()  # ★ 서버 시작 시 한 번만 로드

# ----------------- FastAPI -----------------
app = FastAPI(title="SDXL LoRA img2img API")

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    generated = []
    for idx, (url, caption) in enumerate(zip(req.image_urls, req.captions), 1):
        try:
            src = download_image(url)
        except Exception as e:
            raise HTTPException(400, f"Failed to download image #{idx}: {e}")

        with torch.cuda.amp.autocast():
            result = pipe(
                prompt=caption,
                image=src,
                denoising_strength=0.55,
                guidance_scale=7.0,
                num_inference_steps=35,
            ).images[0]

        base = os.path.splitext(os.path.basename(urlparse(url).path))[0]
        key_out = f"{OUTPUT_PREFIX}/{base}_sdxl.png"
        gen_uri = upload_png(result, OUTPUT_BUCKET, key_out)
        generated.append(gen_uri)

    return GenerateResponse(generated_urls=generated)
