"""
local_experiment_sdxl_cloud.py

사용 예시 cli 명령어:
  # 단일 프롬프트
  python local_experiment_sdxl_cloud.py -p "만들고싶은 사진 프롬프트"

  # 여러 장 한 번에
    python3 local_experiment_sdxl_cloud.py -p \
        "hand-drawn daily-journal scene: cozy dog grooming on dark sofa, chilled glass drink with ice and long spoon, take-out kimbap and golden tempura." \
        "hand-drawn daily-journal scene: twin strawberry cream cakes, vibrant sashimi rice bowl with miso and salad, bronze-hat statue before Dongdaegu Station under soft blue sky. Warm café tones and lively seafood hues meet cool urban metals in one illustrated scene." \
        "hand-drawn daily-journal scene: frosty twin beer mugs, vivid grid of red-blue-green blocks, mint-chip ice cream held against night city street. Warm bar glow, playful game hues, and serene evening lights merge in a single high-detail storybook illustration." \
        "hand-drawn daily-journal scene: neon Hong Kong skyline over shimmering harbor, red double-decker tram weaving through busy street, steaming soup and crusty bread on café table. Vibrant night blues and reds blend with warm food tones in dynamic ink-lined illustration." \
        "hand-drawn daily-journal scene: smiling girl under blooming sakura avenue, parked basketed bikes, pink-cup donuts and iced coffee on white table. Pastel tones, soft bokeh, joyful picnic vibe." \
        "hand-drawn daily-journal scene: pink-center tonkatsu with cabbage & wasabi, rich ramen with soft egg, chili, grilled skewers; backdrop of sunlit street, blue sky, tall trees, modern buildings. Cream-brown pastels, soft cinematic light, cozy joyful mood." \
        "hand-drawn daily-journal scene: grand tree on green lawn with black-and-white dressed pair, chestnut drink on rustic table, serene city stream flanked by bike paths, trees, glass towers. Soft pastel inks, warm evening light meets cool sky, calm urban-nature harmony."

옵션:
    --model   모델 ID (기본: runwayml/stable-diffusion-v1-5)
    --fp16    VRAM 절약용 float16 (기본은 안전한 float32)
    --neg     네거티브 프롬프트
    --steps   sampling step 수 (기본 25)
    --scale   guidance scale CFG (기본 7.5)
    --width/--height   출력 해상도 (기본 512×512)
  
필요 GPU 사양:
    15GB이상
"""
import argparse, time
from pathlib import Path
from typing import List

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

OUT_DIR = Path("cloud_output"); OUT_DIR.mkdir(exist_ok=True)
STYLE_PREFIX = " "

def load_pipe(fp16: bool) -> StableDiffusionXLPipeline:
    dtype = torch.float16 if fp16 else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype
    ).to("cuda")
    pipe.load_lora_weights("pytorch_lora_weights.safetensors") # 학습시킨 가중치 필요.
    pipe.fuse_lora()
    pipe.enable_vae_tiling()
    pipe.enable_attention_slicing()
    return pipe

def generate(
    prompts: List[str],
    neg: str,
    steps: int,
    scale: float,
    w: int,
    h: int,
    fp16: bool,
):
    pipe = load_pipe(fp16)
    for i, prompt in enumerate(prompts, 1):
        full_prompt = STYLE_PREFIX + prompt 
        with (torch.cuda.amp.autocast() if fp16 else torch.no_grad()):
            img: Image.Image = pipe(
                prompt=full_prompt,
                negative_prompt=neg,
                guidance_scale=scale,
                num_inference_steps=steps,
                width=w,
                height=h,
            ).images[0]

        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = OUT_DIR / f"t2i_{i:02}_{ts}.png" #timestamp기준 이름 저장 
        img.save(fname)
        print(f"[{i}/{len(prompts)}] ✅  {fname}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stable Diffusion txt2img CLI")
    ap.add_argument("-p", "--prompts", nargs="+", required=True, help="프롬프트 목록")
    ap.add_argument("--neg", default="", help="네거티브 프롬프트")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--scale", type=float, default=7.5)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--fp16", action="store_true", help="float16 사용")
    args = ap.parse_args()

    generate(
        prompts=args.prompts,
        neg=args.neg,
        steps=args.steps,
        scale=args.scale,
        w=args.width,
        h=args.height,
        fp16=args.fp16,
    )
