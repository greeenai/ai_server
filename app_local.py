import io, time, asyncio, os
from pathlib import Path
from typing import List

import torch, openai
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from fastapi.concurrency import run_in_threadpool

# -------------------- 환경 설정 및 디버깅 --------------------
# 디버깅 로그 활성화
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
torch.set_printoptions(precision=10)

# -------------------- 준비 --------------------
STYLE_PREFIX = "cloud-bread style, "
# STYLE_PREFIX = " "
OUT_DIR = Path("fast_api_output"); OUT_DIR.mkdir(exist_ok=True)

NEGATIVE_PROMPT = "low quality, blurry, distorted"

# ── 파이프라인은 서버 시작 시 1회 로드 ──
def _build_pipe() -> StableDiffusionXLPipeline:
    # 명시적으로 float32 사용 (fp16 대신)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,  # 명시적으로 float32 사용
        use_safetensors=True
    ).to("cuda")
    
    # LoRA 가중치 로드
    try:
        pipe.load_lora_weights("pytorch_lora_weights.safetensors")
        pipe.fuse_lora()
    except Exception as e:
        print(f"LoRA 로딩 오류: {e}")
    
    pipe.enable_vae_tiling()
    pipe.enable_attention_slicing()
    return pipe

# 파이프라인 초기화 (fp32로 일관되게 사용)
try:
    PIPE = _build_pipe()
    print("파이프라인 초기화 성공")
except Exception as e:
    print(f"파이프라인 초기화 실패: {e}")
    raise

# -------------------- 데이터 모델 --------------------
class Entry(BaseModel):
    title: str
    caption: str
    question: str
    answer: str

class GenerateReq(BaseModel):
    entries: List[Entry] = Field(..., min_items=1)

# -------------------- GPT-4o-mini 프롬프트 빌드 --------------------
PROMPT_SYSTEM = (
    """
You are an artist who must reinterpret **exactly 3 diary photos** into one diary-style painting.

Input  
• 3 diary entries, each containing a Korean caption and the user’s stated emotion.

Workflow  
0. From **each** entry, extract **exactly ONE** vivid keyword that captures its main visual subject (→ **keyword1, keyword2, keyword3**). These three concrete words are mandatory.  
1. From those keywords and emotions, identify the single core theme shared by all entries.  
2. **Choose ONE clearly identifiable main object that embodies that theme and place it front-and-center.  
   • The main object must occupy at least 40% of the canvas area.  
   • Render it in sharp focus with higher contrast than the background.**  
3. Select ONE background setting that harmonizes with the main object and reinforces the same theme while remaining visually subordinate (soft focus, lighter contrast).  
4. Distill the combined emotions into 1–2 concise English mood adjectives.  
5. In the POSITIVE prompt, list *all three keywords*, each appearing exactly once (no omissions, no duplicates).  
6. Wrap the English description of the main object with double parentheses: `(( … ))` to boost weight.  

Return **exactly two lines, comma-separated words only**:

POSITIVE: ((<main object>)), <keyword1>, <keyword2>, <keyword3>, conveying <mood adjectives>, hand-drawn diary style illustration, cinematic, ultra-detailed, 8k, masterpiece  
NEGATIVE: low quality, poorly drawn, distorted anatomy, extra limbs, watermark, text, logo, multiple unrelated objects, blurry, cropped

"""
    # "You are a prompt generator for Stable Diffusion. "
    # "Given several diary entries (Korean caption + user emotional answer), "
    # "produce ONE comma-separated English keyword prompt (~30 words) that blends the visual elements and mood. "
    # "Return ONLY the prompt, no extra quotes or commentary."
)
aopenai = openai.AsyncOpenAI()

async def build_prompt(entries: List[Entry]) -> str:
    lines = [
        f"{i}. Caption: {e.caption}\n   Emotion: {e.answer.strip()}"
        for i, e in enumerate(entries, 1)
    ]
    user_msg = "Entries:\n" + "\n".join(lines) + "\nPrompt:"
    print("user_msg",user_msg)
    chat = await aopenai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=120,
        temperature=0.9,
    )
    positive, negative = [
        part.split(":", 1)[1].strip()
        for part in chat.choices[0].message.content.splitlines()[:2]
    ]
    return positive, negative

# -------------------- SDXL 이미지 생성 --------------------
def post_process_image(img_array):
    """NaN과 이상 값을 처리하는 함수"""
    # NaN 값 확인 및 처리
    if np.isnan(img_array).any():
        print("경고: 이미지에 NaN 값 발견")
        img_array = np.nan_to_num(img_array)
    
    # 범위 확인 및 조정 (0-255 범위로)
    img_array = np.clip(img_array, 0, 255)
    
    return img_array.astype('uint8')

def generate_image(prompt: str, negative: str,
                   steps: int = 35, scale: float = 7.0,
                   w: int = 768, h: int = 768) -> Image.Image:
    full_prompt = STYLE_PREFIX + prompt
    print(f"프롬프트: {full_prompt}")
    
    # torch.no_grad() 사용하여 fp32 정밀도 유지
    with torch.no_grad():
        result = PIPE(
            prompt=full_prompt,
            negative_prompt=negative,
            guidance_scale=scale,
            num_inference_steps=steps,
            width=w,
            height=h,
        )
    
    img = result.images[0]
    
    # PIL 이미지를 NumPy 배열로 변환하여 후처리
    img_array = np.array(img)
    img_array = post_process_image(img_array)
    
    # 후처리된 배열을 PIL 이미지로 변환
    img = Image.fromarray(img_array)
    
    # 디버깅: 이미지 로컬 저장
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = OUT_DIR / f"debug_{ts}.png"
    img.save(fname)
    print(f"디버그 이미지 저장됨: {fname}")
    
    return img

# -------------------- FastAPI --------------------
app = FastAPI(title="Diary → Prompt → SDXL image")

@app.post("/generate-image", response_class=StreamingResponse)
async def generate(req: GenerateReq):
    # 1) GPT로 SD 프롬프트 생성
    try:
        positive, negative = await build_prompt(req.entries)
    except Exception as e:
        print(f"프롬프트 생성 실패: {e}")
        raise HTTPException(500, f"Prompt generation failed: {e}")

    print("[SD-positive-Prompt]", positive,"[SD-negative-Prompt]", negative)      # 서버 로그

    # 2) Stable Diffusion XL 이미지 생성 (블로킹 작업 → 스레드풀)
    try:
        img: Image.Image = await run_in_threadpool(generate_image, positive, negative)
    except Exception as e:
        print(f"이미지 생성 실패: {e}")
        raise HTTPException(500, f"Image generation failed: {e}")

    # 3) PNG 응답
    buf = io.BytesIO()
    try:
        img.save(buf, format="PNG")
        buf.seek(0)
    except Exception as e:
        print(f"이미지 저장 실패: {e}")
        raise HTTPException(500, f"Failed to save image: {e}")
    
    fname = f"diary_{time.strftime('%Y%m%d_%H%M%S')}.png"
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="{fname}"'}
    )