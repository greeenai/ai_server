"""
# image_generate_app.py

사용 예시 cli 명령어:
uvicorn image_generate_app:app --host 0.0.0.0 --port 8000 --reload

Diary Entry → 프롬프트 → SDXL 이미지 생성 서비스

다이어리 엔트리(제목, 캡션, 질문, 답변)를 받아 GPT-4o-mini로 프롬프트를 생성하고,
이를 기반으로 Stable Diffusion XL 모델을 통해 이미지를 생성하는 FastAPI 서버
"""

import io
import time
import asyncio
import os
from pathlib import Path
from typing import List

import torch
import openai
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from fastapi.concurrency import run_in_threadpool

# ================================================================================
# 환경 설정
# ================================================================================

# CUDA 메모리 분할 설정으로 Out of Memory 오류 방지
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
torch.set_printoptions(precision=10)  # 디버깅 시 정밀도 표시 설정

# 상수 정의
STYLE_PREFIX = "cloud-bread style, "  # 모든 프롬프트에 적용할 스타일 프리픽스
NEGATIVE_PROMPT = "low quality, blurry, distorted"  # 부정적 프롬프트
OUT_DIR = Path("fast_api_output")
OUT_DIR.mkdir(exist_ok=True)  # 출력 디렉토리 생성

# ================================================================================
# 모델 초기화
# ================================================================================

def initialize_pipeline() -> StableDiffusionXLPipeline:
    """
    Stable Diffusion XL 파이프라인을 초기화하고 최적화 설정을 적용
    
    Returns:
        StableDiffusionXLPipeline: 초기화된 파이프라인 객체
    """
    # 안정성을 위해 명시적으로 float32 사용
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True
    ).to("cuda")
    
    # Cloud-bread 스타일 LoRA 가중치 적용
    try:
        pipe.load_lora_weights("pytorch_lora_weights.safetensors")
        pipe.fuse_lora()  # LoRA 가중치를 모델에 결합
    except Exception as e:
        print(f"LoRA 가중치 로딩 오류: {e}")
    
    # 메모리 최적화 설정
    pipe.enable_vae_tiling()  # VAE 타일링으로 대형 이미지 처리 최적화
    pipe.enable_attention_slicing()  # 어텐션 메모리 사용량 감소
    
    return pipe

# 서버 시작 시 파이프라인 한 번만 로드
try:
    PIPE = initialize_pipeline()
    print("✅ Stable Diffusion XL 파이프라인 초기화 성공")
except Exception as e:
    print(f"❌ 파이프라인 초기화 실패: {e}")
    raise

# ================================================================================
# 데이터 모델
# ================================================================================

class Entry(BaseModel):
    """다이어리 엔트리 데이터 모델"""
    title: str
    caption: str
    question: str
    answer: str

class GenerateReq(BaseModel):
    """이미지 생성 요청 데이터 모델"""
    entries: List[Entry] = Field(..., min_items=1, description="다이어리 엔트리 목록")

# ================================================================================
# 프롬프트 생성
# ================================================================================

# GPT-4o-mini 시스템 프롬프트
PROMPT_SYSTEM = (
    "You are a prompt generator for Stable Diffusion. "
    "Given several diary entries (Korean caption + user emotional answer), "
    "produce ONE comma-separated English keyword prompt (~30 words) that blends the visual elements and mood. "
    "Return ONLY the prompt, no extra quotes or commentary."
)

# OpenAI 클라이언트 초기화
aopenai = openai.AsyncOpenAI()

async def build_prompt(entries: List[Entry]) -> str:
    """
    다이어리 엔트리 목록을 기반으로 GPT-4o-mini를 사용해 이미지 생성 프롬프트 생성
    
    Args:
        entries: 다이어리 엔트리 목록
        
    Returns:
        str: 생성된 Stable Diffusion 프롬프트
    """
    # 엔트리별 캡션과 감정 답변을 구조화
    lines = [
        f"{i}. Caption: {e.caption}\n   Emotion: {e.answer.strip()}"
        for i, e in enumerate(entries, 1)
    ]
    
    # 사용자 메시지 구성
    user_msg = "Entries:\n" + "\n".join(lines) + "\nPrompt:"
    
    # GPT-4o-mini로 프롬프트 생성
    chat = await aopenai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=120,
        temperature=0.9,  # 창의성 조절
    )
    
    return chat.choices[0].message.content.strip()

# ================================================================================
# 이미지 생성
# ================================================================================

def post_process_image(img_array: np.ndarray) -> np.ndarray:
    """
    이미지 배열의 이상값(NaN) 및 범위를 수정하여 안정적인 이미지 생성
    
    Args:
        img_array: 원본 이미지 배열
        
    Returns:
        np.ndarray: 후처리된 이미지 배열
    """
    # NaN 값 확인 및 처리
    if np.isnan(img_array).any():
        print("⚠️ 경고: 이미지에 NaN 값 발견, 보정 중...")
        img_array = np.nan_to_num(img_array)
    
    # 범위 확인 및 조정 (0-255 범위로)
    img_array = np.clip(img_array, 0, 255)
    
    return img_array.astype('uint8')

def generate_image(
    prompt: str,
    steps: int = 35, 
    scale: float = 7.0,
    w: int = 768, 
    h: int = 768
) -> Image.Image:
    """
    Stable Diffusion XL을 사용하여 주어진 프롬프트로 이미지 생성
    
    Args:
        prompt: 이미지 생성에 사용할 프롬프트
        steps: 추론 단계 수
        scale: 가이던스 스케일
        w: 이미지 너비
        h: 이미지 높이
        
    Returns:
        Image.Image: 생성된 PIL 이미지
    """
    # 스타일 프리픽스 추가
    full_prompt = STYLE_PREFIX + prompt
    print(f"📝 프롬프트: {full_prompt}")
    
    # 그래디언트 계산 불필요 (추론만 수행)
    with torch.no_grad():
        result = PIPE(
            prompt=full_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            guidance_scale=scale,
            num_inference_steps=steps,
            width=w,
            height=h,
        )
    
    img = result.images[0]
    
    # 이미지 후처리
    img_array = np.array(img)
    img_array = post_process_image(img_array)
    img = Image.fromarray(img_array)
    
    # 디버깅용 이미지 저장
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fname = OUT_DIR / f"debug_{timestamp}.png"
    img.save(fname)
    print(f"🖼️ 디버그 이미지 저장: {fname}")
    
    return img

# ================================================================================
# FastAPI 서버
# ================================================================================

app = FastAPI(
    title="Diary → Prompt → SDXL Image",
    description="다이어리 엔트리를 기반으로 Stable Diffusion XL 이미지 생성 API",
    version="1.0.0"
)

@app.post("/generate-image", response_class=StreamingResponse)
async def generate(req: GenerateReq):
    """
    다이어리 엔트리를 받아 이미지 생성
    
    1. GPT로 Stable Diffusion 프롬프트 생성
    2. Stable Diffusion XL로 이미지 생성
    3. PNG 이미지로 응답
    
    Args:
        req: 다이어리 엔트리 목록이 포함된 요청 객체
        
    Returns:
        StreamingResponse: 생성된 이미지 파일
    """
    # 1) GPT로 SD 프롬프트 생성
    try:
        sd_prompt = await build_prompt(req.entries)
    except Exception as e:
        print(f"❌ 프롬프트 생성 실패: {e}")
        raise HTTPException(500, f"Prompt generation failed: {e}")

    print(f"🔄 [SD-Prompt] {sd_prompt}")

    # 2) Stable Diffusion XL 이미지 생성 (블로킹 작업 → 스레드풀 실행)
    try:
        img = await run_in_threadpool(generate_image, sd_prompt)
    except Exception as e:
        print(f"❌ 이미지 생성 실패: {e}")
        raise HTTPException(500, f"Image generation failed: {e}")

    # 3) PNG 이미지 형식으로 응답
    buf = io.BytesIO()
    try:
        img.save(buf, format="PNG")
        buf.seek(0)  # 버퍼 포인터를 시작으로 이동
    except Exception as e:
        print(f"❌ 이미지 저장 실패: {e}")
        raise HTTPException(500, f"Failed to save image: {e}")
    
    # 파일명 및 응답 헤더 설정
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"diary_{timestamp}.png"
    
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="{filename}"'}
    )

# 서버 실행 (uvicorn main:app --reload)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)