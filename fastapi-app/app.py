# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import os, json, requests, concurrent.futures, logging, re

app = FastAPI()
log = logging.getLogger("uvicorn.error")

# ─── OpenAI 설정 ──────────────────────────────────────────────────────
OPENAI_KEY = "example"
HEADERS = {"Authorization": f"Bearer {OPENAI_KEY}",
           "Content-Type": "application/json"}
MODEL = "gpt-4o-mini"

# GPT 응답이 ```json ...``` 코드블록일 때 제거하는 정규식
CODEBLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)

PROMPT = """
이 사진은 오늘 사용자가 찍은 일상사진 중에 하나야. 
                                이 사용자의 일기를 작성할 수 있도록 이때의 스토리를 끌어낼 만한 질문을 생성해주는데 의도는 감정을 끌어내는 것이야., 
                                해당 사진의 주요 디테일과 컨텍스트를 담은 캡션도 함께 제공되어. 
                                이 캡션을 참고하여, 각 사진당 질문은 1개만 생성해주고, 예시답변도 객관식 4지선다로 제공해주고 이모지도 포함해줘. 
                                그러므로 사진이 3개면 질문과 예시답변세트도 3개씩 생성되어야해. 
                                출력은 반드시 JSON 형식만 포함되어야 하며, 추가적인 설명이나 텍스트는 없어야해. 아래 예시를 참고해줘. 
                                 {
                                        "title": "전통 건물", 
                                        "caption": "낡았지만 아름다운 전통 건물의 디테일과 주변 풍경이 돋보이는 사진",
                                        "question": "이 전통 건물을 보며 어떤 감정을 느꼈나요?",
                                        "options": [
                                        "역사적 의미를 느껴서 감동했다.😚",
                                        "그냥 스쳐 지나갔다.😏",
                                        "아름다워서 사진을 찍고 싶었다.🤩",
                                        "외로움을 느꼈다.🥲"
                                        ]
                                    },
                    
출력은 반드시 JSON 객체만 포함해야 하며, 추가 텍스트는 절대 넣지 마.
"""

# ─── GPT 호출 함수 ────────────────────────────────────────────────────
def call_gpt_with_image(url: str) -> Dict:
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": url}}
            ],
        }],
        "max_tokens": 800,
        # 모델이 순수 JSON만 보내도록 강제
        "response_format": {"type": "json_object"}
    }

    r = requests.post("https://api.openai.com/v1/chat/completions",
                      headers=HEADERS, json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI {r.status_code}: {r.text}")

    raw = r.json()["choices"][0]["message"]["content"].strip()

    # 코드블록 제거(모델이 규격 안 지켰을 때 대비)
    m = CODEBLOCK_RE.search(raw)
    if m:
        raw = m.group(1)
    data = json.loads(raw)

    # 'question' → 'prompt' 로 이름 교정
    if "question" in data:
        data["prompt"] = data.pop("question")
    return data

# ─── Pydantic 모델 ───────────────────────────────────────────────────
class ImagesRequest(BaseModel):
    image_urls: List[str]

# ─── 엔드포인트 ──────────────────────────────────────────────────────
@app.post("/generate-question")
def generate_questions(req: ImagesRequest):
    max_workers = min(5, len(req.image_urls))   # 동시 호출 제한
    questions: List[Dict] = [None] * len(req.image_urls)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {
            ex.submit(call_gpt_with_image, url): idx
            for idx, url in enumerate(req.image_urls)
        }
        for fut in concurrent.futures.as_completed(future_map):
            idx = future_map[fut]
            try:
                questions[idx] = fut.result()
            except Exception as e:
                log.error("Error on %s: %s", req.image_urls[idx], e)
                raise HTTPException(502, f"{req.image_urls[idx]} → {e}")

    return {"questions": questions}

@app.get("/health")
def health():
    return {"status": "ok"}
