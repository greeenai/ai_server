import boto3, json, base64, io, sys, time, traceback
import PIL.Image as Image
from botocore.config import Config

ENDPOINT = "sdxl-img2img-g4-final-12"
REGION   = "ap-northeast-2"
IMG_PATH = "hacky.JPG"
MAX_SIZE = 1024  # Limit image size to prevent memory issues

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

try:
    # ── 1) 입력 이미지 → base64 (with resizing) ───────────────────────────────
    log(f"loading {IMG_PATH}")
    img = Image.open(IMG_PATH)
    
    # Resize image if it's too large
    width, height = img.size
    log(f"Original image size: {width}x{height}")
    
    if width > MAX_SIZE or height > MAX_SIZE:
        # Calculate new dimensions while maintaining aspect ratio
        if width > height:
            new_width = MAX_SIZE
            new_height = int((height / width) * MAX_SIZE)
        else:
            new_height = MAX_SIZE
            new_width = int((width / height) * MAX_SIZE)
        
        img = img.resize((new_width, new_height), Image.LANCZOS)
        log(f"Resized image to: {new_width}x{new_height}")
    
    # Convert to RGB if it's not already (e.g., if it's RGBA)
    if img.mode != 'RGB':
        img = img.convert('RGB')
        log(f"Converted image to RGB mode")
    
    # Save to buffer and convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    b64in = base64.b64encode(buffer.read()).decode()

    # ── 2) Prepare payload with more parameters ────────────────────────────────
    payload = {
        "prompt": "cloud-bread style, highly detailed, professional photography",
        "image_base64": b64in,
        "strength": 0.75,  # Adjust how much to change the image (0.0-1.0)
        "num_inference_steps": 30,  # Default is usually 50, reducing for faster processing
        "guidance_scale": 7.5  # How closely to follow the prompt
    }

    log("Payload prepared, image size in base64: " + str(len(b64in) // 1024) + " KB")

    # ── 3) SageMaker Runtime 클라이언트 (15분 타임아웃) ───────
    cfg = Config(read_timeout=900, connect_timeout=300, retries={'max_attempts': 3})
    runtime = boto3.client("sagemaker-runtime", region_name=REGION, config=cfg)
    log("invoke_endpoint() 호출 — 콜드 스타트면 2-4분 소요")

    t0 = time.time()
    resp = runtime.invoke_endpoint(
        EndpointName = ENDPOINT,
        ContentType  = "application/json",
        Accept       = "application/json",
        Body         = json.dumps(payload)
    )
    log(f"응답 수신 ({time.time()-t0:.1f}s)")

    # ── 4) 결과 디코딩 & 저장 ────────────────────────────────
    response_body = resp["Body"].read()
    log(f"Response size: {len(response_body) // 1024} KB")
    
    try:
        out = json.loads(response_body)
        img = Image.open(io.BytesIO(base64.b64decode(out["image_base64"])))
        out_path = "result.png"
        img.save(out_path)
        log(f"완료 → {out_path} 저장")
    except Exception as e:
        log(f"Failed to process response: {str(e)}")
        with open("response_debug.txt", "w") as f:
            f.write(str(response_body[:1000]) + "...")  # Save first 1000 chars for debugging
        raise

except Exception:
    log("‼ 예외 발생 — traceback ↓")
    traceback.print_exc()
    sys.exit(1)