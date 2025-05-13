# deploy_sdxl_serverless.py  (실시간 g5.xlarge 버전)

from sagemaker import Model, Session
import botocore, sagemaker
from sagemaker.pytorch import PyTorchModel

bucket   = "sagemaker-ap-northeast-2-762233725630"
model_data = f"s3://{bucket}/sdxl/sdxl_lora_img2img.tar.gz"

image_uri = (
  "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/"
  "pytorch-inference:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker-v1.13"
)

role        = "arn:aws:iam::762233725630:role/ec2"
endpoint    = "sdxl-img2img-g5"

print("Using DLC:", image_uri)

# ----- (선택) 기존 리소스 정리 ------------------------------------------------
sm = sagemaker.Session().sagemaker_client
cleanup = {
    "delete_endpoint"        : "EndpointName",
    "delete_endpoint_config" : "EndpointConfigName",
    "delete_model"           : "ModelName",
}
for api, key in cleanup.items():
    try:
        getattr(sm, api)(**{key: endpoint})
        print(f"{api} ⇒ deleted")
    except botocore.exceptions.ClientError:
        print(f"{api} — nothing to delete")

env = {
    "PYTHONUNBUFFERED": "1",
    "HF_HOME": "/tmp/hf_cache",
    "TRANSFORMERS_CACHE": "/tmp/hf_cache",
    "HUGGINGFACE_HUB_CACHE": "/tmp/hf_cache",
    # g5(A10G) → Compute Capability 8.6
    "TORCH_CUDA_ARCH_LIST": "8.6",
    "MALLOC_TRIM_THRESHOLD_": "0",
}
model = PyTorchModel(
    entry_point      = "inference.py",   # 로컬 파일
    source_dir       = ".",              # requirements.txt, 기타 모듈 포함
    role             = role,
    framework_version= "2.5.1",          # 반드시 컨테이너 버전과 맞춰줍니다
    py_version       = "py311",
    image_uri        = image_uri,
    model_data       = model_data,       # s3 tar.gz (lora_weights/ 등)
    env              = env,
)

# 4. 배포
predictor = model.deploy(
    instance_type          = "ml.g5.xlarge",
    initial_instance_count = 1,
    endpoint_name          = "sdxl-img2img-g4-final-13",
    wait                   = True        # InService 될 때까지 블로킹
)

print("\n🎉  Endpoint ready :", ENDPOINT)

print("\n🎉  Endpoint ready ")
