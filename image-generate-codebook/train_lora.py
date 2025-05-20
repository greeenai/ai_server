"""
train_lora.py
Stable Diffusion XL 모델을 위한 LoRA 기반 text-to-image finetuning train 코드입니다.

1. 학습 실행 명령어 - 자신의 환경에 맞게 경로를 수정이 필요합니다.
accelerate launch --mixed_precision fp16 \
  train_lora.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --instance_data_dir "${INSTANCE_DIR}" 
  --instance_prompt  "${STYLE_TOKEN}" \
  --output_dir "${OUTPUT_DIR}" \
  --resolution 1024 \
  --train_batch_size 1 \
  --num_train_epochs 8 \
  --learning_rate 5e-5 \
  --rank 8 \
  --gradient_checkpointing \
  --checkpointing_steps 500
  --cache_dir "${CACHE_DIR}"
  
2. 제 경우 학습시킨 이미지는 AI 허브 데이터셋에서 다운받은 구름빵 이미지를 사용하여 다음과 같이 INSTANCE_DIR와 STYLE_TOKEN을 설정했습니다. 
--instance_data_dir "cloud_bread_dataset/190.애니메이션_속_캐릭터_얼굴_랜드마크_데이터/01.데이터/1.Training/원천데이터" \
--instance_prompt "cloud-bread style"

# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params, compute_snr
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb

# diffusers 최소 버전 확인
check_min_version("0.34.0.dev0")

logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def save_model_card(
    repo_id: str,
    images: list = None,
    base_model: str = None,
    dataset_name: str = None,
    train_text_encoder: bool = False,
    repo_folder: str = None,
    vae_path: str = None,
):
    """
    모델 카드 생성 및 저장 함수
    
    Args:
        repo_id: 모델 저장소 ID
        images: 예시 이미지 리스트
        base_model: 기반 모델 경로
        dataset_name: 데이터셋 이름
        train_text_encoder: 텍스트 인코더 학습 여부
        repo_folder: 저장소 폴더 경로
        vae_path: 사용된 VAE 경로
    """
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# LoRA text2image fine-tuning - {repo_id}

These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}

LoRA for the text encoder was enabled: {train_text_encoder}.

Special VAE used for training: {vae_path}.
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion-xl",
        "stable-diffusion-xl-diffusers",
        "text-to-image",
        "diffusers",
        "diffusers-training",
        "lora",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    pipeline,
    args,
    accelerator,
    epoch,
    is_final_validation=False,
):
    """
    검증 이미지 생성 및 로깅 함수
    
    Args:
        pipeline: 파이프라인 객체
        args: 학습 인자
        accelerator: 가속기 객체
        epoch: 현재 에폭
        is_final_validation: 최종 검증 여부
    
    Returns:
        생성된 이미지 리스트
    """
    logger.info(
        f"검증 진행 중... \n 프롬프트 '{args.validation_prompt}'로 {args.num_validation_images}개 이미지 생성."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # 추론 실행
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    pipeline_args = {"prompt": args.validation_prompt}
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)]

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )
    return images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    """
    모델 이름/경로에서 텍스트 인코더 클래스 가져오기
    
    Args:
        pretrained_model_name_or_path: 사전 학습된 모델 이름/경로
        revision: 리비전
        subfolder: 하위 폴더 이름
    
    Returns:
        텍스트 인코더 클래스
    """
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} 모델 클래스는 지원되지 않습니다.")


def parse_args(input_args=None):
    """
    명령줄 인자 파싱 함수
    
    Args:
        input_args: 입력 인자
    
    Returns:
        파싱된 인자
    """
    parser = argparse.ArgumentParser(description="학습 스크립트 예제.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="사전 학습된 모델 경로 또는 huggingface.co/models의 모델 식별자",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="더 나은 수치 안정성을 가진 사전 학습된 VAE 모델 경로. 상세 정보: https://github.com/huggingface/diffusers/pull/4038",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="huggingface.co/models의 사전 학습된 모델 식별자 리비전",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="huggingface.co/models의 사전 학습된 모델 파일 변형, 예: 'fp16'",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "학습할 데이터셋 이름(HuggingFace hub에서). 개인 데이터셋일 수도 있습니다. "
            "로컬 파일 시스템의 데이터셋 복사본 경로이거나 🤗 Datasets가 이해할 수 있는 파일을 포함하는 폴더일 수도 있습니다."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="데이터셋 설정, 단일 설정만 있는 경우 None으로 유지",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "학습 데이터가 포함된 폴더. 폴더 내용은 https://huggingface.co/docs/datasets/image_dataset#imagefolder에 설명된 구조를 따라야 합니다."
            " 특히 이미지 캡션을 제공하는 `metadata.jsonl` 파일이 존재해야 합니다. `dataset_name`이 지정된 경우 무시됩니다."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="이미지를 포함하는 데이터셋 열"
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="캡션 또는 캡션 목록을 포함하는 데이터셋 열",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="모델이 학습되고 있는지 확인하기 위해 검증 중에 사용되는 프롬프트",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="`validation_prompt`로 검증 중에 생성되어야 하는 이미지 수",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "X 에폭마다 검증 실행. 검증 과정은 `args.validation_prompt` 프롬프트를 "
            "`args.num_validation_images` 횟수만큼 실행하는 것으로 구성됩니다."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "디버깅 목적이나 더 빠른 학습을 위해, 설정된 경우 학습 예제 수를 이 값으로 자릅니다."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="모델 예측 및 체크포인트가 기록될 출력 디렉토리",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="다운로드된 모델과 데이터셋이 저장될 디렉토리",
    )
    parser.add_argument("--seed", type=int, default=None, help="재현 가능한 학습을 위한 시드")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "입력 이미지의 해상도, 학습/검증 데이터셋의 모든 이미지는 이 해상도로 리사이즈됩니다."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "입력 이미지를 해상도에 맞게 중앙 자르기할지 여부. 설정하지 않으면 이미지는 무작위로 잘립니다. "
            "자르기 전에 이미지는 먼저 해상도로 리사이즈됩니다."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="이미지를 수평으로 무작위 뒤집기 여부",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="텍스트 인코더 학습 여부. 설정하면 텍스트 인코더는 float32 정밀도여야 합니다.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="학습 데이터로더의 배치 크기(장치당)"
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="수행할 총 학습 단계 수. 제공되면 num_train_epochs를 재정의합니다.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "X 업데이트마다 학습 상태의 체크포인트를 저장합니다. 이러한 체크포인트는 마지막 체크포인트보다 "
            "더 좋은 경우 최종 체크포인트로 사용할 수 있으며, `--resume_from_checkpoint`를 사용하여 학습을 "
            "재개하는 데에도 적합합니다."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("저장할 체크포인트의 최대 수"),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "이전 체크포인트에서 학습을 재개할지 여부. `--checkpointing_steps`로 저장된 경로 사용, "
            "또는 `\"latest\"`를 사용하여 마지막으로 사용 가능한 체크포인트를 자동으로 선택합니다."
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="역방향/업데이트 패스를 수행하기 전에 누적할 업데이트 단계 수",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="메모리를 절약하기 위해 역방향 패스 속도를 희생시키면서 그래디언트 체크포인팅을 사용할지 여부",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="초기 학습률(잠재적 워밍업 기간 이후)",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="GPU 수, 그래디언트 누적 단계 및 배치 크기별로 학습률 조정",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            '사용할 스케줄러 유형. 다음 중 선택 ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="lr 스케줄러에서 워밍업 단계 수"
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="손실을 재조정할 때 사용할 SNR 가중치 감마. 권장 값은 5.0입니다. "
        "자세한 내용은 여기를 참조하세요: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Ampere GPU에서 TF32를 허용할지 여부. 학습 속도를 높이는 데 사용할 수 있습니다. 자세한 정보는 "
            "https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices 참조"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "데이터 로딩에 사용할 하위 프로세스 수. 0은 데이터가 메인 프로세스에서 로드됨을 의미합니다."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="bitsandbytes의 8비트 Adam을 사용할지 여부"
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam 옵티마이저의 beta1 매개변수")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam 옵티마이저의 beta2 매개변수")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="사용할 가중치 감쇠")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Adam 옵티마이저의 엡실론 값")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="최대 그래디언트 노름")
    parser.add_argument("--push_to_hub", action="store_true", help="모델을 Hub에 푸시할지 여부")
    parser.add_argument("--hub_token", type=str, default=None, help="Model Hub에 푸시하는 데 사용할 토큰")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="학습에 사용할 prediction_type. 'epsilon'이나 'v_prediction' 중 선택하거나 `None`으로 둡니다. `None`으로 두면 스케줄러의 기본 prediction_type이 선택됩니다: `noise_scheduler.config.prediction_type`.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="로컬 `output_dir`과 동기화할 저장소 이름",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) 로그 디렉토리. 기본값은 "
            "*output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            '결과와 로그를 보고할 통합. 지원되는 플랫폼은 `"tensorboard"`'
            ' (기본값), `"wandb"` 및 `"comet_ml"`입니다. 모든 통합에 보고하려면 `"all"`을 사용하세요.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "mixed precision 사용 여부. fp16과 bf16(bfloat16) 중 선택. Bf16은 PyTorch >="
            " 1.10 및 Nvidia Ampere GPU가 필요합니다. 기본값은 현재 시스템의 accelerate 설정 값이나"
            " `accelerate.launch` 명령과 함께 전달된 플래그 값입니다. 이 인수를 사용하여 accelerate 설정을 재정의합니다."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="분산 학습용: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="xformers 사용 여부"
    )
    parser.add_argument(
        "--enable_npu_flash_attention", action="store_true", help="npu flash attention 사용 여부"
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="노이즈 오프셋 스케일")
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("LoRA 업데이트 행렬의 차원"),
    )
    parser.add_argument(
        "--debug_loss",
        action="store_true",
        help="데이터셋에 파일 이름을 사용할 수 있는 경우 각 이미지의 손실 디버깅",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # 유효성 검사
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("데이터셋 이름이나 학습 폴더가 필요합니다.")

    return args


# 데이터셋 이름 매핑
DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}


def tokenize_prompt(tokenizer, prompt):
    """
    프롬프트 토큰화 함수
    
    Args:
        tokenizer: 토크나이저 객체
        prompt: 토큰화할 프롬프트
    
    Returns:
        토큰화된 입력 ID
    """
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


# pipelines.StableDiffusionXLPipeline.encode_prompt 에서 수정됨
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    """
    프롬프트 인코딩 함수
    
    Args:
        text_encoders: 텍스트 인코더 리스트
        tokenizers: 토크나이저 리스트
        prompt: 인코딩할 프롬프트
        text_input_ids_list: 미리 토큰화된 입력 ID 리스트
    
    Returns:
        인코딩된 프롬프트 임베딩 및 풀링된 프롬프트 임베딩
    """
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
        )

        # 항상 최종 텍스트 인코더의 풀링된 출력에만 관심이 있음
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def main(args):
    """
    메인 학습 함수
    
    Args:
        args: 명령줄 인자
    """
    if args.report_to == "wandb" and args.hub_token is not None: