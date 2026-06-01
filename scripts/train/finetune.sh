#!/bin/bash
# Unified finetune launcher: given ANY vision model, finetune it on Qwen2.
#
# Usage:
#   bash scripts/train/finetune/finetune.sh <VT_VERSION>
#
# The ONLY user input is the vision model (VT_VERSION), e.g.:
#   openai/clip-vit-large-patch14-336
#   facebook/dinov2-large
#   google/siglip-so400m-patch14-384
#   google/siglip2-so400m-patch14-384
#   apple/aimv2-huge-patch14-336
#   OpenGVLab/InternViT-300M-448px-V2_5
#   Intel/dpt-hybrid-midas
#   facebook/sam-vit-large
#   nvidia/RADIO-L
#
# Example:
#   bash scripts/train/finetune/finetune.sh facebook/dinov2-large
set -e

# ---- only user-provided variable: the vision model ---------------------------
if [[ -z "$1" || "$1" == -* ]]; then
    echo "Error: vision model (VT_VERSION) is required as the only argument." >&2
    echo "Usage: bash scripts/train/finetune/finetune.sh <VT_VERSION>" >&2
    exit 1
fi
VT_VERSION="$1"

# ---- static settings (do not change at call time) ----------------------------
FINETUNE_DATA_PATH=./dataset/text_files/llava_v1_5_mix665k.json
FINETUNE_IMAGE_PATH=./dataset
LLM_VERSION=Qwen/Qwen2-0.5B
VT_VERSION2=""              # second vision tower; empty unless using mof
CN_VERSION=mlp2x_gelu       # connector type: qformer, resampler, etc.
CONV_VERSION=qwen2_base     # chat template: phi, llama, gemma, etc.
VERSION=qwen2-0_5b_base     # experiment name for recording runs
TRAIN_RECIPE=common         # training recipe: common, lora, qlora
MODEL_MAX_LENGTH=2048

echo "==> Finetuning vision tower '$VT_VERSION' on LLM '$LLM_VERSION'"

bash scripts/train/qwen2/finetune_qwen2.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
