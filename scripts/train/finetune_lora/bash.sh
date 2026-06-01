#!/bin/bash
# Unified LoRA finetune dispatcher for ALL AVA capabilities.
#
# Usage:
#   bash scripts/train/finetune_lora/bash.sh <MODEL> <AVA>
#
#   <MODEL>  vision tower key (see VT_MAP below): dinov2 clip siglip siglip2
#            siglip2_400m sam intern midas aimv2 radio convnext dinov3
#   <AVA>    capability (see AVA tables below): abs_depth action color counting
#            emotion fine_grained localization ocr orientation recognition
#            rel_depth scene_cls spatial_bbox texture
#
# Examples:
#   bash scripts/train/finetune_lora/bash.sh dinov2 ocr
#   bash scripts/train/finetune_lora/bash.sh siglip2 localization

MODEL=$1
AVA=$2
ROOT="./" #change path
AVA_BENCH="$ROOT/AVA-Bench/train"

# =============================================================================
# Vision tower id  +  pretrained (stage-2 finetune) checkpoint name, per model
# =============================================================================
declare -A VT_MAP=(
    [dinov2]="facebook/dinov2-large"
    [clip]="openai/clip-vit-large-patch14-336"
    [siglip]="google/siglip-so400m-patch14-384"
    [siglip2]="google/siglip2-so400m-patch14-384"
    [sam]="facebook/sam-vit-large"
    [intern]="OpenGVLab/InternViT-300M-448px-V2_5"
    [midas]="Intel/dpt-hybrid-midas"
    [aimv2]="apple/aimv2-huge-patch14-336"
    [radio]="nvidia/RADIO-L"
)

declare -A PRETRAINED_MAP=(
    [dinov2]="llava-Qwen2-0.5B-dinov2-large-qwen2-0_5b_base-finetune"
    [clip]="llava-Qwen2-0.5B-clip-vit-large-patch14-336-qwen2-0_5b_base-finetune"
    [siglip]="llava-Qwen2-0.5B-siglip-so400m-patch14-384-qwen2-0_5b_base-finetune"
    [siglip2]="llava-Qwen2-0.5B-siglip2-so400m-patch14-384-qwen2-0_5b_base-finetune"
    [sam]="llava-Qwen2-0.5B-sam-vit-large-qwen2-0_5b_base-finetune"
    [intern]="llava-Qwen2-0.5B-InternViT-300M-448px-V2_5-qwen2-0_5b_base-finetune"
    [midas]="llava-Qwen2-0.5B-dpt-hybrid-midas-qwen2-0_5b_base-finetune"
    [aimv2]="llava-Qwen2-0.5B-aimv2-huge-patch14-336-qwen2-0_5b_base-finetune"
    [radio]="llava-Qwen2-0.5B-RADIO-L-qwen2-0_5b_base-finetune"
)

# =============================================================================
# Per-AVA configuration. Columns kept identical to the original 14 scripts.
#   DATA / IMG are relative to $AVA_BENCH ; IMG="" means $AVA_BENCH itself.
#   OUTREL is relative to $ROOT/checkpoints/llava_factory ; "" means that root.
# =============================================================================
declare -A DATA=(
    [abs_depth]="Absolute_depth/train.json"
    [action]="Action/train.json"
    [color]="Color/train.json"
    [counting]="Counting/train.json"
    [emotion]="Emotion/train.json"
    [fine_grained]="Fine-grained/train.json"
    [localization]="Localization/train.json"
    [ocr]="OCR/train.json"
    [orientation]="Orientation/train.json"
    [recognition]="Recognition/train.json"
    [rel_depth]="Relative_depth/train.json"
    [scene_cls]="Scene_Classification/train.json"
    [spatial_bbox]="Spatial/train.json"
    [texture]="Texture/train.json"
)

declare -A OUTREL=(
    [abs_depth]="capabilities/abs_depth"
    [action]="capabilities/action"
    [color]="capabilities/color"
    [counting]="capabilities/counting"
    [emotion]="capabilities/emotion"
    [fine_grained]="capabilities/fine_grained"
    [localization]="capabilities/localization"
    [ocr]="capabilities/ocr"                                 
    [orientation]="capabilities/orientation"
    [recognition]="capabilities/recognition"
    [rel_depth]="capabilities/rel_depth"
    [scene_cls]="capabilities/scene_cls"
    [spatial_bbox]="capabilities/spatial_bbox"
    [texture]="capabilities/texture"
)
declare -A CAP=(
    [abs_depth]="AbsDepth"      [action]="Action"          [color]="Color"
    [counting]="Counting"       [emotion]="Emotion"        [fine_grained]="Fine-grained"
    [localization]="Localization" [ocr]="OCR"              [orientation]="Orientation"
    [recognition]="Recognition" [rel_depth]="RelDepth"     [scene_cls]="Scene"
    [spatial_bbox]="Spatial_bbox" [texture]="texture"
)
declare -A EPOCH=(
    [abs_depth]=10 [action]=10 [color]=10 [counting]=10 [emotion]=10 [fine_grained]=10
    [localization]=20 [ocr]=10 [orientation]=10 [recognition]=10 [rel_depth]=10
    [scene_cls]=10 [spatial_bbox]=10 [texture]=10
)

# =============================================================================
# Resolve + validate
# =============================================================================
if [ -z "$MODEL" ] || [ -z "$AVA" ]; then
    echo "Usage: bash scripts/train/finetune_lora/bash.sh <MODEL> <AVA>" >&2
    exit 1
fi

VT_VERSION="${VT_MAP[$MODEL]}"
PRETRAINED_NAME="${PRETRAINED_MAP[$MODEL]}"

if [ -z "$VT_VERSION" ]; then
    echo "Error: vision tower key '$MODEL' not recognized." >&2
    exit 1
fi
if [ -z "${DATA[$AVA]+x}" ]; then
    echo "Error: AVA capability '$AVA' not recognized." >&2
    exit 1
fi

EP="${EPOCH[$AVA]}"
RUN_NAME="${MODEL}_qwen2_lora_epoch_${EP}_${CAP[$AVA]}"
DATA_PATH="$AVA_BENCH/${DATA[$AVA]}"
IMAGE_PATH="$AVA_BENCH/"
OUTPUT_DIR="$ROOT/checkpoints/${OUTREL[$AVA]}/$RUN_NAME"
PRETRAINED_MODEL_PATH="$ROOT/checkpoints/fine_tuned_models/$PRETRAINED_NAME"

# ---- auto-download data from HuggingFace (act13/AVA-Bench) if missing ---------
# The capability dir name (== HF config name) is the parent dir of DATA[$AVA].
HF_CAP="$(dirname "${DATA[$AVA]}")"
if [ ! -f "$DATA_PATH" ]; then
    echo "==> $DATA_PATH not found; preparing '$HF_CAP' from HuggingFace act13/AVA-Bench"
    python scripts/train/finetune_lora/prepare_data.py --cap "$HF_CAP" --out-root "$AVA_BENCH"
fi

echo "==> LoRA finetune | model=$MODEL ($VT_VERSION) | ava=$AVA"
echo "    run_name=$RUN_NAME"
echo "    pretrained=$PRETRAINED_MODEL_PATH"
echo "    output=$OUTPUT_DIR"

bash scripts/train/finetune_lora/train_lora.sh \
    "$VT_VERSION" \
    "$PRETRAINED_MODEL_PATH" \
    "$RUN_NAME" \
    "$DATA_PATH" \
    "$IMAGE_PATH" \
    "$OUTPUT_DIR" \
    "$EP"
