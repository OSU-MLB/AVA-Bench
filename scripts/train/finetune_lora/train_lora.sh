#!/bin/bash
# Generic LoRA finetune runner. Driven entirely by bash.sh — do not call directly.
# All per-AVA / per-model values arrive as positional args so this file stays static.
#
# Positional args (set by bash.sh):
VT_VERSION=$1            # huggingface vision tower id
PRETRAINED_MODEL_PATH=$2 # full path to the pretrained (stage-2 finetune) checkpoint
RUN_NAME=$3             # wandb / output run name
DATA_PATH=$4            # train.json for this AVA capability
IMAGE_PATH=$5           # image folder for this AVA capability
OUTPUT_DIR=$6           # full output checkpoint dir
EPOCH=$7               # num train epochs

# ---- static settings (identical across all 14 AVAs) --------------------------
MODEL_MAX_LENGTH=2048
TRAIN_RECIPE=lora
LLM_VERSION=Qwen/Qwen2-0.5B
CN_VERSION=mlp2x_gelu
CONV_VERSION=qwen2_base
DS_CONFIG=./scripts/zero3.json
GPUS=0,1,2,3
PORT=29503
FP16=True
BATCH=4
GRAD_ACCUM=1

deepspeed --include localhost:$GPUS --master_port $PORT tinyllava/train/custom_finetune.py \
    --deepspeed $DS_CONFIG \
    --data_path  $DATA_PATH \
    --image_folder $IMAGE_PATH \
    --is_multimodal True \
    --conv_version $CONV_VERSION \
    --model_name_or_path $LLM_VERSION \
    --vision_tower $VT_VERSION \
    --connector_type $CN_VERSION \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --fp16 $FP16 \
    --training_recipe $TRAIN_RECIPE \
    --tune_type_llm lora \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --lora_r 128 \
    --lora_alpha 256 \
    --group_by_modality_length False \
    --pretrained_model_path $PRETRAINED_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BATCH \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --tokenizer_use_fast False \
    --run_name $RUN_NAME
