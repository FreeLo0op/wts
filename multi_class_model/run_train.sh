#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 根据实际情况修改 GPU 编号
export OMP_NUM_THREADS=1

# 路径配置
MODEL_PATH="/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v2.7/pt_model"
TRAIN_DATA=/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/sft/train/snt_pa_flu_train_semantic_codes.json
EVAL_DATA=/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/sft/dev/snt_pa_flu_eval_semantic_codes.json
OUTPUT_DIR="/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/ts_model/output_multiclass_flu"

# 训练参数
NUM_LABELS=4
BATCH_SIZE=8
LEARNING_RATE=2e-5
EPOCHS=3
MAX_SEQ_LEN=2048

# 启动训练
# 使用 torchrun 进行分布式训练 (单机多卡)
torchrun --nproc_per_node=8 --master_port=29500 -m multi_class_model.train \
    --model_path "$MODEL_PATH" \
    --train_data_path "$TRAIN_DATA" \
    --eval_data_path "$EVAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --num_labels $NUM_LABELS \
    --max_seq_length $MAX_SEQ_LEN \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 2 \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $EPOCHS \
    --logging_steps 5 \
    --save_strategy "epoch" \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --save_total_limit 30 \
    --bf16 True \
    --dataloader_num_workers 16 \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --report_to "tensorboard"
