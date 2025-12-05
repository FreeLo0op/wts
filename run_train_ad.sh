#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 根据实际情况修改 GPU 编号
export OMP_NUM_THREADS=1

# 路径配置
MODEL_PATH=/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v2.3/pt_model
TRAIN_DATA=/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/audio_detect/train/ad_train_1202_softmax_semantic_codes.json
OUTPUT_DIR=/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/ad_model/ad_model_v2.0

# 训练参数
NUM_LABELS=8
BATCH_SIZE=4
NUM_HIDDEN_LAYERS=3

LEARNING_RATE=2e-5
EPOCHS=3
MAX_SEQ_LEN=512

# 启动训练
# 使用 torchrun 进行分布式训练 (单机多卡)
torchrun --nproc_per_node=8 --master_port=29500 -m multi_class_model.train \
    --model_path "$MODEL_PATH" \
    --train_data_path "$TRAIN_DATA" \
    --eval_ratio 0.2 \
    --output_dir "$OUTPUT_DIR" \
    --num_labels $NUM_LABELS \
    --max_seq_length $MAX_SEQ_LEN \
    --num_hidden_layers $NUM_HIDDEN_LAYERS \
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

cp /mnt/pfs_l2/jieti_team/SFT/hupeng/github/wts/run_train_ad.sh "$OUTPUT_DIR"/run_train.sh