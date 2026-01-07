#!/bin/bash
set -e

# 设置 python 路径，确保能找到 verl 包
export PYTHONPATH=$PYTHONPATH:$(pwd)
export ALFWORLD_DATA=~/.cache/alfworld

# 调试配置
# 如果你是 4090 或 A100，可以直接跑 Qwen2.5-7B
MODEL_PATH="/home/zzh/Workspace/modelscope/models/Qwen/Qwen2___5-VL-3B-Instruct" # 请修改为你本地的模型路径

# 获取绝对路径的 config 目录
CONFIG_DIR="$(pwd)/recipe/ccapo/config"
CONFIG_NAME="alfworld_ccapo"

echo "Using Config Dir: $CONFIG_DIR"
echo "Using Config Name: $CONFIG_NAME"

# 运行指令
# 注意：我们使用 --config-path 和 --config-name 来指定配置文件的位置
python3 verl/trainer/main_ppo.py \
    --config-path "$CONFIG_DIR" \
    --config-name "$CONFIG_NAME" \
    actor_rollout_ref.model.path=$MODEL_PATH \
    trainer.n_gpus_per_node=1 \
    data.train_batch_size=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    2>&1 | tee logger/ccapo_debug.log