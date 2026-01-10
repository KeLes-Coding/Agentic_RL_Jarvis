#!/bin/bash
set -x 

# --- 🔥 [Step 0] 自动清理 (防止僵尸进程) ---
pkill -f "verl.trainer.main_ppo"
pkill -f "ray::"
sleep 2
# ----------------------------------------

export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1

MODEL_PATH="/home/zzh/Workspace/modelscope/models/Qwen/Qwen2___5-VL-3B-Instruct" 
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/agent_system/environments/env_package/alfworld

# ================= 配置区域 =================
# 1. 数据集生成配置
SAMPLE_SIZE=40        # 从ALFWorld提取多少条数据 (-1代表全量，比如3000条)
DATA_SEED=42           # 数据提取种子，不动这个种子，提取的任务永远一样
TRAIN_RATIO=0.8        # 训练集比例

# 2. PPO 训练配置
TRAIN_BATCH_SIZE=8    # PPO update 的 batch size (必须 <= 训练集数量 * rollout.n)
VAL_BATCH_SIZE=8       # 验证时的 batch size
GROUP_SIZE=4           # 组内样本数 (GRPO/CCAPO 核心参数)
EXPERIMENT_NAME="ccapo_alfworld_real_run1"
MAX_STEPS=50           # 环境最大步数
# ===========================================

echo ">>> [1/2] Generating/Updating Real ALFWorld Data..."
# 每次运行都重新生成一次 parquet，确保配置生效 (速度很快)
python3 make_real_alfworld_data.py \
    --total_samples $SAMPLE_SIZE \
    --train_ratio $TRAIN_RATIO \
    --seed $DATA_SEED \
    --output_dir "$(pwd)/data/verl-agent/text"

DATA_DIR="$(pwd)/data/verl-agent/text"

echo ">>> [2/2] Starting CCAPO Training with LoRA (2 GPUs)..."

# 注意：data.train_batch_size 是 PPO 更新时的 batch 大小，不是数据集大小。
# 数据集大小由上面的 python 脚本决定。

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=ccapo \
    actor_rollout_ref.rollout.load_format=safetensors \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=1.0 \
    algorithm.ccapo.stdb_save_path="stdb/alfworld_stdb.json" \
    algorithm.ccapo.stdb_top_k=1 \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=42 \
    env.max_steps=$MAX_STEPS \
    env.rollout.n=$GROUP_SIZE \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_ccapo_debug' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    2>&1 | tee logger/ccapo_run.log