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

# --- 🔥 [修改] 稍微减小数据量，先跑通 ---
train_data_size=8
val_data_size=4
group_size=4
experiment_name="ccapo_alfworld_lora_run1"

echo ">>> [1/2] Generating Local Mock Data..."
python3 make_fake_data.py \
    --train_size $train_data_size \
    --val_size $val_data_size

DATA_DIR="$(pwd)/data/verl-agent/text"

echo ">>> [2/2] Starting CCAPO Training with LoRA (2 GPUs)..."

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=ccapo \
    actor_rollout_ref.rollout.load_format=safetensors \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
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
    env.max_steps=30 \
    env.rollout.n=$group_size \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_ccapo_debug' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    2>&1 | tee logger/ccapo_run.log