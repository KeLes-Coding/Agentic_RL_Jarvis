set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS
export SWANLAB_API_KEY="oB8w36PCJxKeqwif2ijWz"

export CUDA_VISIBLE_DEVICES="1,2"

# 关键改动：将 train_data_size 设置为 1
# 这将确保每个训练批次只包含一个任务
train_data_size=4
val_data_size=4
group_size=4

# We only use data preparation to indicate the modality and the data size.
# python3 -m examples.data_preprocess.prepare \
#     --mode 'text' \
#     --train_data_size $train_data_size \
#     --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/zzh/Workspace/verl-agent/data/atomic_tasks_list_wiki_train.parquet \
    data.val_files=/home/zzh/Workspace/verl-agent/data/atomic_tasks_list_wiki_val.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=8192 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    data.return_full_prompt=True \
    actor_rollout_ref.model.path=/home/zzh/Workspace/modelscope/models/Qwen/Qwen2___5-VL-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=2e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=jarvis \
    env.seed=0 \
    env.max_steps=20 \
    env.rollout.n=$group_size \
    env.jarvis.jarvis_config_path=agent_system/environments/env_package/jarvis/jarvis_v2/config.yaml \
    trainer.critic_warmup=0 \
    trainer.logger='[console,swanlab]' \
    trainer.project_name='verl_agent_jarvis' \
    trainer.experiment_name='grpo_qwen2.5_vl_2.5b' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    trainer.val_before_train=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    critic.model.lora_rank=32 \
    critic.model.lora_alpha=64 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    $@