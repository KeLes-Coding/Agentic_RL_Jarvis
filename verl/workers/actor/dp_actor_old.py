# verl/workers/actor/dp_actor.py

import json # <-- ✅ [V3 新增]
import collections # <-- ✅ [V3 新增]
from typing import List, Tuple, Dict, Union, Any # <-- ✅ [V3 新增]
import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

# --- ✅ [CCAPO] 新增 Imports ---
from agent_system.reward_manager import ccapo_algos
from agent_system.reward_manager.ccapo_algos import _group_steps_by_traj # For STDB return
from agent_system.multi_turn_rollout.utils import to_list_of_dict
from verl.utils.dataset.rl_dataset import collate_fn
# --- 结束 ---

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

        self.smoothed_lambda_sr = 0.5 # 0.5 是一个中性的起始值

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch and micro_batch["multi_modal_inputs"]:
            # micro_batch["multi_modal_inputs"] is a list of dicts, e.g., [ {'image': 'path1'}, {'image': 'path2'} ]
            # or [ {'image': tensor1}, {'image': tensor2} ]
            
            # --- ✅ [VLM Buffer 修复] ---
            # 检查第一个元素的第一个键的值，以确定是张量还是字符串
            first_input = micro_batch["multi_modal_inputs"][0]
            if first_input:
                # 确保 first_input 是字典
                first_input_keys = getattr(first_input, 'keys', lambda: [])()
                if not first_input_keys:
                    logger.warning(f"VLM _forward_micro_batch: multi_modal_inputs[0] ' {first_input} ' 没有 .keys()。跳过 VLM 输入。")
                else:
                    first_key = list(first_input_keys)[0]
                    first_val = first_input[first_key]
                    
                    if isinstance(first_val, torch.Tensor):
                        # 原始逻辑：G_online，批处理张量
                        for key in first_input_keys:
                            multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)
                    else:
                        # G_buffer 逻辑：批处理路径 (或其他非张量)
                        for key in first_input_keys:
                            multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            # --- 修复结束 ---

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    # --- ✅ [VLM Buffer 修复] 检查 multi_modal_inputs 是否真的被填充了 ---
                    is_vlm_model = bool(multi_modal_inputs)
                    # --- 修复结束 ---
                    
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics

    # ======================= ✅ [V11 升级] 奖励回写 (含 summary.json 同步) =======================
    def save_reward_components_to_disk(self, steps_list: List[Dict[str, Any]]):
        """
        将内存中计算出的奖励分数写回到磁盘。
        (✅ Fix V3: 区分 Online(+1) 和 Buffer(不偏移) 的索引逻辑，彻底解决首尾丢失问题)
        """
        if not self.config.get("save_reward_components", True):
            return

        logger.info(f"[Reward Write-Back] 开始回写 {len(steps_list)} 个步骤的奖励...")
        
        # 1. 按 log_dir_path 分组
        steps_by_traj = collections.defaultdict(list)
        for step in steps_list:
            if 'log_dir_path' in step and 'step_index' in step:
                log_dir_path = str(step['log_dir_path'])
                steps_by_traj[log_dir_path].append(step)
        
        # 2. 定义要保存的键
        macro_keys = ['R_tau', 'R_core', 'A_traj']
        reward_keys_to_save = macro_keys + [
            'R_step', 'A_step', 'advantages', 'A_final_raw',
            'R_core_raw', 'R_match_raw', 'R_novelty_bonus', 'R_format_penalty',
            'Z_novelty', 'b_stage', 
            'TokenCost', 'Q_economy',
            'S_necessity', 'S_utility', 'I_action', 'Q_step'
        ]

        updated_count = 0
        
        for log_dir_path, steps in steps_by_traj.items():
            summary_updated = False 
            
            for step in steps:
                try:
                    step_index_val = step['step_index']
                    # 强制转换为 int
                    if hasattr(step_index_val, 'item'):
                        step_index = int(step_index_val.item())
                    else:
                        step_index = int(step_index_val)
                    
                    # --- ✅ [关键修复] 动态索引映射 ---
                    is_buffer = step.get('is_buffer_data', False)
                    
                    if is_buffer:
                        # Buffer 数据 (来自 STDB): 内存索引通常是 1-based (1..T)
                        # 直接对应磁盘 step_1..step_T
                        target_file_index = step_index
                    else:
                        # Online 数据 (来自 Rollout): 内存索引是 0-based (0..T-1)
                        # 需要 +1 才能对应磁盘 step_1..step_T
                        target_file_index = step_index + 1
                    
                    # 双重保险：系统生成的 reset 文件夹通常是 step_0，不应该被写入奖励
                    if target_file_index == 0:
                        continue

                    # --- A. 更新 step_details.json ---
                    step_detail_path = os.path.join(log_dir_path, f"step_{target_file_index}", "step_details.json")
                    
                    if os.path.exists(step_detail_path):
                        # 准备 payload
                        reward_payload = {}
                        for key in reward_keys_to_save:
                            if key in step:
                                val = step[key]
                                if hasattr(val, 'item'): val = val.item()
                                if hasattr(val, 'dtype'): val = float(val)
                                reward_payload[key] = val
                        
                        # 读取 -> 更新 -> 写入
                        with open(step_detail_path, 'r', encoding='utf-8') as f:
                            s_data = json.load(f)
                        
                        if "reward_components" not in s_data:
                            s_data["reward_components"] = {}
                        s_data["reward_components"].update(reward_payload)
                        s_data["reward_components"]["note"] = "Populated by dp_actor (V11 Fixed V3)"

                        with open(step_detail_path, 'w', encoding='utf-8') as f:
                            json.dump(s_data, f, indent=4, ensure_ascii=False)
                        
                        updated_count += 1

                        # --- B. 更新 summary.json (宏观数据) ---
                        if not summary_updated:
                            summary_path = os.path.join(log_dir_path, "summary.json")
                            if os.path.exists(summary_path):
                                macro_payload = {}
                                for k in macro_keys:
                                    if k in reward_payload:
                                        macro_payload[k] = reward_payload[k]
                                
                                if macro_payload:
                                    with open(summary_path, 'r', encoding='utf-8') as f:
                                        sum_data = json.load(f)
                                    
                                    if "reward_summary" not in sum_data:
                                        sum_data["reward_summary"] = {}
                                    sum_data["reward_summary"].update(macro_payload)
                                    
                                    with open(summary_path, 'w', encoding='utf-8') as f:
                                        json.dump(sum_data, f, indent=4, ensure_ascii=False)
                                    
                                    summary_updated = True
                    else:
                        # 路径不存在 (可能是 step_index 计算错误或文件未生成)
                        pass

                except Exception as e:
                    logger.warning(f"[Write-Back] 更新失败 (Traj: {log_dir_path}, Idx: {step_index} -> {target_file_index}): {e}")

        if updated_count > 0:
             logger.info(f"[Reward Write-Back] 完成。更新了 {updated_count} 个步骤文件。")
    
    @GPUMemoryLogger(role="dp actor (CCAPO)", logger=logger)
    def update_policy_ccapo(self, G_online_batch: DataProto, G_buffer_batch: DataProto, embedding_model, ccapo_config: DictConfig):
        """
        [CCAPO] 执行 CCAPO 策略更新 (Sec 7)。
        此方法在 Worker 上执行。
        """
        self.actor_module.train()

        # --- 1. 准备 G_calc 和 G_online 列表 ---
        # to_list_of_dict 应该可以处理 GPU Tensors（如果值是 tensor 的话）
        g_online_steps = to_list_of_dict(G_online_batch)
        
        g_calc_steps = g_online_steps
        if G_buffer_batch:
            g_buffer_steps = to_list_of_dict(G_buffer_batch)
            # --- ✅ [CCAPO V3] 关键修正点 ---
            # 即使 g_buffer_steps 为空, g_calc_steps = g_online_steps + [] 
            # 也会创建一个 g_online_steps 的 *浅拷贝*。
            # 这导致 compute_ccapo_advantages 修改的是 *拷贝*，
            # 而 g_online_steps 保持不变。
            if g_buffer_steps:
                g_calc_steps = g_online_steps + g_buffer_steps
            else:
                g_calc_steps = g_online_steps # 保持对象引用相同
        
        # --- 2. 计算优势 (Sec 1-5) ---
        # ccapo_algos.compute_ccapo_advantages 会 *原地修改* g_calc_steps
        # ccapo_config 是 config.algorithm.ccapo
        g_calc_steps_with_adv, lambda_sr = ccapo_algos.compute_ccapo_advantages(
            g_calc_steps,
            g_online_steps,
            embedding_model,
            ccapo_config
        )
        
        # ======================= ✅ [V3 新增] 奖励回写调用 =======================
        # 在数据仍然完整 (G_calc) 且在 Worker 上时，将其写回磁盘
        # ✅ [Fix] 修复 Step 0 记录问题：在此处过滤，不修改 save_reward_components_to_disk 本身
        # steps_to_save = [s for s in g_calc_steps_with_adv if s.get('step_index', 0) != 0]
        # self.save_reward_components_to_disk(steps_to_save)
        # ======================= ✅ [V3 新增] 结束 =======================

        # --- 3. 重新组合为 DataProto (仍在 GPU 上) ---
        # 过滤出 G_online 并重新 collate
        # --- ✅ [CCAPO V3] 修正：从 g_calc_steps_with_adv (修改后的列表) 中过滤 ---
        online_steps_final = [s for s in g_calc_steps_with_adv if not s.get('is_buffer_data', False)]
        G_online_batch_final = DataProto.from_single_dict(collate_fn(online_steps_final))
        
        # 过滤出 G_buffer 并重新 collate
        buffer_steps_final = [s for s in g_calc_steps_with_adv if s.get('is_buffer_data', False)]
        if buffer_steps_final:
            G_buffer_batch_final = DataProto.from_single_dict(collate_fn(buffer_steps_final))
        else:
            G_buffer_batch_final = None

        # 修正: 'advantages' 是一个 np.ndarray(dtype=object)，因为它可能包含 None。
        # 我们必须手动迭代，将 None 转换
        try:
            if G_online_batch_final and 'advantages' in G_online_batch_final.non_tensor_batch:
                # 1. 弹出 object 数组
                adv_np_array_online = G_online_batch_final.non_tensor_batch.pop('advantages')
                
                # 2. 手动迭代，将 None 替换为 0.0，其他转换为 float
                adv_list_online = [float(adv) if adv is not None else 0.0 for adv in adv_np_array_online]
                
                # 3. 从干净的 float 列表创建张量
                G_online_batch_final.batch['advantages'] = torch.tensor(
                    adv_list_online, 
                    dtype=torch.float32, 
                    device=G_online_batch_final.batch.device # 确保在同一设备上
                )
                
            if G_buffer_batch_final and 'advantages' in G_buffer_batch_final.non_tensor_batch:
                # 对 G_buffer 执行相同操作
                adv_np_array_buffer = G_buffer_batch_final.non_tensor_batch.pop('advantages')
                adv_list_buffer = [float(adv) if adv is not None else 0.0 for adv in adv_np_array_buffer]
                G_buffer_batch_final.batch['advantages'] = torch.tensor(
                    adv_list_buffer, 
                    dtype=torch.float32, 
                    device=G_buffer_batch_final.batch.device # 确保在同一设备上
                )
        except Exception as e:
            logger.error(f"[CCAPO] 转换 'advantages' 键时出错: {e}") 
            raise e

        # --- 4. 准备 PPO 更新 (Sec 7) ---
        temperature = G_online_batch.meta_info.get("temperature", 1.0)
        multi_turn = G_online_batch.meta_info.get("multi_turn", False)
        clip_ratio = self.config.clip_ratio
        clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
        clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
        clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
        entropy_coeff = self.config.entropy_coeff
        loss_agg_mode = self.config.loss_agg_mode

        metrics = {}
        
        # --- ✅ [CCAPO V3] 记录详细信号 (包含 V3 键、直方图和比例) ---
        try:
            # --- START: 仪表盘丰富 ---
            # 1. 初始化所有 G_online 数据的收集器
            online_step_rewards = []        # (微观) R_step
            online_step_advantages = []     # (微观) A_step
            online_step_token_costs = []    # (效率) step_token_usage
            
            online_traj_rewards = {}        # (宏观) R_tau (使用 traj_uid 去重)
            online_traj_advantages = {}     # (宏观) A_traj (使用 traj_uid 去重)
            online_traj_total_steps = {}    # (效率) traj_total_steps
            online_traj_total_tokens = {}   # (效率) traj_total_tokens
            
            # (已存在) 原始信号收集器
            raw_core_values = []
            raw_match_values = []
            
            # (已存在) 比例计数器
            count_total_steps = 0
            count_success = 0
            count_format_error = 0
            count_exec_failure = 0
            
            # (已存在) 用于旧直方图的收集器 (保留)
            metrics_to_log = ccapo_algos.collections.defaultdict(list)
            # --- END: 仪表盘丰富 ---

            # 我们只记录 G_online 的信号，以匹配 PPO 损失
            for step in online_steps_final: 
                
                # --- (已存在) 比例统计 ---
                count_total_steps += 1
                if step.get('action_success', False):
                    count_success += 1
                
                status = step.get('action_status', '')
                if status.startswith('FORMAT_ERROR'):
                    count_format_error += 1
                elif status.startswith('FAILURE'):
                    count_exec_failure += 1
                # --- (已存在) 比例统计结束 ---

                # --- (已存在) 旧的直方图记录 (保留) ---
                metrics_to_log['R_tau'].append(step.get('R_tau', 0.0))
                metrics_to_log['A_traj'].append(step.get('A_traj', 0.0))
                metrics_to_log['R_step'].append(step.get('R_step', 0.0))
                metrics_to_log['A_step'].append(step.get('A_step', 0.0))
                metrics_to_log['advantages'].append(step.get('advantages', 0.0))
                metrics_to_log['R_novelty_bonus'].append(step.get('R_novelty_bonus', 0.0))
                metrics_to_log['R_format_penalty'].append(step.get('R_format_penalty', 0.0))
                # --- (已存在) 旧的直方图记录结束 ---

                # --- START: 仪表盘丰富 (数据收集) ---
                # 1. 收集微观(Step)数据
                online_step_rewards.append(step.get('R_step', 0.0))
                online_step_advantages.append(step.get('A_step', 0.0))
                
                # 'TokenCost' 在 ccapo_algos.py 的 _calculate_R_step_success 中计算
                online_step_token_costs.append(step.get('TokenCost', 0.0)) 

                # 2. 收集宏观(Trajectory)数据 (使用字典自动去重)
                traj_uid = step.get('traj_uid')
                if traj_uid:
                    online_traj_rewards[traj_uid] = step.get('R_tau', 0.0)
                    online_traj_advantages[traj_uid] = step.get('A_traj', 0.0)
                    # 'traj_total_steps' 和 'traj_total_tokens' 来自环境/rollout
                    online_traj_total_steps[traj_uid] = step.get('traj_total_steps', 0)
                    online_traj_total_tokens[traj_uid] = step.get('traj_total_tokens', 0)
                # --- END: 仪表盘丰富 (数据收集) ---
                
                # (已存在) 原始信号
                if 'R_core_raw' in step: # 仅 R_core == 1.0 的步骤
                    raw_core_values.append(step.get('R_core_raw', 0.0))
                
                if 'R_match_raw' in step: # 仅 R_core == -1.0 的步骤
                    raw_match_values.append(step.get('R_match_raw', 0.0))
                # --- (已存在) 原始信号结束 ---

            
            # --- (已存在) 旧的直方图日志记录 (保留) ---
            for key, values in metrics_to_log.items():
                if values:
                    metrics[f'ccapo/online_{key}_mean'] = ccapo_algos.np.mean(values)
                    metrics[f'ccapo/online_{key}_std'] = ccapo_algos.np.std(values)
                    # ✅ [新增] 添加直方图
                    metrics[f'ccapo/online_{key}_hist'] = ccapo_algos.np.array(values, dtype=float)

            # --- (已存在) 比例日志记录 ---
            epsilon = 1e-6
            metrics['ccapo_proportions/online_success_rate'] = count_success / (count_total_steps + epsilon)
            metrics['ccapo_proportions/online_format_error_rate'] = count_format_error / (count_total_steps + epsilon)
            metrics['ccapo_proportions/online_exec_failure_rate'] = count_exec_failure / (count_total_steps + epsilon)
            
            # 总失败率 (对应 "false占比")
            total_failure_count = count_format_error + count_exec_failure
            metrics['ccapo_proportions/online_total_failure_rate'] = total_failure_count / (count_total_steps + epsilon)
            
            # 记录原始计数（用于调试）
            metrics['ccapo_counts/online_total_steps'] = count_total_steps
            metrics['ccapo_counts/online_success'] = count_success
            metrics['ccapo_counts/online_format_error'] = count_format_error
            metrics['ccapo_counts/online_exec_failure'] = count_exec_failure
            # --- (已存在) 比例日志记录结束 ---

            # --- START: 仪表盘丰富 (日志记录) ---
            
            # 辅助函数，用于记录 mean/min/max/hist
            def log_stats(metrics_dict, prefix, values_list):
                if not values_list:
                    metrics_dict[f"{prefix}_mean"] = 0.0
                    metrics_dict[f"{prefix}_min"] = 0.0
                    metrics_dict[f"{prefix}_max"] = 0.0
                    # metrics_dict[f"{prefix}_hist"] = ccapo_algos.np.array([], dtype=float) # 可选：记录空直方图
                    return
                
                values_np = ccapo_algos.np.array(values_list, dtype=float)
                metrics_dict[f"{prefix}_mean"] = ccapo_algos.np.mean(values_np)
                metrics_dict[f"{prefix}_min"] = ccapo_algos.np.min(values_np)
                metrics_dict[f"{prefix}_max"] = ccapo_algos.np.max(values_np)
                metrics_dict[f"{prefix}_hist"] = values_np # 也记录直方图

            # 1. 记录微观(Step)指标
            log_stats(metrics, "ccapo_step/online_reward_R_step", online_step_rewards)
            log_stats(metrics, "ccapo_step/online_advantage_A_step", online_step_advantages)
            log_stats(metrics, "ccapo_efficiency/online_step_token_cost", online_step_token_costs)

            # 2. 记录宏观(Trajectory)指标
            log_stats(metrics, "ccapo_traj/online_reward_R_tau", list(online_traj_rewards.values()))
            log_stats(metrics, "ccapo_traj/online_advantage_A_traj", list(online_traj_advantages.values()))
            log_stats(metrics, "ccapo_efficiency/online_traj_total_steps", list(online_traj_total_steps.values()))
            log_stats(metrics, "ccapo_efficiency/online_traj_total_tokens", list(online_traj_total_tokens.values()))
            
            # --- END: 仪表盘丰富 (日志记录) ---

            # --- ✅ [NEW] 仪表盘新增：Success vs Fail 轨迹分组统计 ---
            # 为了不破坏原有代码结构，我们在最后追加这部分逻辑
            succ_traj_R_tau = []
            succ_traj_steps = []
            fail_traj_R_tau = []
            fail_traj_steps = []
            
            for t_uid, r_tau in online_traj_rewards.items():
                t_steps = online_traj_total_steps.get(t_uid, 0)
                # 简单判断：R_tau > 0 为成功 (假设 R_core=1.0, M_steps > 0)
                if r_tau > 0:
                    succ_traj_R_tau.append(r_tau)
                    succ_traj_steps.append(t_steps)
                else:
                    fail_traj_R_tau.append(r_tau)
                    fail_traj_steps.append(t_steps)

            log_stats(metrics, "ccapo_success/traj_R_tau", succ_traj_R_tau)
            log_stats(metrics, "ccapo_success/traj_steps", succ_traj_steps)
            log_stats(metrics, "ccapo_fail/traj_R_tau", fail_traj_R_tau)
            log_stats(metrics, "ccapo_fail/traj_steps", fail_traj_steps)
            # --- [NEW] 结束 ---

            # --- (已存在) 原始信号日志记录 (保留) ---
            if raw_core_values:
                metrics[f'ccapo_raw/online_R_core_raw_mean'] = ccapo_algos.np.mean(raw_core_values)
                metrics[f'ccapo_raw/online_R_core_raw_std'] = ccapo_algos.np.std(raw_core_values)
                metrics[f'ccapo_raw/online_R_core_raw_hist'] = ccapo_algos.np.array(raw_core_values, dtype=float) # ✅ 直方图
            else:
                metrics[f'ccapo_raw/online_R_core_raw_mean'] = 0.0 
                metrics[f'ccapo_raw/online_R_core_raw_std'] = 0.0

            if raw_match_values:
                metrics[f'ccapo_raw/online_R_match_raw_mean'] = ccapo_algos.np.mean(raw_match_values)
                metrics[f'ccapo_raw/online_R_match_raw_std'] = ccapo_algos.np.std(raw_match_values)
                metrics[f'ccapo_raw/online_R_match_raw_hist'] = ccapo_algos.np.array(raw_match_values, dtype=float) # ✅ 直方图
            else:
                metrics[f'ccapo_raw/online_R_match_raw_mean'] = 0.0
                metrics[f'ccapo_raw/online_R_match_raw_std'] = 0.0
            # --- (已存在) 原始信号日志记录结束 ---

            # (可选) 记录 G_buffer 信号
            if buffer_steps_final:
                buffer_metrics_to_log = ccapo_algos.collections.defaultdict(list)
                
                buffer_raw_core_values = []
                
                for step in buffer_steps_final:
                    buffer_metrics_to_log['R_step'].append(step.get('R_step', 0.0))
                    buffer_metrics_to_log['A_step'].append(step.get('A_step', 0.0))
                    buffer_metrics_to_log['advantages'].append(step.get('advantages', 0.0))
                    
                    # --- ✅ [CCAPO V3] 修正：记录 G_buffer 的 V3 键 ---
                    buffer_metrics_to_log['R_novelty_bonus'].append(step.get('R_novelty_bonus', 0.0))
                    buffer_metrics_to_log['R_format_penalty'].append(step.get('R_format_penalty', 0.0))
                    
                    if 'R_core_raw' in step:
                            buffer_raw_core_values.append(step.get('R_core_raw', 0.0))
                
                for key, values in buffer_metrics_to_log.items():
                    if values:
                        metrics[f'ccapo/buffer_{key}_mean'] = ccapo_algos.np.mean(values)
                        metrics[f'ccapo/buffer_{key}_hist'] = ccapo_algos.np.array(values, dtype=float) # ✅ 直方图
                        
                if buffer_raw_core_values:
                    metrics[f'ccapo_raw/buffer_R_core_raw_mean'] = ccapo_algos.np.mean(buffer_raw_core_values)
                    metrics[f'ccapo_raw/buffer_R_core_raw_hist'] = ccapo_algos.np.array(buffer_raw_core_values, dtype=float) # ✅ 直方图
                else:
                    metrics[f'ccapo_raw/buffer_R_core_raw_mean'] = 0.0
                        
        except Exception as e:
            logger.warning(f"[CCAPO] 无法记录详细信号: {e}")
        # --- 结束记录 ---


       # --- 5. PPO 循环 ---
        for epoch in range(self.config.ppo_epochs):

            self.actor_optimizer.zero_grad()

            # --- START: ✅ [Gem 修复] 动态权重逻辑 ---

            # 1. 确定微批次大小
            ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size_per_gpu
            if not ppo_micro_batch_size_per_gpu:
                logger.warning("[CCAPO] ppo_micro_batch_size_per_gpu 未设置, 默认为 1。")
                ppo_micro_batch_size_per_gpu = 1
            
            # --- 1. [Gem 修复] 定义您期望的权重范围 ---
            # 从 ccapo_config 读取，如果未定义则使用您建议的默认值
            min_online_weight = ccapo_config.get("min_online_weight", 0.3)
            max_online_weight = ccapo_config.get("max_online_weight", 0.7)
            
            # lambda_sr 是当前批次的在线成功率
            # 我们使用平滑后的 SR (self.smoothed_lambda_sr) 来防止剧烈振荡
            smoothing_factor = ccapo_config.get("lambda_smoothing_factor", 0.5)
            
            # 2. 更新平滑的 lambda_sr (EMA)
            # lambda_sr 是 *当前批次* 的在线成功率
            self.smoothed_lambda_sr = (smoothing_factor * lambda_sr) + \
                                     (1.0 - smoothing_factor) * self.smoothed_lambda_sr
            
            sr_for_weighting = self.smoothed_lambda_sr 
            # sr_for_weighting = lambda_sr 

            # 3. [Gem 修复] 实现您设想的 (0.4 -> 0.6) 缩放
            # SR=0.0 -> online_weight = 0.4
            # SR=1.0 -> online_weight = 0.6
            # online_weight = min_online_weight + (max_online_weight - min_online_weight) * sr_for_weighting
            # buffer_weight = 1.0 - online_weight
            online_weight = 0.8
            buffer_weight = 0.2

            # # 4. 记录权重
            # append_to_dict(metrics, {
            #     "actor/lambda_sr_raw": lambda_sr,
            #     "actor/lambda_sr_smoothed": sr_for_weighting,
            #     "actor/weight_online": online_weight, # [日志] 现在 SR 越高, 此值越高
            #     "actor/weight_buffer": buffer_weight, # [日志] 现在 SR 越高, 此值越低
            # })

            # # 5. [Gem 修复] 处理特殊情况 (例如 STDB 为空)
            # batches_to_process = []
            # has_online = G_online_batch_final and G_online_batch_final.batch.batch_size[0] > 0
            # has_buffer = G_buffer_batch_final and G_buffer_batch_final.batch.batch_size[0] > 0

            # if has_online and not has_buffer:
            #     # [关键] STDB 为空, 必须 100% 关注 online
            #     online_weight = 1.0
            #     buffer_weight = 0.0
            #     append_to_dict(metrics, {"actor/weight_note": "STDB_empty_force_online_1.0"})
            # elif not has_online and has_buffer:
            #     # (理论上不应发生, 但作为保护) G_online 为空, 100% 关注 buffer
            #     online_weight = 0.0
            #     buffer_weight = 1.0
            #     append_to_dict(metrics, {"actor/weight_note": "Online_empty_force_buffer_1.0"})
            # elif not has_online and not has_buffer:
            #     # 没有数据，跳过
            #     logger.warning("[CCAPO] G_online 和 G_buffer 均为空。跳过更新。")
            #     pass
            # # else: (has_online and has_buffer)
            #     # 两者都有, 使用上面计算的动态权重
            #     # online_weight 和 buffer_weight 保持不变
            #     append_to_dict(metrics, {"actor/weight_note": "Dynamic_weight_active"})
            debug_notes = {} # <-- [修复] 创建一个单独的字典
    
            append_to_dict(metrics, {
                "actor/lambda_sr_raw": lambda_sr,
                "actor/lambda_sr_smoothed": sr_for_weighting,
                "actor/weight_online": online_weight, 
                "actor/weight_buffer": buffer_weight,
            })

            # 5. [Gem 修复] 处理特殊情况 (例如 STDB 为空)
            batches_to_process = []
            has_online = G_online_batch_final and G_online_batch_final.batch.batch_size[0] > 0
            has_buffer = G_buffer_batch_final and G_buffer_batch_final.batch.batch_size[0] > 0

            if has_online and not has_buffer:
                online_weight = 1.0
                buffer_weight = 0.0
                debug_notes["actor/weight_note"] = "STDB_empty_force_online_1.0" # <-- [修复] 存入新字典
            elif not has_online and has_buffer:
                online_weight = 0.0
                buffer_weight = 1.0
                debug_notes["actor/weight_note"] = "Online_empty_force_buffer_1.0" # <-- [修复] 存入新字典
            elif not has_online and not has_buffer:
                logger.warning("[CCAPO] G_online 和 G_buffer 均为空。跳过更新。")
                pass
            else: # (has_online and has_buffer)
                debug_notes["actor/weight_note"] = "Dynamic_weight_active" # <-- [修复] 存入新字典

            if has_online:
                batches_to_process.append(
                    (G_online_batch_final, online_weight, "online")
                )

            if has_buffer:
                batches_to_process.append(
                    (G_buffer_batch_final, buffer_weight, "buffer")
                )
            # --- END: [Gem 修复] ---

            if not batches_to_process:
                logger.warning("[CCAPO] 没有数据可供更新。跳过此 epoch。")
                continue

            # (保持不变) 
            total_micro_batches = sum(
                (data_proto.batch.batch_size[0] + ppo_micro_batch_size_per_gpu - 1) // ppo_micro_batch_size_per_gpu
                for data_proto, _, _ in batches_to_process
            )

            if total_micro_batches == 0:
                continue

            total_loss_accumulator = 0.0

            for data_proto, loss_weight, name in batches_to_process:
                
                # 5. 将每个批次分割成微批次
                
                # --- 修正: 像 update_policy() 一样分块 DataProto ---
                num_micro_batches_for_this_batch = (data_proto.batch.batch_size[0] + ppo_micro_batch_size_per_gpu - 1) // ppo_micro_batch_size_per_gpu
                if num_micro_batches_for_this_batch == 0:
                    continue
                        
                all_tensor_keys = list(data_proto.batch.keys())
                all_non_tensor_keys = list(data_proto.non_tensor_batch.keys())
                
                micro_batches = data_proto.select(all_tensor_keys, all_non_tensor_keys).chunk(num_micro_batches_for_this_batch)
                # --- 修正结束 ---

                for micro_batch_proto in micro_batches:
                    
                    # --- 修正: 转换为 dict 以支持 VLM ---
                    micro_batch_dict = {
                        **micro_batch_proto.batch.to(get_torch_device().current_device()),
                        **micro_batch_proto.non_tensor_batch
                    }
                    # --- 修正结束 ---

                    entropy, log_prob = self._forward_micro_batch(
                        micro_batch=micro_batch_dict,
                        temperature=temperature,
                        calculate_entropy=(entropy_coeff != 0.0)
                    )
                    
                    response_length = micro_batch_dict["responses"].size(1)
                    
                    if multi_turn and "loss_mask" in micro_batch_dict:
                        response_mask = micro_batch_dict["loss_mask"][:, -response_length:]
                    else:
                        response_mask = micro_batch_dict["attention_mask"][:, -response_length:]

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        # [注意] 这里仍然使用 'rollout_log_probs'
                        old_log_prob=micro_batch_dict["rollout_log_probs"],
                        log_prob=log_prob,
                        advantages=micro_batch_dict["advantages"], # <-- [Gem 修复] 使用我们计算的最终优势
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    policy_loss = pg_loss
                    if entropy_coeff != 0.0 and entropy is not None:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss - entropy_loss * entropy_coeff
                        if epoch == 0:
                            append_to_dict(metrics, {f"actor/entropy_loss_{name}": entropy_loss.detach().item()})

                    # 6. 缩放损失：应用 CCAPO 权重 (online_weight/buffer_weight) 和梯度累积缩放
                    scaled_loss = (loss_weight * policy_loss) / total_micro_batches
                    
                    # 7. 反向传播
                    scaled_loss.backward()

                    total_loss_accumulator += policy_loss.detach().item() * loss_weight
                    
                    if epoch == 0:
                        append_to_dict(metrics, {
                            f"actor/pg_loss_{name}": pg_loss.detach().item(),
                            f"actor/pg_clipfrac_{name}": pg_clipfrac.detach().item(),
                            f"actor/ppo_kl_{name}": ppo_kl.detach().item(),
                        })

            # --- 5.3 优化 ---
            # 在处理完所有微批次（online 和 buffer）后，执行一次优化器步骤
            grad_norm = self._optimizer_step()
            
            if epoch == 0:
                append_to_dict(metrics, {
                    "actor/total_loss": total_loss_accumulator,
                    # "actor/lambda_sr": lambda_sr, # 已在权重计算中记录
                    "actor/grad_norm": grad_norm.detach().item()
                })
            
            # --- END: 修正后的逻辑 ---

        # --- 6. 准备 STDB 更新数据 ---
        # --- ✅ [CCAPO V3] 修正：使用 g_calc_steps_with_adv (修改后的列表) ---
        # 我们需要从 *计算了优势* 的列表中提取
        online_steps_for_stdb_grouping = [s for s in g_calc_steps_with_adv if not s.get('is_buffer_data', False)]
        online_trajs_for_stdb = _group_steps_by_traj(online_steps_for_stdb_grouping)
        
        # Tensors (like prompt_vector) 必须被移到 CPU 以返回给 driver
        cpu_trajs_for_stdb = {}
        for traj_uid, steps in online_trajs_for_stdb.items():
            cpu_steps = []
            for step in steps:
                cpu_step = {}
                for k, v in step.items():
                    if isinstance(v, torch.Tensor):
                        cpu_step[k] = v.cpu().detach()
                    else:
                        cpu_step[k] = v
                cpu_steps.append(cpu_step)
            cpu_trajs_for_stdb[traj_uid] = cpu_steps

        return metrics, cpu_trajs_for_stdb, debug_notes