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
# from agent_system.reward_manager import ccapo_algos
import gigpo.core_ccapo as ccapo_algos
from agent_system.reward_manager.ccapo_algos import _group_steps_by_traj # For STDB return
from agent_system.multi_turn_rollout.utils import to_list_of_dict
from verl.utils.dataset.rl_dataset import collate_fn
import numpy as np
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

    # ======================= ✅ [V12 修复] 奖励回写 (错位修复) =======================
    def save_reward_components_to_disk(self, steps_list: List[Dict[str, Any]]):
        """
        将内存中计算出的奖励分数写回到磁盘。
        (✅ Fix V5: Added Persistence for Anchor Metadata)
        """
        if not self.config.get("save_reward_components", True):
            return

        # 1. 按 log_dir_path 分组
        steps_by_traj = collections.defaultdict(list)
        for step in steps_list:
            if 'log_dir_path' in step and 'step_index' in step:
                log_dir_path = str(step['log_dir_path'])
                steps_by_traj[log_dir_path].append(step)
        
        # 2. 定义要保存的键
        # ✅ [新增] 将 'anchor_uid' 和 'is_anchor' 加入 macro_keys，以便写入 summary.json
        macro_keys = ['R_tau', 'R_core', 'A_traj', 'anchor_uid', 'is_anchor']
        
        # ✅ [关键修复] 在这里添加 meta_lcs_id, meta_anchor_uid, R_repetition
        # 只有将它们加入这个列表，它们才会被 json.dump 写入磁盘
        reward_keys_to_save = macro_keys + [
            'R_step', 'A_step', 'advantages', 'A_final_raw',
            'R_core_raw', 'R_match_raw', 'R_novelty_bonus', 'R_format_penalty',
            'Z_novelty', 'b_stage', 
            'TokenCost', 'Q_economy',
            'S_necessity', 'S_utility', 'I_action', 'Q_step',
            # --- 新增字段 ---
            'meta_lcs_id',      # 用于 Viewer Tooltip 显示具体匹配的 ID
            'meta_anchor_uid',  # 用于 Viewer Tooltip 显示锚点 UID
            'R_repetition'      # 用于调试重复惩罚
        ]

        updated_count = 0
        
        for log_dir_path, steps in steps_by_traj.items():
            summary_updated = False 
            
            for step in steps:
                try:
                    step_index_val = step['step_index']
                    if hasattr(step_index_val, 'item'):
                        step_index = int(step_index_val.item())
                    else:
                        step_index = int(step_index_val)
                    
                    is_buffer = step.get('is_buffer_data', False)
                    
                    if is_buffer:
                        target_file_index = step_index
                    else:
                        target_file_index = step_index + 1
                    
                    if target_file_index < 1:
                        continue

                    # --- A. 更新 step_details.json ---
                    step_detail_path = os.path.join(log_dir_path, f"step_{target_file_index}", "step_details.json")
                    
                    if os.path.exists(step_detail_path):
                        reward_payload = {}
                        for key in reward_keys_to_save:
                            if key in step:
                                val = step[key]
                                if hasattr(val, 'item'): val = val.item()
                                if hasattr(val, 'dtype'): val = float(val)
                                reward_payload[key] = val
                                
                                # ✅ [V3 兼容性] 为了让 viewer 直接读取而不必深挖 reward_components，
                                # 我们也将这些关键 key 直接放在 payload 顶层 (如果 viewer 逻辑是读取顶层的话)
                                # 但通常我们将它们归档在 reward_payload 中，稍后 update 进 json
                        
                        try:
                            with open(step_detail_path, 'r', encoding='utf-8') as f:
                                s_data = json.load(f)
                            
                            # 1. 写入 reward_components (结构化存储)
                            if "reward_components" not in s_data:
                                s_data["reward_components"] = {}
                            s_data["reward_components"].update(reward_payload)
                            s_data["reward_components"]["note"] = "Populated by dp_actor (V12 Fixed)"

                            # 2. ✅ [关键] 将 viewer 需要的字段直接打平到 JSON 根目录
                            # stdb.py 的 _load_trajectory_from_path 通常读取根目录字段
                            for k in ['I_action', 'meta_lcs_id', 'meta_anchor_uid', 'R_repetition', 'R_tau', 'R_core']:
                                if k in reward_payload:
                                    s_data[k] = reward_payload[k]

                            with open(step_detail_path, 'w', encoding='utf-8') as f:
                                json.dump(s_data, f, indent=4, ensure_ascii=False)
                            
                            updated_count += 1
                        except Exception: pass

                        # --- B. 更新 summary.json (关键修改) ---
                        if not summary_updated:
                            summary_path = os.path.join(log_dir_path, "summary.json")
                            if os.path.exists(summary_path):
                                macro_payload = {}
                                for k in macro_keys:
                                    if k in reward_payload:
                                        macro_payload[k] = reward_payload[k]
                                
                                if macro_payload:
                                    try:
                                        with open(summary_path, 'r', encoding='utf-8') as f:
                                            sum_data = json.load(f)
                                        
                                        if "reward_summary" not in sum_data:
                                            sum_data["reward_summary"] = {}
                                        sum_data["reward_summary"].update(macro_payload)
                                        
                                        # ✅ [冗余备份] 也直接写在根层级，方便 Viewer 读取
                                        if 'anchor_uid' in macro_payload:
                                            sum_data['anchor_uid'] = macro_payload['anchor_uid']
                                        if 'is_anchor' in macro_payload:
                                            sum_data['is_anchor'] = macro_payload['is_anchor']
                                        
                                        with open(summary_path, 'w', encoding='utf-8') as f:
                                            json.dump(sum_data, f, indent=4, ensure_ascii=False)
                                        
                                        summary_updated = True
                                    except: pass
                except Exception: pass

        # if updated_count > 0:
        #      logger.info(f"[Reward Write-Back] 完成。更新了 {updated_count} 个步骤文件。")

        # if updated_count > 0:
        #      logger.info(f"[Reward Write-Back] 完成。更新了 {updated_count} 个步骤文件。")

    def _extract_micro_batch(self, data_proto, indices):
        """
        [CCAPO Helper] 从 DataProto 中提取指定索引的 Micro Batch。
        处理 Tensor 和非 Tensor (List) 数据。
        """
        mb = {}
        # 1. 处理 Tensor 数据
        for k, v in data_proto.batch.items():
            mb[k] = v[indices].to(get_torch_device().current_device())
            
        # 2. 处理非 Tensor 数据 (如 multi_modal_inputs 列表)
        for k, v in data_proto.non_tensor_batch.items():
            try:
                # 列表切片
                mb[k] = [v[idx] for idx in indices.tolist()]
            except TypeError:
                # 如果 v 不是列表 (可能是 None 或其他)，直接复制
                mb[k] = v
        return mb

    @GPUMemoryLogger(role="dp actor (CCAPO)", logger=logger)
    def update_policy_ccapo(self, G_online_batch: DataProto, G_buffer_batch: DataProto, embedding_model, ccapo_config: DictConfig):
        """
        [CCAPO Refactored V9.4] 修复 STDB 判定逻辑
        """
        self.actor_module.train()

        # --- 1. 准备数据列表 ---
        g_online_steps = to_list_of_dict(G_online_batch)
        
        g_calc_steps = g_online_steps
        g_buffer_steps = []
        if G_buffer_batch:
            g_buffer_steps = to_list_of_dict(G_buffer_batch)
            if g_buffer_steps:
                g_calc_steps = g_online_steps + g_buffer_steps 
        
        # --- 2. 计算优势 (Advantage Calculation) ---
        g_calc_steps_with_adv, lambda_sr = ccapo_algos.compute_ccapo_advantages(
            g_calc_steps,
            g_online_steps,
            g_buffer_steps if G_buffer_batch else [], 
            embedding_model,
            ccapo_config
        )
        
        # ======================= [奖励回写] =======================
        steps_to_save = g_calc_steps_with_adv
        self.save_reward_components_to_disk(steps_to_save)
        # ==========================================================

        # --- 3. 重新封装为 DataProto (仅 Online) ---
        online_steps_final = [s for s in g_calc_steps_with_adv if not s.get('is_buffer_data', False)]
        G_online_batch_final = DataProto.from_single_dict(collate_fn(online_steps_final))
        
        # [修改点] 强制置空 Buffer Batch
        G_buffer_batch_final = None 

        def _fix_advantages_tensor(batch_proto):
            if batch_proto and 'advantages' in batch_proto.non_tensor_batch:
                adv_np = batch_proto.non_tensor_batch.pop('advantages')
                batch_proto.batch['advantages'] = torch.tensor(
                    [float(x) if x is not None else 0.0 for x in adv_np], 
                    dtype=torch.float32, device=batch_proto.batch.device
                )
        
        _fix_advantages_tensor(G_online_batch_final)
        
        # --- 4. 准备训练参数 ---
        temperature = G_online_batch.meta_info.get("temperature", 1.0)
        ppo_mini_batch_size = self.config.ppo_mini_batch_size
        ppo_micro_batch_size = self.config.ppo_micro_batch_size_per_gpu or 1
        
        logger.info(f"[CCAPO Training] BC Update Removed. Using Only Online Data. (Online=1.0)")

        # --- 5. 联合训练循环 ---
        metrics = {}
        debug_notes = {}
        
        n_online = G_online_batch_final.batch.batch_size[0]
        n_minis = (n_online + ppo_mini_batch_size - 1) // ppo_mini_batch_size

        for epoch in range(self.config.ppo_epochs):
            indices_online = torch.randperm(n_online)
            
            for i_mini in range(n_minis):
                start_idx_mini = i_mini * ppo_mini_batch_size
                end_idx_mini = min((i_mini + 1) * ppo_mini_batch_size, n_online)
                current_mini_indices = indices_online[start_idx_mini:end_idx_mini]
                n_samples_in_mini = len(current_mini_indices)
                n_micros_in_this_mini = (n_samples_in_mini + ppo_micro_batch_size - 1) // ppo_micro_batch_size
                
                self.actor_optimizer.zero_grad()
                
                for i_micro in range(n_micros_in_this_mini):
                    # Online Forward/Backward
                    start_micro = i_micro * ppo_micro_batch_size
                    end_micro = min((i_micro + 1) * ppo_micro_batch_size, n_samples_in_mini)
                    batch_idx = current_mini_indices[start_micro:end_micro]
                    
                    mb_online = self._extract_micro_batch(G_online_batch_final, batch_idx)
                    
                    entropy_on, log_prob_on = self._forward_micro_batch(
                        mb_online, temperature, calculate_entropy=(self.config.entropy_coeff != 0.0)
                    )
                    
                    response_len_on = mb_online["responses"].size(1)
                    mask_on = mb_online["attention_mask"][:, -response_len_on:]
                    if "loss_mask" in mb_online:
                          mask_on = mb_online["loss_mask"][:, -response_len_on:]

                    adv_batch = mb_online["advantages"]
                    if adv_batch.dim() == 1:
                        adv_batch = adv_batch.unsqueeze(-1)
                    
                    adv_batch = torch.clamp(adv_batch, -4.0, 4.0)

                    pg_loss, pg_clipfrac, ppo_kl, _ = compute_policy_loss(
                        old_log_prob=mb_online["old_log_probs"],
                        log_prob=log_prob_on,
                        advantages=adv_batch,
                        response_mask=mask_on,
                        cliprange=self.config.clip_ratio,
                        cliprange_low=self.config.clip_ratio_low,
                        cliprange_high=self.config.clip_ratio_high,
                        clip_ratio_c=self.config.get("clip_ratio_c", 3.0),
                        loss_agg_mode=self.config.loss_agg_mode
                    )
                    
                    policy_loss_on = pg_loss
                    if self.config.entropy_coeff != 0.0 and entropy_on is not None:
                        entropy_loss = agg_loss(loss_mat=entropy_on, loss_mask=mask_on, loss_agg_mode=self.config.loss_agg_mode)
                        policy_loss_on = policy_loss_on - entropy_loss * self.config.entropy_coeff

                    loss_online_scaled = policy_loss_on / n_micros_in_this_mini
                    loss_online_scaled.backward()
                    
                    append_to_dict(metrics, {
                        "actor/pg_loss_online": pg_loss.detach().item(),
                        "actor/ppo_kl_online": ppo_kl.detach().item(),
                        "actor/clip_frac_online": pg_clipfrac.detach().item(),
                        "actor/adv_mean": adv_batch.mean().item()
                    })

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        # --- 6. 仪表盘日志 ---
        buffer_steps_for_stats = [s for s in g_calc_steps_with_adv if s.get('is_buffer_data', False)]
        self._log_dashboard_metrics(metrics, online_steps_final, buffer_steps_for_stats, lambda_sr)

        # --- 7. 准备 STDB 更新数据 (修复逻辑) ---
        online_trajs_for_stdb = ccapo_algos._group_steps_by_traj(online_steps_final)
        
        # --- 🔥 [关键修复] 扩宽成功判定逻辑 ---
        success_count_stdb = 0
        for uid, steps in online_trajs_for_stdb.items():
            if not steps: continue
            
            # 判定优先级:
            # 1. R_core == 1.0 (CCAPO 计算出的最终成功)
            # 2. won == True (ALFWorld 原生标记)
            # 3. traj_task_completed == True (Legacy)
            
            first_step = steps[0]
            is_success = False
            
            if first_step.get('R_core', 0) == 1.0:
                is_success = True
            elif first_step.get('won', False) or str(first_step.get('won', '')).lower() == 'true':
                is_success = True
            elif first_step.get('traj_task_completed', False):
                is_success = True
            
            if is_success:
                # 必须显式写回 traj_task_completed，因为 stdb.py 依赖这个字段
                for s in steps:
                    s['traj_task_completed'] = True
                success_count_stdb += 1
                
        logger.info(f"[CCAPO Actor] Prepared {len(online_trajs_for_stdb)} trajs for STDB. (Success: {success_count_stdb})")
        
        if success_count_stdb > 0:
            logger.info(f">>> SUCCESS CONFIRMED ({success_count_stdb}). Expecting STDB update.")
        else:
            # Debug: 打印第一条失败轨迹的 keys，方便排查
            if len(online_trajs_for_stdb) > 0:
                sample_step = list(online_trajs_for_stdb.values())[0][0]
                logger.warning(f">>> NO Success detected. Sample Keys: {list(sample_step.keys())}")
                logger.warning(f"    Sample R_core: {sample_step.get('R_core')}, Won: {sample_step.get('won')}")
        # ----------------------------------------------------

        cpu_trajs_for_stdb = {}
        for uid, steps in online_trajs_for_stdb.items():
            cpu_steps = []
            for s in steps:
                cpu_s = {k: (v.cpu().detach() if isinstance(v, torch.Tensor) else v) for k, v in s.items()}
                cpu_steps.append(cpu_s)
            cpu_trajs_for_stdb[uid] = cpu_steps

        return metrics, cpu_trajs_for_stdb, debug_notes

    def _log_dashboard_metrics(self, metrics, online_steps, buffer_steps, lambda_sr):
        """
        [Modified for ALFWorld] 
        增强版仪表盘，修复极值统计问题，并适配 ALFWorld 场景。
        """
        try:
            # 1. 基础 SR
            metrics['actor/lambda_sr_raw'] = lambda_sr
            
            # 2. 准备容器
            online_step_rewards = []
            online_step_advantages = []
            online_step_token_costs = []
            online_traj_rewards = {}
            online_traj_total_steps = {}
            raw_core_values = []
            raw_match_values = []
            
            # --- [新增] 诊断计数器 ---
            diag_total_steps = 0
            diag_lcs_match_count = 0      # 命中 Anchor 的步数
            diag_rep_penalty_count = 0    # 触发重复惩罚的步数
            diag_format_bonus_count = 0   # 拿到格式奖励的步数
            diag_novelty_sum = 0.0        # 新颖性奖励总和
            
            count_total = 0
            count_success = 0
            count_format_error = 0
            count_exec_failure = 0
            
            # 3. 遍历 Online
            for step in online_steps:
                count_total += 1
                diag_total_steps += 1
                
                # Success Check
                is_step_success = step.get('action_success', False)
                if is_step_success: count_success += 1
                
                status = step.get('action_status', '')
                if status.startswith('FORMAT'): count_format_error += 1
                elif status.startswith('FAILURE'): count_exec_failure += 1
                
                # --- 🔥 [关键] 只有当值合理时才加入统计 ---
                r_step = step.get('R_step', 0.0)
                if r_step > -1000.0: # 简单的过滤器
                    online_step_rewards.append(r_step)
                
                online_step_advantages.append(step.get('A_step', 0.0))
                online_step_token_costs.append(step.get('TokenCost', 0.0))
                
                # --- [诊断数据采集] ---
                if step.get('S_utility', 0.0) > 0.0:
                    diag_lcs_match_count += 1
                
                if step.get('R_repetition', 0.0) < 0.0:
                    diag_rep_penalty_count += 1
                    
                if step.get('R_format_penalty', 0.0) > 0.0:
                    diag_format_bonus_count += 1
                
                diag_novelty_sum += step.get('Z_novelty', 0.0)

                if 'traj_uid' in step:
                    uid = step['traj_uid']
                    online_traj_rewards[uid] = step.get('R_tau', 0.0)
                    online_traj_total_steps[uid] = step.get('traj_total_steps', 0)
                
                if 'R_core_raw' in step and step.get('R_core') == 1.0:
                    raw_core_values.append(step['R_core_raw'])
                if 'R_match_raw' in step and step.get('R_core') == -1.0:
                    raw_match_values.append(step['R_match_raw'])

            # 4. 记录统计量 (Mean/Min/Max)
            def _log_stats(prefix, vals):
                if not vals: return
                arr = np.array(vals, dtype=float)
                # 双重保险：numpy 级别的过滤
                valid_mask = arr > -1.0e9
                if not np.any(valid_mask): return
                valid_arr = arr[valid_mask]
                
                metrics[f"{prefix}_mean"] = np.mean(valid_arr)
                metrics[f"{prefix}_min"] = np.min(valid_arr)
                metrics[f"{prefix}_max"] = np.max(valid_arr)

            _log_stats("ccapo_step/online_reward_R_step", online_step_rewards)
            _log_stats("ccapo_step/online_advantage_A_step", online_step_advantages)
            _log_stats("ccapo_efficiency/online_step_token_cost", online_step_token_costs)
            _log_stats("ccapo_traj/online_reward_R_tau", list(online_traj_rewards.values()))
            _log_stats("ccapo_raw/online_R_core_raw", raw_core_values)
            _log_stats("ccapo_raw/online_R_match_raw", raw_match_values)

            # 5. 比例
            eps = 1e-6
            metrics['ccapo_proportions/online_success_rate'] = count_success / (count_total + eps)
            metrics['ccapo_proportions/online_format_error_rate'] = count_format_error / (count_total + eps)
            metrics['ccapo_proportions/online_exec_failure_rate'] = count_exec_failure / (count_total + eps)

            # --- [ALFWorld 适配诊断] ---
            if diag_total_steps > 0:
                metrics['ccapo_diag/lcs_match_ratio'] = diag_lcs_match_count / diag_total_steps
                metrics['ccapo_diag/repetition_loop_ratio'] = diag_rep_penalty_count / diag_total_steps
                metrics['ccapo_diag/format_compliance_ratio'] = diag_format_bonus_count / diag_total_steps
                metrics['ccapo_diag/avg_novelty_strength'] = diag_novelty_sum / diag_total_steps

            # 6. Success vs Fail Traj
            succ_tau, fail_tau = [], []
            succ_lens = [] 
            for uid, r in online_traj_rewards.items():
                if r > 0: 
                    succ_tau.append(r)
                    succ_lens.append(online_traj_total_steps[uid])
                else: 
                    fail_tau.append(r)
            
            _log_stats("ccapo_success/traj_R_tau", succ_tau)
            _log_stats("ccapo_success/traj_length", succ_lens) 
            _log_stats("ccapo_fail/traj_R_tau", fail_tau)

            # 7. Buffer 统计
            if buffer_steps:
                buf_rewards = [s.get('R_step', 0.0) for s in buffer_steps]
                _log_stats("ccapo/buffer_R_step", buf_rewards)

        except Exception as e:
            logger.warning(f"[Dashboard] Error: {e}")