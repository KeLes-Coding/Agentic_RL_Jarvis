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
        (✅ Fix V4: 严格修复索引错位问题，防止 Step 0 被写入)
        """
        if not self.config.get("save_reward_components", True):
            return

        # logger.info(f"[Reward Write-Back] 开始回写 {len(steps_list)} 个步骤的奖励...")
        
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
                    
                    # --- ✅ [关键索引修复] ---
                    is_buffer = step.get('is_buffer_data', False)
                    
                    if is_buffer:
                        # Buffer 数据 (来自 STDB): 内存索引通常是 1-based (1..T)
                        target_file_index = step_index
                    else:
                        # Online 数据 (来自 Rollout): 内存索引是 0-based (0..T-1)
                        # 需要 +1 才能对应磁盘 step_1..step_T
                        target_file_index = step_index + 1
                    
                    # ⛔️ [严格防御] 绝对不允许写入 step_0 (Reset 目录)
                    if target_file_index < 1:
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
                        try:
                            with open(step_detail_path, 'r', encoding='utf-8') as f:
                                s_data = json.load(f)
                            
                            if "reward_components" not in s_data:
                                s_data["reward_components"] = {}
                            s_data["reward_components"].update(reward_payload)
                            s_data["reward_components"]["note"] = "Populated by dp_actor (V12 Fixed)"

                            with open(step_detail_path, 'w', encoding='utf-8') as f:
                                json.dump(s_data, f, indent=4, ensure_ascii=False)
                            
                            updated_count += 1
                        except Exception as io_e:
                            pass # 文件读写冲突忽略

                        # --- B. 更新 summary.json (宏观数据，每条轨迹只更新一次) ---
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
                                        
                                        with open(summary_path, 'w', encoding='utf-8') as f:
                                            json.dump(sum_data, f, indent=4, ensure_ascii=False)
                                        
                                        summary_updated = True
                                    except: pass
                    else:
                        pass

                except Exception as e:
                    logger.warning(f"[Write-Back] Traj: {log_dir_path} Idx: {step_index} Error: {e}")

        # if updated_count > 0:
        #      logger.info(f"[Reward Write-Back] 完成。更新了 {updated_count} 个步骤文件。")

    @GPUMemoryLogger(role="dp actor (CCAPO)", logger=logger)
    def update_policy_ccapo(self, G_online_batch: DataProto, G_buffer_batch: DataProto, embedding_model, ccapo_config: DictConfig):
        """
        [CCAPO Refactored] 执行 CCAPO 策略更新。
        Mixed Loss: Online(PPO) + Buffer(BC)
        """
        self.actor_module.train()

        # --- 1. 准备数据列表 ---
        g_online_steps = to_list_of_dict(G_online_batch)
        
        g_calc_steps = g_online_steps
        if G_buffer_batch:
            g_buffer_steps = to_list_of_dict(G_buffer_batch)
            if g_buffer_steps:
                g_calc_steps = g_online_steps + g_buffer_steps 
            else:
                g_calc_steps = g_online_steps 
        
        # --- 2. 计算优势 ---
        g_calc_steps_with_adv, lambda_sr = ccapo_algos.compute_ccapo_advantages(
            g_calc_steps,
            g_online_steps,
            g_buffer_steps if G_buffer_batch else [], # 传入 buffer steps 列表以便分离计算
            embedding_model,
            ccapo_config
        )
        
        # ======================= ✅ [恢复] 奖励回写 =======================
        steps_to_save = g_calc_steps_with_adv
        self.save_reward_components_to_disk(steps_to_save)
        # ===============================================================

        # --- 3. 重新封装为 DataProto ---
        online_steps_final = [s for s in g_calc_steps_with_adv if not s.get('is_buffer_data', False)]
        G_online_batch_final = DataProto.from_single_dict(collate_fn(online_steps_final))
        
        buffer_steps_final = [s for s in g_calc_steps_with_adv if s.get('is_buffer_data', False)]
        G_buffer_batch_final = None
        if buffer_steps_final:
            G_buffer_batch_final = DataProto.from_single_dict(collate_fn(buffer_steps_final))

        # 辅助：修复 advantages 类型
        def _fix_advantages_tensor(batch_proto):
            if batch_proto and 'advantages' in batch_proto.non_tensor_batch:
                adv_np = batch_proto.non_tensor_batch.pop('advantages')
                adv_list = [float(x) if x is not None else 0.0 for x in adv_np]
                # 转换为 Tensor 方便后续处理
                batch_proto.batch['advantages'] = torch.tensor(
                    adv_list, dtype=torch.float32, device=batch_proto.batch.device
                )
        
        _fix_advantages_tensor(G_online_batch_final)
        _fix_advantages_tensor(G_buffer_batch_final)

        # --- 4. 准备训练参数 ---
        temperature = G_online_batch.meta_info.get("temperature", 1.0)
        ppo_micro_batch_size = self.config.ppo_micro_batch_size_per_gpu or 1
        
        # ======================= ✅ [V3 稳定性优化] 动态权重调整 =======================
        # 逻辑：
        # SR 低 -> 模型处于"学徒期" -> 增加 Buffer 权重 (复习标准答案) -> 稳定
        # SR 高 -> 模型处于"出师期" -> 增加 Online 权重 (自我探索) -> 效率
        # 设定一个保底值，防止完全遗忘 Buffer
        
        current_sr_val = max(0.0, min(1.0, lambda_sr))
        
        # 基础配置 (可从 config 读取，这里给默认值)
        base_online = 0.5
        max_online = 0.9
        
        # 线性插值: SR=0 -> w=0.5; SR=1 -> w=0.9
        online_weight = base_online + (max_online - base_online) * current_sr_val
        buffer_weight = 1.0 - online_weight
        
        # 如果 Buffer 为空 (冷启动阶段)，全量 Online
        if not G_buffer_batch_final or G_buffer_batch_final.batch.batch_size[0] == 0:
            online_weight = 1.0
            buffer_weight = 0.0
            
        logger.info(f"[CCAPO Weights] SR={current_sr_val:.2f} | Online_W={online_weight:.2f} | Buffer_W={buffer_weight:.2f}")
        # ===========================================================================
        
        metrics = {}
        debug_notes = {}

        # --- 5. 训练循环 ---
        for epoch in range(self.config.ppo_epochs):
            self.actor_optimizer.zero_grad()
            
            batches_to_process = []
            if G_online_batch_final and G_online_batch_final.batch.batch_size[0] > 0:
                batches_to_process.append((G_online_batch_final, online_weight, "online"))
            if G_buffer_batch_final and G_buffer_batch_final.batch.batch_size[0] > 0:
                batches_to_process.append((G_buffer_batch_final, buffer_weight, "buffer"))
            
            if not batches_to_process: break

            total_micro_batches = sum(
                (proto.batch.batch_size[0] + ppo_micro_batch_size - 1) // ppo_micro_batch_size
                for proto, _, _ in batches_to_process
            )
            if total_micro_batches == 0: break

            total_loss_acc = 0.0

            for data_proto, loss_weight, name in batches_to_process:
                batch_size = data_proto.batch.batch_size[0]
                num_micros = (batch_size + ppo_micro_batch_size - 1) // ppo_micro_batch_size
                
                # Shuffle micro-batches for better stability (Optional but recommended)
                indices = torch.randperm(batch_size)
                
                # 手动切分 Chunk
                for i in range(num_micros):
                    start_idx = i * ppo_micro_batch_size
                    end_idx = min((i + 1) * ppo_micro_batch_size, batch_size)
                    batch_indices = indices[start_idx:end_idx]
                    
                    # 提取 Micro Batch 数据
                    mb_dict = {}
                    # Tensor data
                    for k, v in data_proto.batch.items():
                        mb_dict[k] = v[batch_indices].to(get_torch_device().current_device())
                    # Non-tensor data (list slicing)
                    for k, v in data_proto.non_tensor_batch.items():
                        # Handle potential list vs tensor mismatch in non_tensor_batch
                        try:
                            mb_dict[k] = [v[idx] for idx in batch_indices.tolist()]
                        except:
                            mb_dict[k] = v # Fallback
                    
                    # Forward
                    entropy, log_prob = self._forward_micro_batch(
                        micro_batch=mb_dict,
                        temperature=temperature,
                        calculate_entropy=(self.config.entropy_coeff != 0.0)
                    )
                    
                    response_length = mb_dict["responses"].size(1)
                    if "loss_mask" in mb_dict:
                        response_mask = mb_dict["loss_mask"][:, -response_length:]
                    else:
                        response_mask = mb_dict["attention_mask"][:, -response_length:]

                    if name == "online":
                        # [Online] PPO
                        
                        # ======================= ✅ [V3 稳定性优化] 优势截断 =======================
                        # 防止单条极好或极差的轨迹产生过大梯度
                        adv_batch = mb_dict["advantages"]
                        adv_batch = torch.clamp(adv_batch, -4.0, 4.0) # 限制在 +/- 4.0 Sigma 内
                        # =========================================================================

                        pg_loss, pg_clipfrac, ppo_kl, _ = compute_policy_loss(
                            old_log_prob=mb_dict["rollout_log_probs"], # 注意这里要是 tensor
                            log_prob=log_prob,
                            advantages=adv_batch,
                            response_mask=response_mask,
                            cliprange=self.config.clip_ratio,
                            cliprange_low=self.config.clip_ratio_low,
                            cliprange_high=self.config.clip_ratio_high,
                            clip_ratio_c=self.config.get("clip_ratio_c", 3.0),
                            loss_agg_mode=self.config.loss_agg_mode
                        )
                        policy_loss = pg_loss
                        if epoch == 0 and i == 0:
                            append_to_dict(metrics, {
                                "actor/pg_loss_online": pg_loss.detach().item(),
                                "actor/ppo_kl_online": ppo_kl.detach().item(),
                                "actor/clip_frac_online": pg_clipfrac.detach().item(),
                                "actor/adv_mean": adv_batch.mean().item(), # Log Adv Mean
                                "actor/adv_max": adv_batch.max().item()    # Log Adv Max
                            })

                    elif name == "buffer":
                        # [Buffer] BC Loss
                        masked_log_prob = log_prob * response_mask
                        valid_tokens = response_mask.sum() + 1e-6
                        bc_loss = - masked_log_prob.sum() / valid_tokens
                        policy_loss = bc_loss
                        if epoch == 0 and i == 0:
                            append_to_dict(metrics, {"actor/bc_loss_buffer": bc_loss.detach().item()})
                    
                    if self.config.entropy_coeff != 0.0 and entropy is not None:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
                        policy_loss = policy_loss - entropy_loss * self.config.entropy_coeff

                    # Loss Scaling
                    scaled_loss = (loss_weight * policy_loss) / total_micro_batches
                    scaled_loss.backward()
                    total_loss_acc += policy_loss.detach().item() * loss_weight

            grad_norm = self._optimizer_step()
            if epoch == 0:
                append_to_dict(metrics, {
                    "actor/total_weighted_loss": total_loss_acc,
                    "actor/grad_norm": grad_norm.detach().item(),
                    "actor/weight_online": online_weight
                })

        # --- 6. 仪表盘日志 ---
        self._log_dashboard_metrics(metrics, online_steps_final, buffer_steps_final, lambda_sr)

        # --- 7. 准备 STDB 更新数据 ---
        online_trajs_for_stdb = ccapo_algos._group_steps_by_traj(online_steps_final)
        cpu_trajs_for_stdb = {}
        for uid, steps in online_trajs_for_stdb.items():
            cpu_steps = []
            for s in steps:
                cpu_s = {k: (v.cpu().detach() if isinstance(v, torch.Tensor) else v) for k, v in s.items()}
                cpu_steps.append(cpu_s)
            cpu_trajs_for_stdb[uid] = cpu_steps

        return metrics, cpu_trajs_for_stdb, debug_notes

    def _log_dashboard_metrics(self, metrics, online_steps, buffer_steps, lambda_sr):
        """辅助函数：记录丰富的仪表盘指标 (恢复原版统计)"""
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
            
            count_total = 0
            count_success = 0
            count_format_error = 0
            count_exec_failure = 0
            
            # 3. 遍历 Online
            for step in online_steps:
                count_total += 1
                if step.get('action_success', False): count_success += 1
                
                status = step.get('action_status', '')
                if status.startswith('FORMAT'): count_format_error += 1
                elif status.startswith('FAILURE'): count_exec_failure += 1
                
                online_step_rewards.append(step.get('R_step', 0.0))
                online_step_advantages.append(step.get('A_step', 0.0))
                online_step_token_costs.append(step.get('TokenCost', 0.0))
                
                if 'traj_uid' in step:
                    uid = step['traj_uid']
                    online_traj_rewards[uid] = step.get('R_tau', 0.0)
                    online_traj_total_steps[uid] = step.get('traj_total_steps', 0)
                
                if 'R_core_raw' in step and step.get('R_core') == 1.0:
                    raw_core_values.append(step['R_core_raw'])
                if 'R_match_raw' in step and step.get('R_core') == -1.0:
                    raw_match_values.append(step['R_match_raw'])

            # 4. 记录统计量 (Mean/Min/Max/Hist)
            def _log_stats(prefix, vals):
                if not vals: return
                arr = np.array(vals, dtype=float)
                metrics[f"{prefix}_mean"] = np.mean(arr)
                metrics[f"{prefix}_min"] = np.min(arr)
                metrics[f"{prefix}_max"] = np.max(arr)
                metrics[f"{prefix}_hist"] = arr # ✅ 直方图恢复

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

            # 6. Success vs Fail Traj
            succ_tau, fail_tau = [], []
            for uid, r in online_traj_rewards.items():
                if r > 0: succ_tau.append(r)
                else: fail_tau.append(r)
            _log_stats("ccapo_success/traj_R_tau", succ_tau)
            _log_stats("ccapo_fail/traj_R_tau", fail_tau)

            # 7. Buffer 统计
            if buffer_steps:
                buf_rewards = [s.get('R_step', 0.0) for s in buffer_steps]
                _log_stats("ccapo/buffer_R_step", buf_rewards)

        except Exception as e:
            logger.warning(f"[Dashboard] Error: {e}")