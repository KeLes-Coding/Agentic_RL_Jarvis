# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

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

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

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
                    is_vlm_model = "multi_modal_inputs" in micro_batch
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
    
    # --- ✅ [CCAPO] 新增：CCAPO 策略更新方法 ---
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
            g_calc_steps = g_online_steps + g_buffer_steps
        
        # --- 2. 计算优势 (Sec 1-5) ---
        # 这会在 g_calc_steps 列表中原地添加 'advantages'
        # ccapo_config 是 config.algorithm.ccapo
        g_calc_steps_with_adv, lambda_sr = ccapo_algos.compute_ccapo_advantages(
            g_calc_steps,
            g_online_steps,
            embedding_model,
            ccapo_config
        )
        
        # --- 3. 重新组合为 DataProto (仍在 GPU 上) ---
        # 过滤出 G_online 并重新 collate
        online_steps_final = [s for s in g_calc_steps_with_adv if not s.get('is_buffer_data', False)]
        G_online_batch_final = DataProto.from_single_dict(collate_fn(online_steps_final))
        
        # 过滤出 G_buffer 并重新 collate
        buffer_steps_final = [s for s in g_calc_steps_with_adv if s.get('is_buffer_data', False)]
        if buffer_steps_final:
            G_buffer_batch_final = DataProto.from_single_dict(collate_fn(buffer_steps_final))
        else:
            G_buffer_batch_final = None

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

        # --- 5. PPO 循环 ---
        for epoch in range(self.config.ppo_epochs):
            # (注意: 我们的实现假设 PPO epochs > 1 时，在整个 G_calc 上重复训练)
            
            self.actor_optimizer.zero_grad()
            
            total_loss = 0.0
            
            # --- 5.1 计算 L_online ---
            if G_online_batch_final and G_online_batch_final.batch_size > 0:
                # 前向传播 G_online
                entropy_online, log_prob_online = self._forward_micro_batch(
                    micro_batch=G_online_batch_final.batch, # 假设整个 batch 是一个 micro_batch
                    temperature=temperature,
                    calculate_entropy=(entropy_coeff != 0.0)
                )
                
                # 获取 response_mask
                response_length_online = G_online_batch_final.batch["responses"].size(1)
                if multi_turn and "loss_mask" in G_online_batch_final.batch:
                    response_mask_online = G_online_batch_final.batch["loss_mask"][:, -response_length_online:]
                else:
                    response_mask_online = G_online_batch_final.batch["attention_mask"][:, -response_length_online:]

                pg_loss_online, pg_clipfrac_online, ppo_kl_online, pg_clipfrac_lower_online = compute_policy_loss(
                    old_log_prob=G_online_batch_final.batch["rollout_log_probs"], # 这是 pi_theta_old
                    log_prob=log_prob_online,
                    advantages=G_online_batch_final.batch["advantages"], # A_final
                    response_mask=response_mask_online,
                    cliprange=clip_ratio,
                    cliprange_low=clip_ratio_low,
                    cliprange_high=clip_ratio_high,
                    clip_ratio_c=clip_ratio_c,
                    loss_agg_mode=loss_agg_mode,
                )
                
                policy_loss_online = pg_loss_online
                if entropy_coeff != 0.0 and entropy_online is not None:
                    entropy_loss_online = agg_loss(loss_mat=entropy_online, loss_mask=response_mask_online, loss_agg_mode=loss_agg_mode)
                    policy_loss_online = policy_loss_online - entropy_loss_online * entropy_coeff
                    if epoch == 0: # 只记录一次
                        append_to_dict(metrics, {"actor/entropy_loss_online": entropy_loss_online.detach().item()})

                total_loss += (1.0 - lambda_sr) * policy_loss_online
                
                if epoch == 0: # 只在第一次 epoch 记录
                    append_to_dict(metrics, {
                        "actor/pg_loss_online": pg_loss_online.detach().item(),
                        "actor/pg_clipfrac_online": pg_clipfrac_online.detach().item(),
                        "actor/ppo_kl_online": ppo_kl_online.detach().item(),
                    })

            # --- 5.2 计算 L_buffer ---
            if G_buffer_batch_final and G_buffer_batch_final.batch_size > 0:
                # 前向传播 G_buffer
                entropy_buffer, log_prob_buffer = self._forward_micro_batch(
                    micro_batch=G_buffer_batch_final.batch,
                    temperature=temperature,
                    calculate_entropy=(entropy_coeff != 0.0)
                )
                
                # 获取 response_mask
                response_length_buffer = G_buffer_batch_final.batch["responses"].size(1)
                if multi_turn and "loss_mask" in G_buffer_batch_final.batch:
                    response_mask_buffer = G_buffer_batch_final.batch["loss_mask"][:, -response_length_buffer:]
                else:
                    response_mask_buffer = G_buffer_batch_final.batch["attention_mask"][:, -response_length_buffer:]

                pg_loss_buffer, pg_clipfrac_buffer, ppo_kl_buffer, pg_clipfrac_lower_buffer = compute_policy_loss(
                    old_log_prob=G_buffer_batch_final.batch["rollout_log_probs"], # 这是 pi_theta_stored
                    log_prob=log_prob_buffer,
                    advantages=G_buffer_batch_final.batch["advantages"], # A_final
                    response_mask=response_mask_buffer,
                    cliprange=clip_ratio,
                    cliprange_low=clip_ratio_low,
                    cliprange_high=clip_ratio_high,
                    clip_ratio_c=clip_ratio_c,
                    loss_agg_mode=loss_agg_mode,
                )

                policy_loss_buffer = pg_loss_buffer
                if entropy_coeff != 0.0 and entropy_buffer is not None:
                    entropy_loss_buffer = agg_loss(loss_mat=entropy_buffer, loss_mask=response_mask_buffer, loss_agg_mode=loss_agg_mode)
                    policy_loss_buffer = policy_loss_buffer - entropy_loss_buffer * entropy_coeff
                    if epoch == 0: # 只记录一次
                        append_to_dict(metrics, {"actor/entropy_loss_buffer": entropy_loss_buffer.detach().item()})

                total_loss += lambda_sr * policy_loss_buffer
                
                if epoch == 0: # 只在第一次 epoch 记录
                    append_to_dict(metrics, {
                        "actor/pg_loss_buffer": pg_loss_buffer.detach().item(),
                        "actor/pg_clipfrac_buffer": pg_clipfrac_buffer.detach().item(),
                        "actor/ppo_kl_buffer": ppo_kl_buffer.detach().item(),
                    })

            # --- 5.3 反向传播和优化 ---
            if isinstance(total_loss, torch.Tensor):
                total_loss.backward()
                grad_norm = self._optimizer_step()
                
                if epoch == 0: # 只在第一次 epoch 记录
                    append_to_dict(metrics, {
                        "actor/total_loss": total_loss.detach().item(),
                        "actor/lambda_sr": lambda_sr,
                        "actor/grad_norm": grad_norm.detach().item()
                    })
            else:
                logger.warning("[CCAPO] No loss computed (total_loss=0.0). Skipping optimizer step.")
                if epoch == 0:
                    append_to_dict(metrics, {"actor/total_loss": 0.0, "actor/lambda_sr": lambda_sr})

        # --- 6. 准备 STDB 更新数据 ---
        # `g_online_steps` 已经被 `compute_ccapo_advantages` 原地修改，包含了 R_tau
        online_trajs_for_stdb = _group_steps_by_traj(g_online_steps)
        
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

        return metrics, cpu_trajs_for_stdb
    # --- 结束 CCAPO 方法 ---
