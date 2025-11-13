# verl/workers/actor/dp_actor.py

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
            metrics_to_log = ccapo_algos.collections.defaultdict(list)
            
            # --- ✅ [CCAPO V2] 新增：用于原始分量的专用列表 ---
            raw_core_values = []
            raw_match_values = []
            # --- 结束 ---

            # --- ✅ [CCAPO V4] 新增：用于比例统计的计数器 ---
            count_total_steps = 0
            count_success = 0
            count_format_error = 0
            count_exec_failure = 0
            # --- 结束 ---
            
            # 我们只记录 G_online 的信号，以匹配 PPO 损失
            for step in online_steps_final: 
                
                # --- ✅ [CCAPO V4] 累加比例统计 ---
                count_total_steps += 1
                if step.get('action_success', False):
                    count_success += 1
                
                status = step.get('action_status', '')
                if status.startswith('FORMAT_ERROR'):
                    count_format_error += 1
                elif status.startswith('FAILURE'):
                    count_exec_failure += 1
                # --- 结束 ---

                # 标准化/最终分数
                metrics_to_log['R_tau'].append(step.get('R_tau', 0.0))
                metrics_to_log['A_traj'].append(step.get('A_traj', 0.0))
                metrics_to_log['R_step'].append(step.get('R_step', 0.0))
                metrics_to_log['A_step'].append(step.get('A_step', 0.0))
                metrics_to_log['advantages'].append(step.get('advantages', 0.0)) # 记录最终优势
                
                # --- ✅ [CCAPO V3] 修正：记录新的原始分量 ---
                metrics_to_log['R_novelty_bonus'].append(step.get('R_novelty_bonus', 0.0))
                metrics_to_log['R_format_penalty'].append(step.get('R_format_penalty', 0.0))
                # --- 结束 V3 修正 ---
                
                # --- ✅ [CCAPO V2] 仅在适用时添加原始分量 ---
                if 'R_core_raw' in step: # 仅 R_core == 1.0 的步骤
                    raw_core_values.append(step.get('R_core_raw', 0.0))
                
                if 'R_match_raw' in step: # 仅 R_core == -1.0 的步骤
                    raw_match_values.append(step.get('R_match_raw', 0.0))
                # --- 结束 ---

            
            # 记录 G_online 的 mean/std/hist
            for key, values in metrics_to_log.items():
                if values:
                    metrics[f'ccapo/online_{key}_mean'] = ccapo_algos.np.mean(values)
                    metrics[f'ccapo/online_{key}_std'] = ccapo_algos.np.std(values)
                    # ✅ [新增] 添加直方图
                    metrics[f'ccapo/online_{key}_hist'] = ccapo_algos.np.array(values, dtype=float)

            # --- ✅ [CCAPO V4] 计算并记录比例 ---
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
            # --- 结束 ---

            # --- ✅ [CCAPO V2] 记录新的原始分量 (如果存在) ---
            # 我们使用 'ccapo_raw' 前缀来区分
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
            # --- 结束 ---

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

            # --- START: 修正后的微批次和梯度累积逻辑 ---

            # 1. 确定微批次大小
            ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size_per_gpu
            if not ppo_micro_batch_size_per_gpu:
                logger.warning("[CCAPO] ppo_micro_batch_size_per_gpu 未设置, 默认为 1。")
                ppo_micro_batch_size_per_gpu = 1
            
            # (注意: 我们假设不使用 use_dynamic_bsz，因为您的配置中未启用)

            # 2. 准备所有要处理的数据
            batches_to_process = []
            if G_online_batch_final and G_online_batch_final.batch.batch_size[0] > 0:
                batches_to_process.append(
                    (G_online_batch_final, (1.0 - lambda_sr), "online") # <-- 修正: 传递 DataProto
                )
            
            if G_buffer_batch_final and G_buffer_batch_final.batch.batch_size[0] > 0:
                 batches_to_process.append(
                    (G_buffer_batch_final, lambda_sr, "buffer") # <-- 修正: 传递 DataProto
                )

            if not batches_to_process:
                logger.warning("[CCAPO] 没有数据可供更新。跳过此 epoch。")
                continue

            # 3. 计算总的梯度累积步数
            total_micro_batches = sum(
                (data_proto.batch.batch_size[0] + ppo_micro_batch_size_per_gpu - 1) // ppo_micro_batch_size_per_gpu # <-- 修正
                for data_proto, _, _ in batches_to_process # <-- 修正
            )

            if total_micro_batches == 0:
                continue

            total_loss_accumulator = 0.0 # 用于日志记录

            # 4. 循环处理 G_online 和 G_buffer
            for data_proto, loss_weight, name in batches_to_process: # <-- 修正
                
                # 5. 将每个批次分割成微批次
                
                # --- 修正: 像 update_policy() 一样分块 DataProto ---
                num_micro_batches_for_this_batch = (data_proto.batch.batch_size[0] + ppo_micro_batch_size_per_gpu - 1) // ppo_micro_batch_size_per_gpu
                if num_micro_batches_for_this_batch == 0:
                    continue
                    
                all_tensor_keys = list(data_proto.batch.keys())
                all_non_tensor_keys = list(data_proto.non_tensor_batch.keys())
                
                micro_batches = data_proto.select(all_tensor_keys, all_non_tensor_keys).chunk(num_micro_batches_for_this_batch)
                # --- 修正结束 ---

                for micro_batch_proto in micro_batches: # <-- 修正: micro_batch_proto is a DataProto
                    
                    # --- 修正: 转换为 dict 以支持 VLM ---
                    micro_batch_dict = {
                        **micro_batch_proto.batch.to(get_torch_device().current_device()),
                        **micro_batch_proto.non_tensor_batch
                    }
                    # --- 修正结束 ---

                    entropy, log_prob = self._forward_micro_batch(
                        micro_batch=micro_batch_dict, # <-- 修正
                        temperature=temperature,
                        calculate_entropy=(entropy_coeff != 0.0)
                    )
                    
                    response_length = micro_batch_dict["responses"].size(1) # <-- 修正
                    
                    if multi_turn and "loss_mask" in micro_batch_dict: # <-- 修正
                        response_mask = micro_batch_dict["loss_mask"][:, -response_length:] # <-- 修正
                    else:
                        response_mask = micro_batch_dict["attention_mask"][:, -response_length:] # <-- 修正

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        # --- ✅ [CCAPO V2] 修正：使用 'rollout_log_probs' (来自 STDB/Online) ---
                        old_log_prob=micro_batch_dict["rollout_log_probs"], # <-- 修正
                        log_prob=log_prob,
                        advantages=micro_batch_dict["advantages"], # <-- 修正
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
                        if epoch == 0: # 只记录一次
                            append_to_dict(metrics, {f"actor/entropy_loss_{name}": entropy_loss.detach().item()})

                    # 6. 缩放损失：应用 CCAPO 权重 (lambda_sr) 和梯度累积缩放
                    scaled_loss = (loss_weight * policy_loss) / total_micro_batches
                    
                    # 7. 反向传播（累积梯度）
                    scaled_loss.backward()

                    total_loss_accumulator += policy_loss.detach().item() * loss_weight # 记录未缩放的损失
                    
                    if epoch == 0: # 只在第一次 epoch 记录
                        append_to_dict(metrics, {
                            f"actor/pg_loss_{name}": pg_loss.detach().item(),
                            f"actor/pg_clipfrac_{name}": pg_clipfrac.detach().item(),
                            f"actor/ppo_kl_{name}": ppo_kl.detach().item(),
                        })

            # --- 5.3 优化 ---
            # 在处理完所有微批次（online 和 buffer）后，执行一次优化器步骤
            grad_norm = self._optimizer_step()
            
            if epoch == 0: # 只在第一次 epoch 记录
                append_to_dict(metrics, {
                    "actor/total_loss": total_loss_accumulator, # 记录总损失
                    "actor/lambda_sr": lambda_sr,
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

        return metrics, cpu_trajs_for_stdb
    # --- 结束 CCAPO 方法 ---