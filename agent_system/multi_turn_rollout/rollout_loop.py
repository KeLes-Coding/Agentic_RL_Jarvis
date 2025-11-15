# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from verl.models.transformers.qwen2_vl import get_rope_index
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict, Any

# ======================= ✅ 1. 新增：ROLLOUT 专用文件日志器 ✅ =======================
import os
import logging

# --- 1. 标准日志器 (用于 STDOUT / 主日志) ---
logger = logging.getLogger(__name__)

# --- 2. 专用文件日志器 (用于 logger/ROLLOUT/rollout_operations.log) ---
rollout_file_logger = logging.getLogger("ROLLOUT_FILE")
rollout_file_logger.setLevel(logging.INFO) # 捕获 INFO 及以上级别
rollout_file_logger.propagate = False      # 防止重复记录到 root logger

# 仅在日志器没有处理器时才添加，以防止重复
if not rollout_file_logger.handlers:
    try:
        log_dir = "logger/ROLLOUT" # <--- 新的日志目录
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "rollout_operations.log") # <--- 新的日志文件
        
        # 创建文件处理器 (追加模式)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - [ROLLOUT_FILE] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # 添加处理器
        rollout_file_logger.addHandler(file_handler)
        rollout_file_logger.info("--- ROLLOUT 专用文件日志器已初始化 ---")
        
    except Exception as e:
        logger.error(f"[ROLLOUT] 无法创建专用文件日志器: {e}")
# ======================= 日志设置结束 =======================


class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        
        # ======================= ✅ 2. 分配专用日志器 ✅ =======================
        self.file_logger = rollout_file_logger
        self.file_logger.info("--- 初始化 TrajectoryCollector 实例 ---")
        # =====================================================================

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
        prompt_index: Any,      # <-- ✅ [STDB 修复] 新增
        prompt_vector: Any,     # <-- ✅ [STDB 修复] 新增
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
            prompt_index (Any): [STDB 修复] The prompt_index for this item
            prompt_vector (Any): [STDB 修复] The prompt_vector for this item
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        print(f"\n--- 监控: 进入 preprocess_single_sample (item {item}) ---")
        obs_text_sample = obs.get('text', [''])[item]
        # print(f"收到的 obs['text'] (前200字符): '{obs_text_sample[:200]}'")
        # print(f"收到的 obs['image'] 类型: {type(obs.get('image', [None])[item])}")

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        
        # --- 新增: 提取 ground_truth_answer ---
        # 使用 .get() 来安全地获取，如果键不存在，则提供一个默认值
        ground_truth_answer = gen_batch.non_tensor_batch.get('ground_truth_answer', [""]*len(gen_batch.non_tensor_batch['raw_prompt']))[item]

        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False
        )

        # print(f"生成的 prompt_with_chat_template (前200字符): '{prompt_with_chat_template[:200]}'")
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.config.data.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source,
            # --- 新增: 将 ground_truth_answer 添加到样本字典中 ---
            'ground_truth_answer': ground_truth_answer,
            # --- ✅ [STDB 修复] 将传入的索引添加到 row_dict ---
            'prompt_index': prompt_index,
            'prompt_vector': prompt_vector,
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()

        # print(f"最终 raw_prompt (前200字符): '{raw_prompt[:200]}'")
        # print(f"检查 '<|image_pad|>' 是否在最终 raw_prompt 中: {'<|image_pad|>' in raw_prompt}")
        print(f"--- 监控: 退出 preprocess_single_sample (item {item}) ---\n")

        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # --- ✅ [STDB 修复] 提取一次, 供所有样本使用 ---
        prompt_index_list = gen_batch.non_tensor_batch.get('prompt_index', [None] * batch_size)
        prompt_vector_tensor = gen_batch.batch.get('prompt_vector', None)
        
        if prompt_vector_tensor is None:
            self.file_logger.warning(f"[preprocess_batch] 'prompt_vector' not found in gen_batch.batch. STDB data will be incomplete.")
        if any(p is None for p in prompt_index_list):
            self.file_logger.warning(f"[preprocess_batch] 'prompt_index' not found in gen_batch.non_tensor_batch. STDB data will be incomplete.")
        # --- 修复结束 ---

        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
                # --- ✅ [STDB 修复] 传入 STDB 索引 ---
                prompt_index=prompt_index_list[item],
                prompt_vector=prompt_vector_tensor[item] if prompt_vector_tensor is not None else None
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            gen_batch: DataProto, # <--- ✅ 接收 gen_batch
            total_batch_list: List[List[Dict]],
            total_infos: List[List[Dict]], # <--- ✅ [CCAPO] 接收 total_infos
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts <--- ✅
            total_batch_list (List[List[Dict]]): List of trajectory data for each environment
            total_infos (List[List[Dict]]): List of info dicts from env.step()
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        
        Returns:
            DataProto: Collected and organized trajectory data
        """
        self.file_logger.info(f"--- [gather_rollout_data] 开始聚合 {len(total_batch_list)} 条轨迹 ---")
        
        batch_size = len(total_batch_list)

        episode_rewards_mean = np.mean(episode_rewards)
        episode_rewards_min = np.min(episode_rewards)
        episode_rewards_max = np.max(episode_rewards)

        episode_lengths_mean = np.mean(episode_lengths)
        episode_lengths_min = np.min(episode_lengths)
        episode_lengths_max = np.max(episode_lengths)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        # ======================= ✅ [修复] 从 gen_batch 的正确位置提取 Prompt 级信息 =======================
        self.file_logger.info(f"[gather_rollout_data] 正在从 gen_batch 中提取 STDB 关键信息...")
        self.file_logger.info(f"[gather_rollout_data] gen_batch.batch keys: {list(gen_batch.batch.keys())}")
        self.file_logger.info(f"[gather_rollout_data] gen_batch.non_tensor_batch keys: {list(gen_batch.non_tensor_batch.keys())}")
        
        # 1. prompt_index 是非张量, 来自 non_tensor_batch
        prompt_index_list = gen_batch.non_tensor_batch.get('prompt_index', [None] * batch_size)
        
        # 2. prompt_vector 是张量, 来自 batch
        # ❗️ [修复] 使用 default=None 防止 KeyError 崩溃
        prompt_vector_tensor = gen_batch.batch.get('prompt_vector', None) 
        
        if any(p is None for p in prompt_index_list) or prompt_vector_tensor is None:
             # ❗️ 这个警告现在是关键：如果它在修复 ray_trainer.py 后仍然出现，说明 pop 列表还是错了
             print(f"警告: (gather_rollout_data) 未能从 gen_batch 中提取 'prompt_index' 或 'prompt_vector'。STDB 录入可能会失败。")
             self.file_logger.warning(f"[gather_rollout_data] 未能从 gen_batch 中提取 'prompt_index' (found: {not any(p is None for p in prompt_index_list)}) 或 'prompt_vector' (found: {prompt_vector_tensor is not None})。STDB 将失败。")
        else:
             self.file_logger.info(f"[gather_rollout_data] 成功从 gen_batch 中提取了 'prompt_index' 和 'prompt_vector'。")
        # ======================= 修复结束 =======================
            
        effective_batch = []
        for bs in range(batch_size):
            
            # --- ✅ [CCAPO] Phase 1. 轨迹级别(Macro)数据聚合 ---
            
            # 1.1. 从最后一步的 info 中提取 trajectory summary
            final_summary = {}
            
            # --- ✅ [CCAPO V3] 修正：从最后一个“活动”步骤获取 final_summary ---
            # 'total_batch_list[bs]' 只包含活动步骤 (active_masks=True)
            # 'total_infos[bs]' 包含所有步骤 (包括 done=True 后的填充步骤)
            num_active_steps = len(total_batch_list[bs])
            
            if num_active_steps > 0 and total_infos[bs] and len(total_infos[bs]) >= num_active_steps:
                # 我们需要最后一个活动步骤的 info, 索引是 num_active_steps - 1
                last_active_info = total_infos[bs][num_active_steps - 1]
                final_summary = last_active_info.get('final_summary', {})
            # --- 结束修正 ---

            # 1.2. 提取 $R_\tau$ 所需的轨迹级数据
            # R_success
            traj_task_completed = final_summary.get('task_completed', False)
            # P_steps
            traj_total_steps = episode_lengths[bs] # 这是 TotalSteps_tau
            # P_token
            traj_total_tokens = final_summary.get('token_usage', {}).get('total_tokens', 0)

            # 1.3. 预计算 $N_{success}(\tau)$ (Sec 5.1.2.1, 5.1.3.1)
            traj_n_success_steps = 0
            # --- ✅ [CCAPO V3] 修正：只迭代活动的 info ---
            for i in range(num_active_steps):
                step_info = total_infos[bs][i]
                if step_info.get('action_success', False):
                    traj_n_success_steps += 1
            
            # ======================= ✅ [修复] 获取此轨迹(bs)的 Prompt 级信息 =======================
            traj_prompt_index = prompt_index_list[bs]
            # ❗️ [修复] 安全地从张量中索引
            traj_prompt_vector = prompt_vector_tensor[bs] if prompt_vector_tensor is not None else None
            # ======================= 修复结束 =======================

            # --- ✅ [CCAPO] Phase 2. 步骤级别(Micro)数据聚合 ---
            step_index_in_traj = 0
            
            # ======================= ✅ 日志：在聚合前打印第一个步骤的 STDB 关键信息 =======================
            if bs == 0 and num_active_steps > 0: # 只为第一个轨迹打印
                self.file_logger.info(f"--- [gather_rollout_data] 轨迹 {bs} (uid={traj_uid[bs]}) 的 STDB 关键信息预览 ---")
                self.file_logger.info(f"  - traj_prompt_index: {traj_prompt_index} (Type: {type(traj_prompt_index)})")
                self.file_logger.info(f"  - traj_prompt_vector: {'Tensor' if traj_prompt_vector is not None else 'None'} (Type: {type(traj_prompt_vector)})")
                first_step_info = total_infos[bs][0] if total_infos[bs] else {}
                self.file_logger.info(f"  - log_dir_path (from info): {first_step_info.get('log_dir_path', 'N/A')}")
                self.file_logger.info(f"----------------------------------------------------------")
            # ===================================================================================

            for i, data in enumerate(total_batch_list[bs]): # 这里的 'i' 已经是 0 到 num_active_steps-1
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']: # 这个检查其实多余，因为 total_batch_list[bs] 已经只包含 active
                    
                    # 2.1. 附加轨迹级(Macro)信息到每一步
                    data['traj_task_completed'] = traj_task_completed
                    data['traj_total_steps'] = traj_total_steps
                    data['traj_total_tokens'] = traj_total_tokens
                    data['traj_n_success_steps'] = traj_n_success_steps
                    
                    # 2.2. 附加步骤级(Micro)信息 (来自 infos)
                    step_info = total_infos[bs][i] if i < len(total_infos[bs]) else {}
                    
                    data['step_index'] = step_index_in_traj # $t$
                    data['thought'] = step_info.get('thought', '') # (Sec 5.2)
                    data['parsed_action'] = step_info.get('parsed_action', '') # (Sec 5.2)
                    data['action_type'] = step_info.get('action_type', '') # (Sec 5.1.1)
                    data['action_success'] = step_info.get('action_success', False) # (Sec 5.1.2.1, 5.1.3.1)
                    data['step_token_usage'] = step_info.get('token_usage', {}) # (Sec 5.1.3.2)
                    data['step_confidence'] = step_info.get('confidence_metrics', {}).get('average_confidence', 0.0)
                    data['log_dir_path'] = step_info.get('log_dir_path', '') # <--- ✅ [CCAPO] 新增
                    # --- ✅ [CCAPO V3] 关键修正：添加 action_status ---
                    data['action_status'] = step_info.get('action_status', '')
                    
                    # 2.3. 附加原始的 `rollout` 数据 (用于 IS 和 VF)
                    # `data['rollout_log_probs']` 已经由 `to_list_of_dict(batch)` 自动添加
                    # `data['values']` (如果存在) 也已自动添加
                    
                    # ======================= ✅ [修复] 将 Prompt 级信息附加到每一步 =======================
                    data['prompt_index'] = traj_prompt_index
                    data['prompt_vector'] = traj_prompt_vector
                    # ======================= 修复结束 =======================
                    
                    step_index_in_traj += 1

                    # --- 保留VERL原有数据 ---
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    data['episode_rewards_mean'] = episode_rewards_mean
                    data['episode_rewards_min'] = episode_rewards_min
                    data['episode_rewards_max'] = episode_rewards_max
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    data['episode_lengths_mean'] = episode_lengths_mean
                    data['episode_lengths_min'] = episode_lengths_min
                    data['episode_lengths_max'] = episode_lengths_max
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
                
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        self.file_logger.info(f"--- [gather_rollout_data] 聚合完成。总步骤: {len(effective_batch)} ---")
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        为物理设备环境优化的轨迹收集循环。
        """
        self.file_logger.info(f"--- [vanilla_multi_turn_loop] 开始 (Batch size: {len(gen_batch)}) ---")
        
        # ======================= ✅ 1. 准备并传递任务信息给 reset 方法 ✅ =======================
        tasks_for_this_batch = []
        try:
            # 确保 gen_batch.non_tensor_batch 及其键存在
            if (hasattr(gen_batch, 'non_tensor_batch') and gen_batch.non_tensor_batch and 
                'ground_truth_answer' in gen_batch.non_tensor_batch and 
                'prompt_index' in gen_batch.non_tensor_batch): # <-- ✅ 检查 prompt_index
                
                raw_prompts = gen_batch.non_tensor_batch['raw_prompt']
                tasks_list = [item[0]['content'] for item in raw_prompts]
                ground_truth_answers = gen_batch.non_tensor_batch['ground_truth_answer']
                
                # --- ✅ [STDB 修复] 提取 prompt_index 和 prompt_vector ---
                prompt_index_list = gen_batch.non_tensor_batch['prompt_index']
                prompt_vector_tensor = gen_batch.batch.get('prompt_vector')
                
                if prompt_vector_tensor is None:
                    self.file_logger.error("!!! [vanilla_multi_turn_loop] 'prompt_vector' is missing from gen_batch.batch! STDB will fail. !!!")
                
                for i in range(len(gen_batch)):
                    # 为每个环境创建一个包含任务描述和参考答案的字典
                    task_data = {
                        "task": tasks_list[i],
                        "ground_truth_answer": ground_truth_answers[i],
                        "prompt_index": prompt_index_list[i], # <-- ✅ [STDB 修复]
                    }
                    
                    if prompt_vector_tensor is not None:
                         task_data["prompt_vector"] = prompt_vector_tensor[i] # <-- ✅ [STDB 修复]
                    
                    tasks_for_this_batch.append(task_data)

                print("--- [rollout_loop.py] 已成功准备任务、答案、prompt_index 和 prompt_vector 用于环境重置。 ---")
                self.file_logger.info(f"已准备 {len(tasks_for_this_batch)} 个任务 (含 STDB 索引) 用于环境重置。")
            else:
                print("警告: 在 gen_batch 中未找到 'ground_truth_answer', 'raw_prompt' 或 'prompt_index'。环境将以无任务信息的方式重置。")
                self.file_logger.warning(f"在 gen_batch 中未找到 'ground_truth_answer', 'raw_prompt' 或 'prompt_index' (non_tensor_batch keys: {list(gen_batch.non_tensor_batch.keys())})。STDB 将失败。")
                tasks_for_this_batch = [{} for _ in range(len(gen_batch))]

        except (KeyError, IndexError, TypeError) as e:
            print(f"严重警告: 准备任务信息时出错: {e}")
            self.file_logger.error(f"准备任务信息时出错: {e}")
            # 如果出错，创建一个空列表以避免崩溃
            tasks_for_this_batch = [{} for _ in range(len(gen_batch))]
        
        # 这个修改假设 envs (EnvironmentManager) 的 reset 方法已被更新，
        # 可以接收 tasks 列表，并在内部处理日志初始化、prompt构建和底层环境的重置。
        # 并且, envs.py 内部会把 "prompt_index" 和 "prompt_vector" 存储起来，
        # 并在每次调用 record_step 时自动附加它们。
        obs, infos = envs.reset(tasks=tasks_for_this_batch)
        # ====================================================================================

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])

        # 对于物理环境，我们期望批次大小与环境数直接匹配。
        # 不再使用 gen_batch.repeat()，因为它适用于可无限实例化的模拟环境。
        assert len(gen_batch.batch) == lenght_obs, \
            f"对于物理设备环境，初始数据批次大小 ({len(gen_batch.batch)}) 必须与检测到的设备数 ({lenght_obs}) 完全匹配。" \
            "请检查你的 data.train_batch_size 和 data.val_batch_size 配置。"

        batch_size = len(gen_batch.batch['input_ids'])
        batch_output = None

        if self.config.env.rollout.n > 0:
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else:
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)

        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.int32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)

        # 记录 reset 这一步（作为 step 0）
        # 注意：这里的 'task' 依赖于 envs 对象正确地存储了任务信息
        if hasattr(envs, 'info_pool_managers') and hasattr(envs, 'tasks'):
            for i in range(len(infos)):
                if i in envs.info_pool_managers:
                    # ✅ [STDB 修复] 确保在 record_step 时传递了 reset 期间设置的 infos
                    # (假设 envs.reset 已经将 prompt_index/vector 放入了 infos[i])
                    step_data = {
                        "task": envs.tasks[i],
                        "thought": "Episode started.",
                        "parsed_action": "reset()",
                        "action_success": True,
                        "raw_obs_data": infos[i].get("raw_obs_data", {}),
                        "compressed_screenshot_bytes": infos[i].get("compressed_screenshot_bytes"),
                        "llm_prompt": "N/A",
                        "raw_llm_response": "N/A"
                    }
                    
                    # ✅ [STDB 修复] 
                    # 显式地将 reset 时应该知道的信息（来自 tasks_for_this_batch）
                    # 传递给 info_pool。这是必要的，因为 envs.py 我们无法修改。
                    if i < len(tasks_for_this_batch):
                        step_data["prompt_index"] = tasks_for_this_batch[i].get("prompt_index")
                        step_data["prompt_vector"] = tasks_for_this_batch[i].get("prompt_vector")
                    
                    envs.info_pool_managers[i].record_step(step_data)

        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            print("\n" + "="*50)
            print(f"--- 监控: 即将输入到 LLM 的完整 Prompt (Batch Item 0) (Step {_step+1}) ---")
            full_prompt_for_llm = self.tokenizer.decode(batch.batch['input_ids'][0], skip_special_tokens=False)
            # print(full_prompt_for_llm)
            print("="*50 + "\n")

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            # --- ✅ 新增: 从将要 pop 的 keys 中移除 'ground_truth_answer'，确保它保留在 batch 中 ---
            # --- ✅ [STDB 修复] 也不 pop 'prompt_index' 或 'prompt_vector' ---
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            
            # ======================= ✅ 日志：检查 pop 前的 gen_batch 键 =======================
            self.file_logger.info(f"[vanilla_multi_turn_loop] Step {_step+1}: 准备 pop batch。")
            self.file_logger.info(f"  - batch.batch keys (pop 前): {list(batch.batch.keys())}")
            self.file_logger.info(f"  - batch.non_tensor_batch keys (pop 前): {list(batch.non_tensor_batch.keys())}")
            # =========================================================================

            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info
            batch_output = actor_rollout_wg.generate_sequences(batch_input)
            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid
            batch = batch.union(batch_output)

            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)

            print("\n" + "*"*50)
            print(f"--- 监控: LLM 的完整回复 (Step {_step+1}) ---")
            for i, response in enumerate(text_actions):
                print(f"  [环境 {i}]: {response}")
            print("*"*50 + "\n")

            # ======================= ✅ [CCAPO] 计算并暂存 Token、置信度、对数概率 ✅ =======================
            try:
                # 1. 计算 Token
                input_token_counts = torch.sum(batch_input.batch["attention_mask"], dim=1)
                full_token_counts = torch.sum(batch.batch["attention_mask"], dim=1)
                output_token_counts = full_token_counts - input_token_counts

                # 2. 计算置信度
                log_probs = batch.batch['rollout_log_probs']
                # 注意：这里的 mask 需要精确对应 `log_probs` 张量的形状
                response_mask = batch.batch["attention_mask"][:, -log_probs.shape[1]:]
                
                # 屏蔽掉 padding token 的 log_probs
                masked_log_probs = log_probs * response_mask
                # 对每个样本的有效 log_probs 求和
                sum_of_log_probs = torch.sum(masked_log_probs, dim=1)
                # 计算每个样本的有效 token 数量
                num_of_tokens = torch.sum(response_mask, dim=1)
                # 避免除以零
                num_of_tokens[num_of_tokens == 0] = 1
                # 计算平均对数概率
                average_log_probs = sum_of_log_probs / num_of_tokens
                # 转换为平均概率（置信度）
                average_confidence = torch.exp(average_log_probs)

                # 3. 准备传递给 env_manager 的数据
                token_usage_list = []
                confidence_metrics_list = []
                log_probs_list = [] # <--- ✅ [CCAPO] 新增
                
                # ======================= ✅ [ 修复 G_Buffer Bug ] =======================
                # 我们需要传递 PPO 更新所需的所有张量
                tensors_for_env_list = []
                # =====================================================================
                
                for i in range(batch_size):
                    token_usage_list.append({
                        "prompt_tokens": input_token_counts[i].item(),
                        "completion_tokens": output_token_counts[i].item(),
                        "total_tokens": full_token_counts[i].item(),
                    })
                    confidence_metrics_list.append({
                        "average_log_probability": average_log_probs[i].item(),
                        "average_confidence": average_confidence[i].item(),
                    })
                    # <--- ✅ [CCAPO] 收集每个样本的 log_probs 张量 (Sec 6.1)
                    log_probs_list.append(log_probs[i])
                    
                    # ======================= ✅ [ 修复 G_Buffer Bug ] =======================
                    # 收集 PPO 更新所需的核心张量
                    # batch_input 包含 input_ids, attention_mask, position_ids
                    # batch 包含 responses (因为它是 batch_input.union(batch_output))
                    tensors_for_env_list.append({
                        "input_ids": batch.batch["input_ids"][i],
                        "attention_mask": batch.batch["attention_mask"][i],
                        "position_ids": batch.batch["position_ids"][i],
                        "responses": batch.batch["responses"][i],
                    })
                    # =====================================================================
                
                # 4. 暂存数据
                if hasattr(envs, "set_last_step_token_usage"):
                    envs.set_last_step_token_usage(token_usage_list)
                if hasattr(envs, "set_last_step_confidence"):
                    envs.set_last_step_confidence(confidence_metrics_list)
                # <--- ✅ [CCAPO] 暂存 log_probs
                if hasattr(envs, "set_last_step_log_probs"):
                    envs.set_last_step_log_probs(log_probs_list)
                
                # ======================= ✅ [ 修复 G_Buffer Bug ] =======================
                # 暂存 PPO 核心张量
                if hasattr(envs, "set_last_step_tensors"):
                    envs.set_last_step_tensors(tensors_for_env_list)
                # =====================================================================
                
                # ======================= ✅ [ 修复 STDB Bug ] =======================
                # 暂存 STDB 索引信息 (这是必须的，因为 envs.py 不会自动传递)
                if hasattr(envs, "set_last_step_stdb_info"):
                    stdb_info_list = []
                    
                    # --- ✅ [STDB 修复] 从当前的 'batch' 对象中提取索引/向量 ---
                    # (不再使用 'tasks_for_this_batch'，因为它只适用于 Step 0)
                    current_prompt_index_list = batch.non_tensor_batch.get('prompt_index', [None] * batch_size)
                    current_prompt_vector_tensor = batch.batch.get('prompt_vector', None)
                    
                    if current_prompt_vector_tensor is None:
                         self.file_logger.error(f"!!! [vanilla_multi_turn_loop] Step {_step+1}: 'prompt_vector' not found in current step's batch! STDB will fail. !!!")
                    if any(p is None for p in current_prompt_index_list):
                         self.file_logger.error(f"!!! [vanilla_multi_turn_loop] Step {_step+1}: 'prompt_index' not found in current step's batch! STDB will fail. !!!")

                    for i in range(batch_size):
                        stdb_info_list.append({
                            "prompt_index": current_prompt_index_list[i],
                            "prompt_vector": current_prompt_vector_tensor[i] if current_prompt_vector_tensor is not None else None
                        })
                    envs.set_last_step_stdb_info(stdb_info_list)
                # =====================================================================

            except Exception as e:
                import traceback
                print(f"!!!!!! [Rollout Step: {_step+1}] 计算 Token 和置信度时出错: {e} !!!!!!")
                self.file_logger.error(f"[vanilla_multi_turn_loop] Step {_step+1}: 计算 Token 和置信度时出错: {e}\n{traceback.format_exc()}")
                print(traceback.format_exc())
            # ==============================================================================

            next_obs, rewards, dones, infos = envs.step(text_actions)

            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)

            batch_list: list[dict] = to_list_of_dict(batch)
            for i in range(batch_size):
                # 只为 active (is_done=False) 的环境添加数据
                if active_masks[i]:
                    total_batch_list[i].append(batch_list[i])
                    total_infos[i].append(infos[i]) # <--- ✅ [CCAPO] 必须收集 infos

                    # ======================= ✅ [CCAPO V3] 关键修正：在此处检查并终结 ✅ =======================
                    # 如果这个环境 *刚刚* 终结 (dones[i] is True) 并且它仍在 info_pool 中
                    if dones[i] and hasattr(envs, 'info_pool_managers') and i in envs.info_pool_managers:
                        
                        task_completed = infos[i].get("won", False) # 'won' 来自 _step_device
                        status = "SUCCESS" if task_completed else "FAILURE"
                        summary_text = "Task finished by agent." if task_completed else "Task failed during execution."
                        
                        print(f"--- [rollout_loop.py] 正在为自然终结的环境 {i} (Status: {status}) 终结日志 ---")
                        self.file_logger.info(f"[vanilla_multi_turn_loop] Step {_step+1}: 环境 {i} (uid={traj_uid[i]}) 自然终结 (Status: {status})。")
                        
                        final_summary = envs.info_pool_managers[i].finalize_run(
                            status=status,
                            summary=summary_text,
                            run_start_time=envs.run_start_times[i],
                            task=envs.tasks[i],
                            task_completed=task_completed 
                        )
                        
                        # 将 final_summary 注入到我们刚刚 .append() 的 info 中
                        total_infos[i][-1]['final_summary'] = final_summary
                        
                        # 终结后从池中移除
                        envs.info_pool_managers.pop(i, None)
                    # ===================================================================================

            is_done = np.logical_or(is_done, dones)
            obs = next_obs
            if is_done.all():
                self.file_logger.info(f"[vanilla_multi_turn_loop] Step {_step+1}: 所有环境完成。")
                break
        
        if not is_done.all():
            self.file_logger.info(f"[vanilla_multi_turn_loop] 达到 Max Steps ({self.config.env.max_steps})。")

        # ======================= ✅ 2. 修复超时的终结逻辑 ✅ =======================
        # 确保因超时而结束的轨迹也被正确终结
        for i in range(batch_size):
            # 如果环境没有被标记为 'done' (is_done[i] is False)，说明它是因达到 max_steps 而超时的
            # 同时检查 info_pool_managers 是否存在且包含该环境的 manager
            if not is_done[i] and hasattr(envs, 'info_pool_managers') and i in envs.info_pool_managers:
                print(f"--- [rollout_loop.py] 正在为超时的环境 {i} 强制终结日志 ---")
                self.file_logger.warning(f"[vanilla_multi_turn_loop] 环境 {i} (uid={traj_uid[i]}) 超时。强制终结。")
                final_status = "TIMEOUT"
                summary_text = f"Task stopped due to reaching max steps ({self.config.env.max_steps})."

                # 检查 info_pool_managers[i] 是否仍然存在
                if i in envs.info_pool_managers:
                    # 使用新的 finalize_run 签名，task_completed 明确设置为 False
                    # ✅ [CCAPO] 捕获 finalize_run 的返回
                    final_summary = envs.info_pool_managers[i].finalize_run(
                        status=final_status,
                        summary=summary_text,
                        run_start_time=envs.run_start_times[i],
                        task=envs.tasks[i],
                        task_completed=False 
                    )
                    # ✅ [CCAPO] 将 final_summary 存入最后一步的 info
                    if total_infos[i]:
                        total_infos[i][-1]['final_summary'] = final_summary
                    
                    # 终结后从池中移除
                    envs.info_pool_managers.pop(i, None)
        # ========================================================================

        success: Dict[str, np.ndarray] = envs.success_evaluator(
            total_infos=total_infos,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
        )
        self.file_logger.info(f"--- [vanilla_multi_turn_loop] 结束 ---")
        return total_batch_list, total_infos, episode_rewards, episode_lengths, success, traj_uid # <--- ✅ [CCAPO] 返回 total_infos
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_infos (List[List[Dict]]): List of info dicts from env.step()
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_infos = [] # <--- ✅ [CCAPO] 收集 total_infos
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches
        
        self.file_logger.info(f"--- [dynamic_multi_turn_loop] 开始 (Target size: {self.config.data.train_batch_size * self.config.env.rollout.n}) ---")

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1
            
            self.file_logger.info(f"[dynamic_multi_turn_loop] 动态采样循环 第 {try_count}/{max_try_count} 次")

            batch_list, infos, episode_rewards, episode_lengths, success, traj_uid = self.vanilla_multi_turn_loop( # <--- ✅ [CCAPO] 接收 infos
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid = filter_group_data(batch_list=batch_list,
                                                                                                 episode_rewards=episode_rewards, 
                                                                                                 episode_lengths=episode_lengths, 
                                                                                                 success=success, 
                                                                                                 traj_uid=traj_uid, 
                                                                                                 config=self.config,
                                                                                                 last_try=(try_count == max_try_count),
                                                                                                 )
            
            total_batch_list += batch_list
            total_infos += infos # <--- ✅ [CCAPO] 聚合 infos
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        
        self.file_logger.info(f"--- [dynamic_multi_turn_loop] 结束 (收集到 {len(total_batch_list)} 条轨迹) ---")

        return total_batch_list, total_infos, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid # <--- ✅ [CCAPO] 返回 total_infos

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        # ======================= ✅ 日志：在 multi_turn_loop 记录 gen_batch 内容 =======================
        self.file_logger.info(f"--- [multi_turn_loop] 开始 (is_train: {is_train}) ---")
        self.file_logger.info(f"[multi_turn_loop] 传入的 gen_batch.batch keys: {list(gen_batch.batch.keys())}")
        self.file_logger.info(f"[multi_turn_loop] 传入的 gen_batch.non_tensor_batch keys: {list(gen_batch.non_tensor_batch.keys())}")
        if 'prompt_index' not in gen_batch.non_tensor_batch:
             self.file_logger.error("[multi_turn_loop] 关键错误：'prompt_index' 不在传入的 gen_batch.non_tensor_batch 中！")
        if 'prompt_vector' not in gen_batch.batch:
             self.file_logger.error("[multi_turn_loop] 关键错误：'prompt_vector' 不在传入的 gen_batch.batch 中！")
        # ===================================================================================

        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_infos, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling   
            total_batch_list, total_infos, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            gen_batch=gen_batch, # <--- ✅ [修复] 确保 gen_batch 被传递
            total_batch_list=total_batch_list,
            total_infos=total_infos, # <--- ✅ [CCAPO] 传递 total_infos
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
        )
        
        self.file_logger.info(f"--- [multi_turn_loop] 结束 ---")
        return gen_batch_output