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

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
import datetime
import re
import yaml
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory
from agent_system.environments.env_package.jarvis.envs import build_jarvis_envs
from agent_system.environments.env_package.jarvis.projection import jarvis_projection
# 1. 修改导入的 prompt，使用新的函数
from agent_system.environments.prompts.jarvis import get_jarvis_step_1_prompt, get_jarvis_intermediate_prompt, JARVIS_TEMPLATE, JARVIS_TEMPLATE_NO_HIS, SYSTEM_PROMPT
# from agent_system.memory import Trajectory
from .env_package.jarvis.info_pool import InfoPoolManager

import json

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
        
        # [CCAPO 修复] 获取环境数量的健壮逻辑
        num_envs = 1
        if hasattr(envs, "__len__"):
            num_envs = len(envs)
        elif hasattr(envs, "num_envs"):
            num_envs = envs.num_envs
        self.num_envs = num_envs 
        
        # 初始化数据缓存
        self._cached_token_usage = [{} for _ in range(num_envs)]
        self._cached_confidence = [{} for _ in range(num_envs)]
        self.trajectory_buffer = [[] for _ in range(num_envs)]
        self.trajectory_meta = [{} for _ in range(num_envs)]

        # [CCAPO 新增] 轨迹存储配置
        import os
        from datetime import datetime
        
        # 1. 批次计数器
        self.global_rollout_batch_idx = 0
        
        # 2. 实验根目录结构: experiments/experiment_name/runs_日期_时间
        self.exp_name = config.get('trainer', {}).get('experiment_name', 'default_exp')
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M%S")
        
        self.log_root = os.path.join(os.getcwd(), "experiments", self.exp_name, f"runs_{date_str}_{time_str}")
        
        try:
            os.makedirs(self.log_root, exist_ok=True)
            print(f"[EnvManager] Trajectories will be saved to: {self.log_root}")
        except Exception as e:
            print(f"[EnvManager] Error creating log dir: {e}")

    def reset(self, tasks=None, **kwargs):
        """
        Modified reset to handle CCAPO metadata initialization.
        """
        # [CCAPO] 每次 Reset 意味着一个新的 Rollout Batch 开始
        self.global_rollout_batch_idx += 1
        
        # 调用底层 reset
        text_obs, image_obs, infos = self.envs.reset()
        
        self.gamefile = parse_gamefile(infos)
        
        # initialize the history buffer
        self.memory.reset(batch_size=len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        
        # [CCAPO 修改] 初始化轨迹元数据
        import hashlib
        
        current_batch_size = len(infos)
        self.trajectory_meta = [{} for _ in range(current_batch_size)]
        
        if len(self.trajectory_buffer) != current_batch_size:
            self.trajectory_buffer = [[] for _ in range(current_batch_size)]
            self._cached_token_usage = [{} for _ in range(current_batch_size)]
            self._cached_confidence = [{} for _ in range(current_batch_size)]
        
        for i, info in enumerate(infos):
            seed = info.get('seed', kwargs.get('seed', 0))
            raw_gamefile = info.get("extra.gamefile", "")
            task_type = self._parse_task_type(raw_gamefile)
            
            current_task_desc = self.tasks[i] if len(self.tasks) > i else "unknown_task"
            group_key = f"{seed}_{current_task_desc}"
            group_id = hashlib.md5(group_key.encode()).hexdigest()[:16]
            
            self.trajectory_meta[i] = {
                "env_id": i,
                "seed": seed,
                "task_type": task_type,
                "group_id": group_id,
                "task_string": current_task_desc,
                "gamefile": raw_gamefile,
                # 记录它是哪个 batch 的，方便后续 debug
                "batch_idx": self.global_rollout_batch_idx 
            }
            
            self.trajectory_buffer[i] = []

        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos

    # [CCAPO 新增] 接收来自 rollout_loop 的 Token 统计
    def set_last_step_token_usage(self, usage_list: List[Dict]):
        """
        接收上一轮生成的 Token 消耗信息。
        usage_list: list of dict, e.g., [{'prompt_tokens': 10, ...}, ...]
        """
        if len(usage_list) != len(self._cached_token_usage):
            # 简单的容错，防止 batch size 不对齐
            return
        self._cached_token_usage = usage_list

    # [CCAPO 新增] 接收来自 rollout_loop 的置信度统计
    def set_last_step_confidence(self, metrics_list: List[Dict]):
        """
        接收上一轮生成的置信度信息。
        metrics_list: list of dict, e.g., [{'average_confidence': 0.9, ...}, ...]
        """
        if len(metrics_list) != len(self._cached_confidence):
            return
        self._cached_confidence = metrics_list
    
    def step(self, text_actions: List[str]):
        # 1. 动作映射与执行 (原有逻辑)
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        
        # 2. 内存更新 (原有逻辑)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # 3. 添加 action_valid 标记 (原有逻辑)
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        # ================= [CCAPO 修改开始] =================
        current_rewards = to_numpy(rewards)
        current_dones = to_numpy(dones)
        
        # [新增] 获取配置中的最大步数，默认 50 以防万一
        max_steps = self.config.env.get('max_steps', 50)
        
        for i in range(len(text_actions)):
            # A. 动作抽象
            raw_action = text_actions[i]
            abstract_action = self._abstract_action(raw_action)
            
            # B. 里程碑检测
            meta = self.trajectory_meta[i] if i < len(self.trajectory_meta) else {}
            milestones = self._detect_milestones(text_obs[i], meta.get('task_type', 'unknown'))
            
            # C. 获取 LLM 内部状态
            token_stats = self._cached_token_usage[i] if i < len(self._cached_token_usage) else {}
            conf_stats = self._cached_confidence[i] if i < len(self._cached_confidence) else {}
            
            # D. 组装单步数据
            step_record = {
                "step_idx": len(self.trajectory_buffer[i]),
                "action_raw": raw_action,
                "action_abstract": abstract_action,
                "observation": text_obs[i],
                "reward_env": float(current_rewards[i]),
                "is_valid": bool(valids[i]),
                "milestones": milestones,
                "llm_stats": {**token_stats, **conf_stats}
            }
            
            self.trajectory_buffer[i].append(step_record)
            
            # [关键修复]：检测是否超时截断 (Buffer长度 >= MaxSteps)
            # 注意：这里我们用 >= 是为了保险，防止之前 missed 掉
            is_truncated = len(self.trajectory_buffer[i]) >= max_steps
            
            # F. 如果 Episode 结束 (Done 或 Truncated)
            if current_dones[i] or is_truncated:
                is_success = bool(infos[i].get('won', False))
                
                full_trajectory_log = {
                    "meta": self.trajectory_meta[i],
                    "metrics": {
                        "total_steps": len(self.trajectory_buffer[i]),
                        "is_success": is_success,
                        "final_env_reward": float(current_rewards[i]),
                        # [新增] 标记是否为超时截断，方便后续分析
                        "is_truncated": is_truncated and not current_dones[i]
                    },
                    "steps": self.trajectory_buffer[i]
                }
                
                # 1. 挂载到 info (供 RewardManager 使用)
                infos[i]['ccapo_trajectory'] = full_trajectory_log
                
                # 2. [关键] 立即保存到本地磁盘 (Worker 本地)
                self._save_trajectory_to_disk(full_trajectory_log)
                
                # 3. 如果是 Truncated 但还没 Done，为了保证 RewardManager 能统计到 fail，
                #    这里不需要手动把 current_dones 改为 True (因为 verl 外层循环会处理)，
                #    只要确保 ccapo_trajectory 挂载到了 infos[i] 上即可。

        # ================= [CCAPO 修改结束] =================

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = current_rewards
        dones = current_dones

        return next_observations, rewards, dones, infos

    # [CCAPO 新增] 私有辅助方法
    def _save_trajectory_to_disk(self, traj_data):
        import os
        import json
        import hashlib
        
        try:
            meta = traj_data.get('meta', {})
            task_type = meta.get('task_type', 'unknown_task')
            group_id = str(meta.get('group_id', 'unknown_group'))
            
            # [CCAPO 修改] 目录结构: log_root / batch_X / task_type / group_id
            # 这样可以清晰地看到每一轮训练产生了什么
            batch_folder = f"batch_{self.global_rollout_batch_idx}"
            save_dir = os.path.join(self.log_root, batch_folder, task_type, group_id)
            
            os.makedirs(save_dir, exist_ok=True)
            
            # 文件名: STATUS_steps_uuid.json
            is_success = "SUCCESS" if traj_data['metrics']['is_success'] else "FAIL"
            steps = traj_data['metrics']['total_steps']
            traj_hash = hashlib.md5(json.dumps(traj_data['steps'], default=str).encode()).hexdigest()[:8]
            
            filename = f"{is_success}_s{steps}_{traj_hash}.json"
            filepath = os.path.join(save_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(traj_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"[EnvManager] Failed to save trajectory log: {e}")
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
            
    def _parse_task_type(self, gamefile_path):
        """
        从 ALFWorld 的 gamefile 路径中解析任务类型。
        例如: 'pick_and_place_simple-...' -> 'pick_and_place'
        """
        if not gamefile_path:
            return "unknown"
        
        # 常见任务类型映射
        task_keywords = [
            "pick_and_place",
            "look_at_obj",
            "pick_clean_then_place",
            "pick_heat_then_place",
            "pick_cool_then_place",
            "pick_two_obj",
            "clean", # fallback
            "heat",  # fallback
            "cool"   # fallback
        ]
        
        lower_path = gamefile_path.lower()
        for kw in task_keywords:
            if kw in lower_path:
                # 规范化名称
                if "clean" in kw: return "clean_object"
                if "heat" in kw: return "heat_object"
                if "cool" in kw: return "cool_object"
                if "look" in kw: return "examine_object"
                if "two" in kw: return "pick_two_object"
                return "pick_and_place" # default logic
                
        return "general_task"

    def _abstract_action(self, action_str):
        """
        Logic Stream 核心：将动作抽象化。
        规则：移除物体具体的数字 ID，保留动词和物体类别。
        Example: "put apple 1 in fridge 2" -> "put apple in fridge"
        """
        if not action_str:
            return ""
        
        import re
        # 1. 转小写并去首尾空格
        text = action_str.lower().strip()
        # 2. 移除 " 1", " 23" 这种数字后缀 (注意保留前面的空格，防止把 apple123 变成 appl)
        # 正则含义：匹配 空格+数字，替换为空
        abstracted = re.sub(r'\s\d+', '', text)
        return abstracted

    def _detect_milestones(self, obs_text, task_type):
        """
        Milestone Mining 核心：基于观察文本检测关键事件。
        返回触发的事件列表。
        """
        milestones = []
        obs_lower = obs_text.lower()
        
        # === 通用事件 ===
        # 1. 打开容器/门
        if "you open the" in obs_lower:
            milestones.append("event_opened_container")
        
        # 2. 拿到物体
        if "you pick up the" in obs_lower or "you take the" in obs_lower:
            milestones.append("event_picked_up_item")
            
        # 3. 放置物体 (ALFWorld 通常会说 You put the X in/on the Y)
        if "you put the" in obs_lower:
            milestones.append("event_placed_item")

        # === 任务特定事件 (Task Specific) ===
        # 4. 清洁任务
        if "clean" in task_type and ("clean" in obs_lower or "rinsing" in obs_lower):
            # 只有在拿起物体并在水槽操作时才会触发 clean
            milestones.append("event_object_cleaned")
            
        # 5. 加热任务
        if "heat" in task_type and ("heat" in obs_lower or "hot" in obs_lower):
            milestones.append("event_object_heated")
            
        # 6. 冷却任务
        if "cool" in task_type and ("cool" in obs_lower or "chilled" in obs_lower or "cold" in obs_lower):
            milestones.append("event_object_cooled")

        return milestones
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    if "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': 'eval_in_distribution', # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs)
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)