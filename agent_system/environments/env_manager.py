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
    
    def reset(self):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

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


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
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
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs
    
# 2. 覆盖更新 JarvisEnvironmentManager 类
def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)

class JarvisEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
        self.num_envs = self.envs.num_envs
        self.tasks: List[str] = []
        self.ground_truth_answers: List[str] = []
        
        # --- ✅ 修改: LLM config 现在由底层的 envs.py 读取和管理 ---
        # self.llm_config = self._load_llm_config(...) # 这一行不再需要

        # --- 修改：为整个训练运行创建一个唯一的顶级日志目录 ---
        log_root_dir = config.env.jarvis.get("log_dir", "trajectory_logs")
        run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join(log_root_dir, f"training_run_{run_timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"所有轨迹日志将保存在唯一的训练目录中: {self.log_dir}")
        
        self.info_pool_managers: Dict[int, InfoPoolManager] = {}
        self.run_start_times: Dict[int, datetime.datetime] = {}
        self.last_prompts: List[str] = [""] * self.num_envs
        self.active_batch_size = 0
        
        # ======================= ✅ 添加用于暂存 Token 和置信度信息的变量 ✅ =======================
        self.last_token_usage: List[dict] = None
        self.last_confidence: List[dict] = None
        self.last_log_probs: List[torch.Tensor] = None # <--- ✅ [CCAPO] 新增
        # ======================= ✅ [ 修复 G_Buffer Bug ] =======================
        self.last_tensors: List[dict] = None # <--- ✅ 新增：暂存 PPO 张量
        # ===================================================================================

    # --- 🗑️ 移除: 不再需要此方法，配置由底层 envs.py 管理 ---
    # def _load_llm_config(...): ...

    # --- 🗑️ 移除: set_tasks 的逻辑将被合并到 reset 方法中 ---
    # def set_tasks(...): ...

    # ======================= ✅ 添加用于接收和暂存数据的新方法 ✅ =======================
    def set_last_step_token_usage(self, token_usage_list: List[dict]):
        """从外部（rollout_loop）接收并暂存当前步骤的 token 使用情况。"""
        self.last_token_usage = token_usage_list

    def set_last_step_confidence(self, confidence_list: List[dict]):
        """从外部（rollout_loop）接收并暂存当前步骤的置信度信息。"""
        self.last_confidence = confidence_list

    def set_last_step_log_probs(self, log_probs_list: List[torch.Tensor]): # <--- ✅ [CCAPO] 新增
        """从外部（rollout_loop）接收并暂存当前步骤的对数概率。"""
        self.last_log_probs = log_probs_list

    # ================================================================================

    # ======================= ✅ [ 修复 G_Buffer Bug ] =======================
    def set_last_step_tensors(self, tensors_list: List[dict]): # <--- ✅ 新增
        """从外部 (rollout_loop) 接收并暂存 G_Buffer 所需的核心张量。"""
        self.last_tensors = tensors_list
    # ================================================================================

    def _initialize_loggers_for_new_run(self):
        """为当前批次的所有环境初始化或重置日志记录器。"""
        print("--- [env_manager.py] 正在为新批次初始化日志记录器 ---")
        self.info_pool_managers.clear()
        
        for i in range(self.active_batch_size):
            task = self.tasks[i]
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            safe_task_name = re.sub(r'[^\w\-_\. ]', '_', task)[:50]
            run_dir_name = f"{timestamp}_env{i}_{safe_task_name}"
            full_path = os.path.join(self.log_dir, run_dir_name)
            
            self.info_pool_managers[i] = InfoPoolManager(full_path)
            self.run_start_times[i] = datetime.datetime.now(datetime.timezone.utc)
            print(f"  [环境 {i}] 的日志目录: {full_path}")
        print("-------------------------------------------------")

    # [新增] 显式暴露后台重置接口，供 rollout_loop 调用
    def start_background_reset(self):
        """
        触发底层环境的后台重置。
        这允许在主进程进行训练 (Training) 时，后台线程并行执行耗时的 ADB 重置操作。
        """
        # self.envs 是 JarvisMultiDeviceEnv 的实例
        if hasattr(self.envs, "start_background_reset"):
            print("--- [EnvManager] ⚡️ 触发底层 JarvisMultiDeviceEnv 的后台重置 (Async Reset) ---")
            self.envs.start_background_reset()
        else:
            print(f"--- [EnvManager] 警告: 底层环境 {type(self.envs)} 没有 start_background_reset 方法 ---")

    # ======================= ✅ 1. 修改 reset 方法以接收 tasks 参数 ✅ =======================
    def reset(self, tasks: List[Dict] = None):
        """
        重置所有环境，并使用新的任务信息进行初始化。
        """
        if tasks:
            self.tasks = [t.get('task', 'No task provided') for t in tasks]
            self.ground_truth_answers = [t.get('ground_truth_answer', '') for t in tasks]
            self.active_batch_size = len(tasks)
            print(f"--- [env_manager.py] 接收到 {self.active_batch_size} 个任务并准备重置环境 ---")
            
            # 初始化日志记录器
            self._initialize_loggers_for_new_run()
        else:
            # 如果没有提供任务（例如，在某些初始化阶段），则进行默认重置
            self.tasks = ["Initializing..."] * self.num_envs
            self.ground_truth_answers = [""] * self.num_envs
            self.active_batch_size = self.num_envs

        # 将 tasks 列表传递给底层的 JarvisMultiDeviceEnv
        raw_obs, infos = self.envs.reset(tasks=tasks)
        self.memory.reset(batch_size=self.num_envs)
        
        batched_images = raw_obs['image']
        full_text_obs = self.build_text_obs(raw_obs['text'], self.tasks, init=True)

        return {'text': full_text_obs, 'image': batched_images, 'anchor': raw_obs['text']}, infos
    # =====================================================================================

    def step(self, text_actions: List[str]):
        parsed_actions, valids, thoughts = self.projection_f(text_actions)
        next_raw_obs, rewards, dones, infos = self.envs.step(parsed_actions)

        # 只处理活动环境的数据
        for i in range(self.active_batch_size):
            
            # --- ✅ [CCAPO] 1. 解析 action_type (Sec 5.1.2) ---
            action_type = "unknown"
            if "(" in parsed_actions[i]:
                action_type = parsed_actions[i].split("(", 1)[0].strip()
            
            # --- ✅ [CCAPO] 2. 准备要记录和传递的完整 step_data ---
            if i in self.info_pool_managers:
                step_data = {
                    "task": self.tasks[i],
                    "thought": thoughts[i],
                    "parsed_action": parsed_actions[i],
                    "action_type": action_type, # <--- ✅ [CCAPO] 新增
                    "action_success": infos[i].get("action_success", False),
                    "raw_obs_data": infos[i].get("raw_obs_data", {}),
                    "compressed_screenshot_bytes": infos[i].get("compressed_screenshot_bytes"),
                    "llm_prompt": self.last_prompts[i],
                    "raw_llm_response": text_actions[i],
                    "log_dir_path": self.info_pool_managers[i].log_dir # <--- ✅ [CCAPO] 新增
                }
                
                if self.last_token_usage and i < len(self.last_token_usage):
                    step_data["token_usage"] = self.last_token_usage[i]
                
                if self.last_confidence and i < len(self.last_confidence):
                    step_data["confidence_metrics"] = self.last_confidence[i]
                
                if self.last_log_probs and i < len(self.last_log_probs): # <--- ✅ [CCAPO] 新增
                    step_data["rollout_log_probs"] = self.last_log_probs[i]

                # ======================= ✅ [ 修复 G_Buffer Bug ] =======================
                # 将暂存的 PPO 张量 (来自 set_last_step_tensors) 复制到 step_data
                if self.last_tensors and i < len(self.last_tensors):
                    step_data.update(self.last_tensors[i])
                # =====================================================================

                self.info_pool_managers[i].record_step(step_data)

                # --- ✅ [CCAPO] 3. 将所有微观数据复制到 infos 字典中 ---
                #    (以便 gather_rollout_data 稍后可以访问它们)
                infos[i]['thought'] = step_data['thought']
                infos[i]['parsed_action'] = step_data['parsed_action']
                infos[i]['action_type'] = step_data['action_type']
                # 'action_success' 已经由 self.envs.step() 放入 infos[i]
                infos[i]['token_usage'] = step_data.get('token_usage', {})
                infos[i]['confidence_metrics'] = step_data.get('confidence_metrics', {})
                infos[i]['log_dir_path'] = step_data['log_dir_path'] # <--- ✅ [CCAPO] 新增
                # rollout_log_probs 不需要放入 infos,因为它已经在 batch_list 中

                # ======================= ✅ 2. 修改终结逻辑以匹配新签名 ✅ =======================
                if dones[i]:
                    # 从 infos 中获取底册环境返回的 task_completed 状态
                    task_completed = infos[i].get("task_completed", False)
                    final_status = "SUCCESS" if task_completed else "FAILURE"
                    
                    summary_text = "Task finished."
                    # 尝试从原始动作中解析 summary
                    if parsed_actions[i].startswith("finish"):
                        match = re.search(r"summary=['\"](.*?)['\"]", parsed_actions[i], re.DOTALL)
                        if match: 
                            summary_text = match.group(1).strip()
                        else: # 如果正则失败，使用备用方案
                            start_index = parsed_actions[i].find('(')
                            end_index = parsed_actions[i].rfind(')')
                            if start_index != -1 and end_index > start_index:
                                content = parsed_actions[i][start_index + 1:end_index].strip()
                                if content.lower().startswith("summary="):
                                    summary_text = content[len("summary="):].strip().strip("'\" ")
                                else:
                                    summary_text = content.strip("'\" ")

                    # 使用 info_pool.py 中新的 finalize_run 签名
                    final_summary = self.info_pool_managers[i].finalize_run( # <--- ✅ [CCAPO] 捕获返回
                        status=final_status,
                        summary=summary_text,
                        run_start_time=self.run_start_times[i],
                        task=self.tasks[i],
                        task_completed=task_completed #直接传递评估结果
                    )
                    
                    infos[i]['final_summary'] = final_summary # <--- ✅ [CCAPO] 存入 info
                    
                    # 清理完成的任务，避免重复终结
                    self.info_pool_managers.pop(i, None)
                # =============================================================================

        batched_images = next_raw_obs['image']
        self.memory.store({'thought': thoughts, 'action': parsed_actions})
        full_text_obs = self.build_text_obs(next_raw_obs['text'], self.tasks)
        
        self.last_prompts = full_text_obs

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': batched_images, 'anchor': next_raw_obs['text']}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, text_obs: List[str], tasks: List[str], init: bool = False) -> List[str]:
        postprocess_text_obs = []
        num_obs = len(text_obs)
        for i in range(num_obs):
            current_task = tasks[i] if i < len(tasks) else "Task not assigned yet."
            if init or len(self.memory[i]) == 0:
                user_content = get_jarvis_step_1_prompt(
                    task=current_task,
                    simplified_ui=text_obs[i]
                )
            else:
                last_record = self.memory[i][-1]
                prev_thought = last_record.get('thought', 'N/A')
                prev_action = last_record.get('action', 'N/A')
                user_content = get_jarvis_intermediate_prompt(
                    task=current_task,
                    prev_thought=prev_thought,
                    prev_action=prev_action,
                    simplified_ui=text_obs[i]
                )
            final_obs = f"{SYSTEM_PROMPT}\n\n{user_content}"
            postprocess_text_obs.append(final_obs)
        return postprocess_text_obs

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    if "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
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
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "jarvis" in config.env.env_name.lower():
        _envs = build_jarvis_envs(
            jarvis_config_path=config.env.jarvis.jarvis_config_path,
            max_steps=config.env.jarvis.max_steps_per_episode
        )
        _val_envs = build_jarvis_envs(
            jarvis_config_path=config.env.jarvis.jarvis_config_path,
            max_steps=config.env.jarvis.max_steps_per_episode
        )

        projection_f = partial(jarvis_projection)
        envs = JarvisEnvironmentManager(_envs, projection_f, config)
        val_envs = JarvisEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)