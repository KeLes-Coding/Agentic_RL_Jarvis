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

import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray
import sys # Added for flush

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

@ray.remote(num_cpus=0.2)
class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env):
        self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        self.env.seed(seed)
        # 🔥 [Cache] 用来存储原始的游戏列表，防止被我们修改后找不回去了
        self._original_game_files = None
        self.worker_id = seed # Just for debug ID
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos
    
    def reset(self, game_file=None):
        """
        Reset the environment.
        🔥 [Fix V4.0] 强制指定游戏文件，并重置内部索引，确保环境真的加载该文件。
        """
        # 1. 打印接收到的请求 (Debug)
        print(f"[ALF-WORKER-{self.worker_id}] Reset request. Target: '{game_file}'", flush=True)

        if game_file:
            # A. 首次运行时备份原始游戏列表
            if self._original_game_files is None:
                # 递归查找含有 game_files 的层级
                env_ptr = self.env
                while hasattr(env_ptr, 'env') and not hasattr(env_ptr, 'game_files'):
                    env_ptr = env_ptr.env
                
                if hasattr(env_ptr, 'game_files'):
                    self._original_game_files = env_ptr.game_files
                else:
                    self._original_game_files = []
                    print(f"[ALF-WORKER-{self.worker_id}] ⚠️ Error: Cannot find 'game_files' in env hierarchy.", flush=True)

            # B. 在原始列表中查找目标文件
            if self._original_game_files:
                target_file = None
                search_key = str(game_file).strip()
                
                # 模糊匹配
                for f in self._original_game_files:
                    if search_key in f:
                        target_file = f
                        break
                
                if target_file:
                    # C. 核心 HACK: 穿透所有 Wrapper，修改底层环境的列表和索引
                    found_layer = False
                    curr = self.env
                    # 尝试最多 5 层穿透
                    for _ in range(5):
                        if hasattr(curr, 'game_files'):
                            # 1. 锁定列表
                            curr.game_files = [target_file]
                            # 2. 锁定数量
                            if hasattr(curr, 'num_games'): 
                                curr.num_games = 1
                            # 3. 🔥 [CRITICAL] 重置索引，防止越界或错位
                            if hasattr(curr, 'game_file_index'): 
                                curr.game_file_index = 0
                            # 某些变体可能叫 file_index
                            if hasattr(curr, 'file_index'): 
                                curr.file_index = 0
                            
                            found_layer = True
                            # 继续向下查找，防止有多个层级都缓存了列表
                        
                        if hasattr(curr, 'env'):
                            curr = curr.env
                        else:
                            break
                    
                    if found_layer:
                        print(f"[ALF-WORKER-{self.worker_id}] ✅ Hack applied. Locked to: ...{target_file[-40:]}", flush=True)
                    else:
                        print(f"[ALF-WORKER-{self.worker_id}] ❌ Failed to apply hack: 'game_files' attr not found.", flush=True)

                else:
                    print(f"[ALF-WORKER-{self.worker_id}] ❌ Warning: Requested game '{search_key}' NOT FOUND in env list.", flush=True)

        # 2. 执行 Reset
        obs, infos = self.env.reset()
        
        # 3. 结果验证日志
        loaded_file = infos.get('extra.gamefile', 'Unknown')
        match_status = "MATCH" if (game_file and str(game_file) in str(loaded_file)) else "MISMATCH"
        print(f"[ALF-WORKER-{self.worker_id}] Post-Reset Status: {match_status} | Loaded: ...{str(loaded_file)[-40:]}", flush=True)
        
        infos['observation_text'] = obs
        return obs, infos
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed=0, env_num=1, group_n=1, is_train=True, env_kwargs={}):
        super().__init__()
        
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)

        # 🔥 [Fix V2] 强力覆盖策略
        # 1. 打印传入的所有参数，确认 max_steps 是否存在
        print(f"[ALFWorld Wrapper] env_kwargs keys: {list(env_kwargs.keys())}", flush=True)

        target_steps = 50 # 默认保底值
        
        # 尝试从 kwargs 获取并转为 int
        if 'max_steps' in env_kwargs:
            target_steps = int(env_kwargs['max_steps'])
            print(f"[ALFWorld Wrapper] 🚀 Found max_steps in kwargs: {target_steps}", flush=True)
        else:
            print(f"[ALFWorld Wrapper] ⚠️ max_steps NOT found in kwargs, using default: {target_steps}", flush=True)

        # 2. 覆盖 config 中的每一个角落
        if 'general' not in config: config['general'] = {}
        config['general']['max_steps'] = target_steps
        
        if 'rl' in config:
            if 'training' not in config['rl']: config['rl']['training'] = {}
            config['rl']['training']['max_nb_steps_per_episode'] = target_steps
            
        if 'dagger' in config:
            if 'training' not in config['dagger']: config['dagger']['training'] = {}
            config['dagger']['training']['max_nb_steps_per_episode'] = target_steps

        # 3. 强制修改 env 部分（部分旧版本会读这里）
        if 'env' not in config: config['env'] = {}
        config['env']['max_steps'] = target_steps
        
        print(f"[ALFWorld Wrapper] Config updated. General: {config['general'].get('max_steps')}, RL: {config.get('rl',{}).get('training',{}).get('max_nb_steps_per_episode')}", flush=True)
        # -------------------------------------------------------------

        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        self.workers = []
        for i in range(self.num_processes):
            worker = AlfworldWorker.remote(config, seed + (i // self.group_n), base_env)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self, tasks=None, **kwargs):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            # 提取目标游戏文件
            target_game = None
            if tasks is not None and i < len(tasks):
                target_game = tasks[i].get('prompt_index')
            
            # 传递给 worker (带参数)
            future = worker.reset.remote(game_file=target_game)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        return self.prev_admissible_commands

    def close(self):
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, is_train, env_kwargs)