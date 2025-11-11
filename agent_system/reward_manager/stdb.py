# agent_system/reward_manager/stdb.py

import os
import json
import torch
import heapq
import logging
import collections
from typing import List, Dict, Any, Tuple, Set
from sentence_transformers import util

# --- 1. 标准日志器 (用于 STDOUT / 主日志) ---
logger = logging.getLogger(__name__)

# --- 2. 专用文件日志器 (用于 logger/STDB/stdb_operations.log) ---
stdb_file_logger = logging.getLogger("STDB_FILE")
stdb_file_logger.setLevel(logging.INFO) # 捕获 INFO 及以上级别
stdb_file_logger.propagate = False      # 防止重复记录到 root logger

# 仅在日志器没有处理器时才添加，以防止重复
if not stdb_file_logger.handlers:
    try:
        log_dir = "logger/STDB"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "stdb_operations.log")
        
        # 创建文件处理器 (追加模式)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - [STDB_FILE] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # 添加处理器
        stdb_file_logger.addHandler(file_handler)
        stdb_file_logger.info("--- STDB 专用文件日志器已初始化 ---")
        
    except Exception as e:
        logger.error(f"[STDB] 无法创建专用文件日志器: {e}")
# --- 日志设置结束 ---


# 定义数据结构
# (轨迹宏观奖励, 轨迹日志的磁盘路径)
StdbEntry = Tuple[float, str]
# { prompt_index: [ (R_tau, log_dir_path), ... ] }
StdbDatabase = Dict[int, List[StdbEntry]]
# { prompt_index: torch.Tensor }
StdbVectorMap = Dict[int, torch.Tensor]


class SuccessTrajectoryDatabase:
    def __init__(self, save_path: str, top_k: int = 8):
        """
        初始化成功轨迹数据库 (STDB)。

        Args:
            save_path: STDB 索引文件的保存路径 (例如 'checkpoints/stdb_index.pt')。
            top_k: 每个 prompt 存储的 Top-K 轨迹数。
        """
        self.save_path = save_path
        self.top_k = top_k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- 核心数据结构 ---
        
        # 1. 数据库: { prompt_index -> min_heap[(R_tau, log_dir_path)] }
        self.db: StdbDatabase = collections.defaultdict(list)
        
        # 2. 向量映射: { prompt_index -> prompt_vector }
        self.prompt_vectors: StdbVectorMap = {}
        
        # 3. 向量索引 (用于快速余弦相似度搜索)
        self.known_vectors: torch.Tensor = None  # 在 CPU 上的 N x D 张量
        self.known_indices: List[int] = []      # 长度为 N
        
        # 4. 辅助 set，用于防止重复添加同一条轨迹
        self.all_log_paths: Set[str] = set()
        
        # --- ✅ [V3] 分配专用日志器 ---
        self.file_logger = stdb_file_logger

        # --- 初始化 ---
        self.file_logger.info(f"--- 初始化 STDB 实例 ---")
        self.file_logger.info(f"Save path: {self.save_path}")
        self.file_logger.info(f"Top-K: {self.top_k}")
        self.load() # 尝试从磁盘加载

    def _rebuild_vector_index(self):
        """
        根据 self.prompt_vectors 重建用于快速搜索的 self.known_vectors。
        """
        if not self.prompt_vectors:
            self.known_vectors = None
            self.known_indices = []
            return

        # 确保所有向量都在 CPU 上且类型一致
        self.known_indices = list(self.prompt_vectors.keys())
        vectors_on_cpu = [self.prompt_vectors[idx].cpu().float() for idx in self.known_indices]
        
        self.known_vectors = torch.stack(vectors_on_cpu)
        logger.info(f"[STDB] 向量索引已重建。索引了 {len(self.known_indices)} 个 prompt。")
        self.file_logger.info(f"向量索引已重建。索引了 {len(self.known_indices)} 个 prompt。")


    def save(self):
        """将 STDB 索引保存到磁盘。"""
        try:
            # 我们只保存 db 和 prompt_vectors。known_vectors 会在 load 时重建。
            # 将 defaultdict 转换为常规 dict 以便安全保存
            data_to_save = (dict(self.db), self.prompt_vectors, self.all_log_paths)
            torch.save(data_to_save, self.save_path)
            logger.info(f"[STDB] 成功将 STDB 索引保存到 {self.save_path}")
            self.file_logger.info(f"索引已保存。总 prompts: {len(self.db)}, 总 paths: {len(self.all_log_paths)}。")
        except Exception as e:
            logger.error(f"[STDB] 无法保存 STDB 索引: {e}")
            self.file_logger.error(f"无法保存 STDB 索引: {e}")

    def load(self):
        """从磁盘加载 STDB 索引。"""
        if not os.path.exists(self.save_path):
            logger.warning(f"[STDB] 未找到 STDB 索引文件: {self.save_path}. 将创建一个新的数据库。")
            self.file_logger.warning(f"未找到索引文件 {self.save_path}。将创建新数据库。")
            return

        try:
            db_dict, self.prompt_vectors, self.all_log_paths = torch.load(self.save_path)
            self.db = collections.defaultdict(list, db_dict)
            
            # 确保堆结构被正确加载
            for k, v in self.db.items():
                heapq.heapify(v) # 重新确保最小堆属性
                
            self._rebuild_vector_index()
            logger.info(f"[STDB] 成功从 {self.save_path} 加载 STDB 索引。")
            self.file_logger.info(f"成功从 {self.save_path} 加载索引。DB中已有 {len(self.db)} 个 prompts。")
        except Exception as e:
            logger.error(f"[STDB] 无法加载 STDB 索引: {e}. 将从空数据库开始。")
            self.file_logger.error(f"无法加载 STDB 索引: {e}. 将从空数据库开始。")
            self.db = collections.defaultdict(list)
            self.prompt_vectors = {}
            self.all_log_paths = set()

    def add_online_trajectories(self, online_trajectories: Dict[str, List[Dict[str, Any]]]):
        """
        在计算完 R_tau 和 R_core 后，尝试将新的在线轨迹添加到 STDB。
        
        Args:
            online_trajectories: 字典 {traj_uid -> list_of_steps}
                                 每个 step 必须已包含 'R_tau', 'R_core', 
                                 'prompt_index', 'prompt_vector', 和 'log_dir_path'。
        """
        added_count = 0
        rebuild_needed = False
        
        self.file_logger.info(f"--- 开始 STDB 更新 (收到 {len(online_trajectories)} 条轨迹) ---")
        
        for traj_uid, steps in online_trajectories.items():
            if not steps:
                continue
            
            first_step = steps[0]
            
            # --- ✅ [CCAPO V2] 修正：提取 R_core ---
            # R_tau 和 R_core 是由 ccapo_algos 计算的，必须存在
            R_tau = first_step.get('R_tau')
            R_core = first_step.get('R_core')
            
            # 提取其他关键信息
            prompt_index = first_step.get('prompt_index')
            log_dir_path = first_step.get('log_dir_path')
            prompt_vector = first_step.get('prompt_vector')

            if R_tau is None or R_core is None or prompt_index is None or log_dir_path is None or prompt_vector is None:
                logger.warning(f"[STDB] 跳过 traj_uid {traj_uid}，缺少关键信息。")
                logger.debug(f"(R_tau: {R_tau is not None}, R_core: {R_core is not None}, prompt_index: {prompt_index is not None}, log_dir_path: {log_dir_path is not None}, prompt_vector: {prompt_vector is not None})")
                self.file_logger.warning(f"跳过 traj {traj_uid}: 缺少关键信息 (R_core: {R_core is not None}, R_tau: {R_tau is not None}, ...)")
                continue
            
            # --- ✅ [CCAPO V2] 修正：使用 R_core == 1.0 作为唯一的成功标准 ---
            # 1. 检查是否为“有效成功”的轨迹
            if R_core != 1.0:
                # 这将自动过滤掉失败 (-1.0) 和作弊 (0.0) 的轨迹
                self.file_logger.debug(f"跳过 traj {traj_uid}: 非有效成功 (R_core = {R_core:.1f})。")
                continue
                
            # 2. 检查轨迹是否已被添加
            if log_dir_path in self.all_log_paths:
                self.file_logger.debug(f"跳过 traj {traj_uid}: 路径 {log_dir_path} 已在 STDB 中。")
                continue

            # 3. 更新 prompt 向量表
            if prompt_index not in self.prompt_vectors:
                self.prompt_vectors[prompt_index] = prompt_vector.cpu() # 存储在 CPU
                rebuild_needed = True
                self.file_logger.info(f"为 prompt {prompt_index} 添加了新的 prompt_vector。")
                
            # 4. 更新 Top-K 堆
            heap = self.db[prompt_index]
            entry: StdbEntry = (R_tau, log_dir_path)

            if len(heap) < self.top_k:
                heapq.heappush(heap, entry)
                self.all_log_paths.add(log_dir_path)
                added_count += 1
                self.file_logger.info(f"[ADD]     Prompt {prompt_index}: 添加 traj {traj_uid} (R_tau: {R_tau:.4f}). 堆大小: {len(heap)}/{self.top_k}")
            elif R_tau > heap[0][0]: # 如果比堆中的最小值要大
                evicted_entry = heapq.heappushpop(heap, entry)
                self.all_log_paths.remove(evicted_entry[1]) # 移除被挤出的路径
                self.all_log_paths.add(log_dir_path)
                added_count += 1
                self.file_logger.info(f"[EVICT]   Prompt {prompt_index}: 添加 traj {traj_uid} (R_tau: {R_tau:.4f}). 替换 {evicted_entry[1]} (R_tau: {evicted_entry[0]:.4f})")
            else:
                self.file_logger.debug(f"跳过 traj {traj_uid} (R_tau: {R_tau:.4f}): 分数未进入 Prompt {prompt_index} 的 Top-K。 (最低分: {heap[0][0]:.4f})")
        
        if added_count > 0:
            logger.info(f"[STDB] 添加了 {added_count} 条新的有效成功轨迹 (R_core=1.0)。")
        
        self.file_logger.info(f"--- STDB 更新完成 (新增: {added_count}，需重建索引: {rebuild_needed}) ---")
        
        if rebuild_needed:
            self._rebuild_vector_index()
            
        if added_count > 0 or rebuild_needed:
            self.save() # 每次更新后自动保存

    def get_buffer_trajectories(self, online_batch_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        获取 G_buffer (Sec 3.2):
        根据在线批次中的 prompts，提取 STDB 中的锚点轨迹。
        实现直接匹配和相似度回退。
        """
        unique_prompts = {} # {prompt_index -> prompt_vector}
        for step in online_batch_steps:
            if 'prompt_index' in step and step['prompt_index'] not in unique_prompts:
                unique_prompts[step['prompt_index']] = step.get('prompt_vector')

        if not unique_prompts:
            self.file_logger.info("get_buffer_trajectories: 在线批次中没有 prompt_index，返回空 G_buffer。")
            return []
            
        self.file_logger.info(f"--- 开始获取 G_buffer (请求 {len(unique_prompts)} 个 unique prompts) ---")
        self.file_logger.debug(f"请求的 prompts: {list(unique_prompts.keys())}")
            
        buffer_log_paths: Set[str] = set()
        
        for prompt_index, prompt_vector in unique_prompts.items():
            
            # --- 1. 直接匹配 ---
            if prompt_index in self.db:
                self.file_logger.debug(f"Prompt {prompt_index}: 直接命中。添加 {len(self.db[prompt_index])} 条轨迹。")
                for R_tau, log_dir_path in self.db[prompt_index]:
                    buffer_log_paths.add(log_dir_path)
            
            # --- 2. 相似度回退 (如果直接匹配失败) ---
            elif self.known_vectors is not None and prompt_vector is not None:
                logger.debug(f"[STDB] Prompt {prompt_index} 未找到, 执行相似度搜索...")
                self.file_logger.info(f"Prompt {prompt_index}: 未命中。执行相似度回退...")
                # 将查询向量移到 CPU 并确保类型正确
                query_vec = prompt_vector.cpu().float().unsqueeze(0)
                
                # 计算余弦相似度
                similarities = util.cos_sim(query_vec, self.known_vectors)[0]
                
                # 找到最佳匹配
                best_match_tensor_idx = torch.argmax(similarities).item()
                best_match_prompt_index = self.known_indices[best_match_tensor_idx]
                
                logger.debug(f"     -> 最佳匹配: Prompt {best_match_prompt_index} (相似度: {similarities[best_match_tensor_idx]:.4f})")
                self.file_logger.info(f"  -> Prompt {prompt_index} 匹配到 {best_match_prompt_index} (相似度: {similarities[best_match_tensor_idx]:.4f})。")
                
                # 使用最佳匹配的轨迹
                for R_tau, log_dir_path in self.db[best_match_prompt_index]:
                    buffer_log_paths.add(log_dir_path)
            else:
                self.file_logger.warning(f"Prompt {prompt_index}: 未命中，且无法执行相似度搜索 (known_vectors: {self.known_vectors is not None}, prompt_vector: {prompt_vector is not None})")
        
        if not buffer_log_paths:
            self.file_logger.warning(f"未找到任何锚点轨迹。")
            return []

        # --- 3. 从磁盘“软链接”加载轨迹 ---
        logger.info(f"[STDB] 加载 {len(buffer_log_paths)} 条锚点轨迹...")
        self.file_logger.info(f"将从磁盘加载 {len(buffer_log_paths)} 条唯一的锚点轨迹...")
        g_buffer_steps: List[Dict[str, Any]] = []
        for log_path in buffer_log_paths:
            try:
                traj_steps = self._load_trajectory_from_path(log_path)
                g_buffer_steps.extend(traj_steps)
                self.file_logger.debug(f"  -> 成功从 {log_path} 加载 {len(traj_steps)} 个步骤。")
            except Exception as e:
                logger.error(f"[STDB] 无法从 {log_path} 加载轨迹: {e}")
                self.file_logger.error(f"无法从 {log_path} 加载轨迹: {e}")
                
        self.file_logger.info(f"--- G_buffer 获取完成 (共加载 {len(g_buffer_steps)} 个步骤) ---")
        return g_buffer_steps

    def _load_trajectory_from_path(self, log_dir_path: str) -> List[Dict[str, Any]]:
        """
        按需从磁盘加载轨迹数据（“软链接”解析）。
        这会从 summary.json 和 /step_N/step_details.json 中重新组合轨迹。
        """
        
        # --- 1. 加载轨迹级(Macro)数据 ---
        summary_path = os.path.join(log_dir_path, "summary.json")
        if not os.path.exists(summary_path):
            self.file_logger.error(f"加载轨迹失败: 未找到 summary.json于 {log_dir_path}")
            raise FileNotFoundError(f"未找到 summary.json于 {log_dir_path}")
            
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
            
        traj_task_completed = summary_data.get('task_completed', False)
        traj_total_steps = summary_data.get('step_count', 0)
        traj_total_tokens = summary_data.get('token_usage', {}).get('total_tokens', 0)
        
        # --- 2. 加载步骤级(Micro)数据 ---
        loaded_steps = []
        
        # 2.1 预计算 N_success(tau)
        traj_n_success_steps = 0
        for i in range(traj_total_steps + 1): # 假设 step_count 是从0开始的
            step_detail_path = os.path.join(log_dir_path, f"step_{i}", "step_details.json")
            if os.path.exists(step_detail_path):
                try:
                    with open(step_detail_path, 'r', encoding='utf-8') as f:
                        step_data = json.load(f)
                    if step_data.get('action_success', False):
                        traj_n_success_steps += 1
                except Exception as e:
                    self.file_logger.warning(f"读取 {step_detail_path} 时出错: {e}")
            
        # 2.2 加载每一步
        for i in range(traj_total_steps + 1): # 假设 step_count 是从0开始的
            step_detail_path = os.path.join(log_dir_path, f"step_{i}", "step_details.json")
            if not os.path.exists(step_detail_path):
                continue
                
            with open(step_detail_path, 'r', encoding='utf-8') as f:
                step_data = json.load(f)

            # R_step, A_step, R_tau, A_tau 等将在 Phase 3 计算
            # 我们只加载计算它们所需的原始数据
            
            # 将 log_probs 从 list 转回 tensor
            log_probs_list = step_data.get('rollout_log_probs')
            if isinstance(log_probs_list, list):
                rollout_log_probs = torch.tensor(log_probs_list)
            else:
                rollout_log_probs = torch.tensor([]) # 占位
            
            rehydrated_step = {
                # 轨迹级(Macro)数据
                'traj_task_completed': traj_task_completed,
                'traj_total_steps': traj_total_steps,
                'traj_total_tokens': traj_total_tokens,
                'traj_n_success_steps': traj_n_success_steps,
                
                # 步骤级(Micro)数据
                'step_index': step_data.get('step_number', i),
                'thought': step_data.get('thought', ''),
                'parsed_action': step_data.get('parsed_action', ''),
                'action_type': step_data.get('action_type', ''),
                'action_success': step_data.get('action_success', False),
                'step_token_usage': step_data.get('token_usage', {}),
                'step_confidence': step_data.get('confidence_metrics', {}).get('average_confidence', 0.0),
                # --- ✅ [CCAPO V3] 关键修正：加载 action_status ---
                'action_status': step_data.get('action_status', ''),
                
                # 核心 RL 数据
                'rollout_log_probs': rollout_log_probs, # $\pi_{\theta_{stored}}$
                
                # 标识符
                'log_dir_path': log_dir_path,
                'is_buffer_data': True, # 标记这是来自 STDB 的数据
            }
            loaded_steps.append(rehydrated_step)
            
        return loaded_steps