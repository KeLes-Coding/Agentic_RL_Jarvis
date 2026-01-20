# CCAPO/adapter.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import hashlib
import re

class CCAPOAdapter(ABC):
    """
    CCAPO 协议适配器基类。
    负责将特定环境（如 ALFWorld）的状态转译为 CCAPO 通用的抽象信号。
    """

    @abstractmethod
    def get_context_id(self, meta: Dict) -> str:
        """
        生成环境/任务的唯一指纹 (Group ID)。
        """
        pass

    @abstractmethod
    def get_task_type(self, meta: Dict) -> str:
        """
        生成任务类型的通用标识。
        """
        pass

    @abstractmethod
    def abstract_action(self, raw_action: str) -> str:
        """
        将原始动作转换为抽象逻辑 Token。
        """
        pass

class ALFWorldAdapter(CCAPOAdapter):
    """
    ALFWorld 环境的特定实现。
    逻辑复刻自当前的 EnvManager，确保行为一致。
    """
    def get_context_id(self, meta: Dict) -> str:
        # 使用 seed + task_string 组合哈希，确保唯一性
        seed = str(meta.get('seed', 0))
        # 优先使用 meta 中的 task_string，如果没有则尝试从 prompt_id 等字段获取
        task_str = meta.get('task_string', str(meta.get('prompt_id', 'unknown')))
        
        raw_key = f"{seed}_{task_str}"
        return hashlib.md5(raw_key.encode()).hexdigest()[:16]

    def get_task_type(self, meta: Dict) -> str:
        # 优先直接读取 meta 中的 task_type
        if 'task_type' in meta:
            return meta['task_type']
            
        # Fallback: 从 gamefile 路径解析
        gf = meta.get('gamefile', '')
        if 'pick_and_place' in gf: return 'pick_and_place'
        if 'clean' in gf: return 'clean_object'
        if 'heat' in gf: return 'heat_object'
        if 'cool' in gf: return 'cool_object'
        if 'look' in gf: return 'examine_object'
        return 'general_task'

    def abstract_action(self, raw_action: str) -> str:
        # 逻辑复刻：去除数字后缀，保留核心语义
        if not raw_action: return ""
        text = raw_action.lower().strip()
        # 移除 " 1", " 23" 等数字 ID
        abstracted = re.sub(r'\s\d+', '', text)
        return abstracted