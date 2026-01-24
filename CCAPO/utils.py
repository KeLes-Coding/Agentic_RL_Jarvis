# CCAPO/utils.py

import numpy as np

def detect_loop(action_history: list, window: int = 3) -> bool:
    """简单的死循环检测"""
    if len(action_history) < window * 2: return False
    last = action_history[-1]
    # 连续重复: A, A, A
    if action_history[-2] == last and action_history[-3] == last:
        return True
    # 振荡: A, B, A, B
    if len(action_history) >= 4:
        if action_history[-1] == action_history[-3] and action_history[-2] == action_history[-4]:
            return True
    return False