# CCAPO/utils.py
import numpy as np

def compute_lcs_mask(sequence: list, anchor: list) -> list:
    """计算 sequence 相对于 anchor 的 LCS 匹配掩码"""
    n = len(sequence)
    m = len(anchor)
    if n == 0 or m == 0: return [False] * n
    
    # DP Table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if sequence[i-1] == anchor[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Backtrack
    mask = [False] * n
    i, j = n, m
    while i > 0 and j > 0:
        if sequence[i-1] == anchor[j-1]:
            mask[i-1] = True
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    return mask

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