# agent_system/environments/env_package/jarvis/projection.py

import numpy as np
import json
from typing import List, Tuple

def jarvis_projection(text_actions: List[str]) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    解析模型生成的JSON字符串。

    Args:
        text_actions: 模型生成的包含 "thought" 和 "action" 的JSON字符串列表。

    Returns:
        - A list of action strings (e.g., "tap(1)").
        - A numpy array of booleans indicating if the parsing was valid.
        - A list of thought strings.
    """
    parsed_actions = []
    thoughts = []
    valids = []

    for text_action in text_actions:
        try:
            # 去除可能的 markdown 代码块标记
            cleaned_text = text_action.strip().removeprefix("```json").removesuffix("```").strip()
            
            data = json.loads(cleaned_text)
            thought = data.get("thought", "")
            action = data.get("action", "finish(reason='Parsing error')")
            
            parsed_actions.append(action)
            thoughts.append(thought)
            valids.append(True)
        except json.JSONDecodeError:
            # 如果JSON解析失败，我们默认执行 finish 动作并记录错误
            parsed_actions.append("finish(reason='Invalid JSON format')")
            thoughts.append("Error: Failed to parse LLM response as valid JSON.")
            valids.append(False)
            
    return parsed_actions, np.array(valids, dtype=bool), thoughts