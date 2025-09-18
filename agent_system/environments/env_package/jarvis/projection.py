# agent_system/environments/env_package/jarvis/projection.py

import numpy as np
import json
from typing import List, Tuple

def jarvis_projection(text_actions: List[str]) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    解析模型生成的JSON字符串。
    如果解析失败，会生成一个特殊的'format_error'动作，以便环境可以捕获并提供反馈。
    """
    parsed_actions = []
    thoughts = []
    valids = []

    for text_action in text_actions:
        try:
            # 去除可能的 markdown 代码块标记
            cleaned_text = text_action.strip().removeprefix("```json").removesuffix("```").strip()
            
            data = json.loads(cleaned_text)
            thought = data.get("thought", "Missing 'thought' key in JSON.")
            action = data.get("action", "format_error(reason='Missing action key in JSON')")
            
            # 如果 thought 或 action 为空，也视为一种格式错误
            if not thought or not action:
                 action = f"format_error(reason='Empty thought or action value in JSON')"
                 valids.append(False)
            else:
                 valids.append(True)

            parsed_actions.append(action)
            thoughts.append(thought)

        except json.JSONDecodeError as e:
            # --- 修改：不再发送finish，而是发送一个可识别的格式错误动作 ---
            error_reason = f"Invalid JSON format: {e}"
            parsed_actions.append(f"format_error(reason='{error_reason}')")
            thoughts.append("Error: Failed to parse LLM response as valid JSON.")
            valids.append(False)
            
    return parsed_actions, np.array(valids, dtype=bool), thoughts