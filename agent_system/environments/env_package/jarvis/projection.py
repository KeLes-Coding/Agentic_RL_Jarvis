# agent_system/environments/env_package/jarvis/projection.py

import numpy as np
import json
from typing import List, Tuple

def jarvis_projection(text_actions: List[str]) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    解析模型生成的JSON字符串。
    如果解析失败或格式不符合预期（例如返回列表而非对象），
    会生成一个特殊的'format_error'动作，以便环境可以捕获并提供反馈。
    """
    parsed_actions = []
    thoughts = []
    valids = []

    for text_action in text_actions:
        try:
            # 去除可能的 markdown 代码块标记
            cleaned_text = text_action.strip().removeprefix("```json").removesuffix("```").strip()
            
            # --- ✅ 核心修改：处理 json.loads 可能返回列表的情况 ✅ ---
            loaded_data = json.loads(cleaned_text)

            # 检查加载的数据类型
            if isinstance(loaded_data, dict):
                data = loaded_data # 如果是字典，直接使用
            elif isinstance(loaded_data, list) and loaded_data and isinstance(loaded_data[0], dict):
                # 如果是列表，并且列表不为空，且第一个元素是字典，则使用第一个元素
                print(f"警告: LLM返回了一个列表，将使用列表中的第一个JSON对象: {loaded_data[0]}")
                data = loaded_data[0]
            else:
                # 其他情况（空列表、非字典列表等）视为格式错误
                raise TypeError(f"Expected a JSON object (dict), but got {type(loaded_data)}")
            # --- ✅ 修改结束 ✅ ---

            thought = data.get("thought", "Missing 'thought' key in JSON.")
            action = data.get("action", "format_error(reason='Missing action key in JSON')")
            
            # --- ✅ 新增：类型健壮性检查 (解决 AttributeError) ✅ ---
            # 如果 action 存在但不是字符串（例如 LLM 返回了 {"action": {...}}），强制转为错误信息
            # 这一步防止了后续调用 .startswith() 时发生崩溃
            if action is not None and not isinstance(action, str):
                # 将非字符串的 action 内容转为字符串以便 debug，并标记为格式错误
                action = f"format_error(reason='Invalid action type: expected str but got {type(action).__name__}. Content: {str(action)}')"
            # --- ✅ 新增结束 ✅ ---

            # 如果 thought 或 action 为空，也视为一种格式错误
            if not thought or not action:
                 action = f"format_error(reason='Empty thought or action value in JSON')"
                 valids.append(False)
            # 检查 action 是否已经是 format_error，避免重复标记
            elif action.startswith("format_error"):
                 valids.append(False)
            else:
                 valids.append(True)

            parsed_actions.append(action)
            thoughts.append(thought)

        except (json.JSONDecodeError, TypeError) as e: # 增加了 TypeError 捕获
            # --- 修改：不再发送finish，而是发送一个可识别的格式错误动作 ---
            error_reason = f"Invalid JSON format or unexpected type: {e}"
            parsed_actions.append(f"format_error(reason='{error_reason}')")
            thoughts.append("Error: Failed to parse LLM response as valid JSON object.")
            valids.append(False)
            
    return parsed_actions, np.array(valids, dtype=bool), thoughts