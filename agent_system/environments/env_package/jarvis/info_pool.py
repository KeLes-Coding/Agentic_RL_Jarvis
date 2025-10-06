# agent_system/environments/env_package/jarvis/info_pool.py

import os
import json
import datetime
import io
from typing import List, Dict, Any

try:
    from PIL import Image
except ImportError:
    Image = None

class InfoPoolManager:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.trajectory_data: List[Dict[str, Any]] = []
        self.step_count = -1  # 从-1开始，reset是第0步

        # --- 修改：定义 trace 文件路径并在初始化时创建空文件 ---
        self.trace_path = os.path.join(self.log_dir, "execution_trace.json")
        # 初始化一个空的JSON列表，确保文件存在且格式正确
        with open(self.trace_path, "w", encoding="utf-8") as f:
            json.dump([], f)

    def record_step(self, step_data: Dict[str, Any]):
        self.step_count += 1
        step_dir = os.path.join(self.log_dir, f"step_{self.step_count}")
        os.makedirs(step_dir, exist_ok=True)

        # 准备要记录的步骤数据
        current_step_record = {
            "step": self.step_count,
            "thought": step_data["thought"],
            "action": step_data["parsed_action"],
            "action_success": step_data["action_success"]
        }
        
        # 添加到内存轨迹列表中
        self.trajectory_data.append(current_step_record)

        # --- 修改：在每一步都将完整的轨迹数据覆写到文件中 ---
        try:
            with open(self.trace_path, "w", encoding="utf-8") as f:
                json.dump(self.trajectory_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error synchronously writing to {self.trace_path} at step {self.step_count}: {e}")
        
        # --- 保存每一步的详细文件 (这部分逻辑保持不变) ---
        try:
            # 1. layout.xml
            xml_content = step_data["raw_obs_data"].get("xml", "")
            if xml_content:
                with open(os.path.join(step_dir, "layout.xml"), "w", encoding="utf-8") as f:
                    f.write(xml_content)

            # 2. llm_dialogue.json
            dialogue_data = {
                "prompt": step_data["llm_prompt"],
                "response": step_data["raw_llm_response"]
            }
            with open(os.path.join(step_dir, "llm_dialogue.json"), "w", encoding="utf-8") as f:
                json.dump(dialogue_data, f, indent=4, ensure_ascii=False)

            # 3. screenshot.png
            if Image and step_data.get("compressed_screenshot_bytes"):
                screenshot_bytes = step_data["compressed_screenshot_bytes"]
                if screenshot_bytes:
                    img = Image.open(io.BytesIO(screenshot_bytes))
                    img.save(os.path.join(step_dir, "screenshot.png"))

            # 4. simplified_layout.txt
            simplified_layout = step_data["raw_obs_data"].get("simplified_elements_str", "")
            with open(os.path.join(step_dir, "simplified_layout.txt"), "w", encoding="utf-8") as f:
                f.write(simplified_layout)
                 
            # 5. step_details.json
            details = {
                "step_number": self.step_count,
                "task": step_data["task"],
                "thought": step_data["thought"],
                "parsed_action": step_data["parsed_action"],
                "action_success": step_data["action_success"],
            }
            with open(os.path.join(step_dir, "step_details.json"), "w", encoding="utf-8") as f:
                json.dump(details, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"Error saving step artifacts for step {self.step_count}: {e}")

    def finalize_run(self, status: str, summary: str, run_start_time: datetime, task: str, token_usage: Dict[str, int] = None):
        end_time = datetime.datetime.now(datetime.timezone.utc)
        duration = end_time - run_start_time

        # --- 修改：execution_trace.json 已经是最新状态，这里的写入作为最终确认 ---
        # 这一步也可以移除，但保留可以作为一种保障机制
        try:
            with open(self.trace_path, "w", encoding="utf-8") as f:
                json.dump(self.trajectory_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing final execution_trace.json: {e}")

        # --- 保存 summary.json (这部分逻辑保持不变) ---
        if token_usage is None:
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
             
        total_tokens = token_usage.get("prompt_tokens", 0) + token_usage.get("completion_tokens", 0)
         
        summary_data = {
            "task": task,
            "status": status,
            "summary": summary,
            "start_time_utc": run_start_time.isoformat(),
            "end_time_utc": end_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "step_count": self.step_count,
            "token_usage": {
                "prompt_tokens": token_usage.get("prompt_tokens", 0),
                "completion_tokens": token_usage.get("completion_tokens", 0),
                "total_tokens": total_tokens
            }
        }
        summary_path = os.path.join(self.log_dir, "summary.json")
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing summary.json: {e}")