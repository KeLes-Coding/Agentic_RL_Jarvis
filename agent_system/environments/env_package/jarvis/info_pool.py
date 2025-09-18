import os
import json
import datetime
import logging
import copy
import re

class InfoPoolManager:
    """
    管理单个任务运行的日志记录，为每个任务实例创建一个独立的日志目录。
    """
    def __init__(self, run_directory: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.run_dir = run_directory

        if not os.path.isdir(self.run_dir):
            try:
                os.makedirs(self.run_dir, exist_ok=True)
                self.logger.info(f"日志目录已创建: {self.run_dir}")
            except OSError as e:
                self.logger.error(f"创建运行目录 {self.run_dir} 失败: {e}")
                raise

        self.full_trace = []
        self.step_count = 0
        self.logger.info(f"信息池已关联到目录: {self.run_dir}")

    def record_step(self, step_data: dict):
        """
        按照 jarvis_v2 的格式记录一个步骤的数据。
        """
        self.step_count += 1
        step_folder_name = f"step_{self.step_count:03d}"
        step_dir = os.path.join(self.run_dir, step_folder_name)

        try:
            os.makedirs(step_dir, exist_ok=True)
        except OSError as e:
            self.logger.error(f"为步骤 {self.step_count} 创建目录失败: {e}")
            return

        # --- 仿照 jarvis_v2 格式构建 step_details.json 的内容 ---
        step_details = {
            "step_id": self.step_count,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "overall_task": step_data.get("task"),
            "observation": {},
            "llm_response": {
                "thought": step_data.get("thought"),
                "action": step_data.get("parsed_action"),
            },
            "execution": {
                "validated_action": step_data.get("parsed_action"),
                "status": "SUCCESS" if step_data.get("action_success") else "FAILURE",
            },
        }

        raw_obs = step_data.get("raw_obs_data", {})
        
        # 处理文件保存和路径更新
        if step_data.get("compressed_screenshot_bytes"):
            screenshot_path = os.path.join(step_dir, "screenshot.png")
            with open(screenshot_path, "wb") as f:
                f.write(step_data["compressed_screenshot_bytes"])
            step_details["observation"]["screenshot_path"] = os.path.join(step_folder_name, "screenshot.png")

        if raw_obs.get("xml_content"):
            xml_path = os.path.join(step_dir, "layout.xml")
            with open(xml_path, "w", encoding="utf-8") as f:
                f.write(raw_obs["xml_content"])
            step_details["observation"]["xml_path"] = os.path.join(step_folder_name, "layout.xml")
        
        if raw_obs.get("simplified_elements_str"):
            simplified_path = os.path.join(step_dir, "simplified_layout.txt")
            with open(simplified_path, "w", encoding="utf-8") as f:
                f.write(raw_obs["simplified_elements_str"])
            step_details["observation"]["simplified_layout_path"] = os.path.join(step_folder_name, "simplified_layout.txt")
            step_details["observation"]["simplified_elements_str"] = raw_obs["simplified_elements_str"]


        # 保存 llm_dialogue.json
        dialogue_path = os.path.join(step_dir, "llm_dialogue.json")
        dialogue_content = {
            "prompt": step_data.get("llm_prompt"),
            "response": step_data.get("raw_llm_response"),
        }
        with open(dialogue_path, "w", encoding="utf-8") as f:
            json.dump(dialogue_content, f, indent=2, ensure_ascii=False)
        step_details["llm_dialogue_path"] = os.path.join(step_folder_name, "llm_dialogue.json")

        # 保存 step_details.json
        step_details_path = os.path.join(step_dir, "step_details.json")
        try:
            with open(step_details_path, "w", encoding="utf-8") as f:
                json.dump(step_details, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存步骤 {self.step_count} JSON数据失败: {e}")

        # 将格式化后的数据追加到完整轨迹中
        self.full_trace.append(step_details)

    def finalize_run(self, status: str, summary: str, run_start_time: datetime.datetime, task: str):
        run_end_time = datetime.datetime.now(run_start_time.tzinfo)
        duration = run_end_time - run_start_time

        summary_data = {
            "run_start_time": run_start_time.isoformat(),
            "run_end_time": run_end_time.isoformat(),
            "duration_seconds": round(duration.total_seconds(), 2),
            "task_description": task,
            "final_status": status,
            "total_steps": self.step_count,
            "summary_text": summary,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, # Placeholder
        }
        
        summary_path = os.path.join(self.run_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"任务运行总结已保存: {summary_path}")

        trace_data = {"metadata": summary_data, "trace": self.full_trace}
        trace_path = os.path.join(self.run_dir, "execution_trace.json")
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"完整执行轨迹已保存: {trace_path}")