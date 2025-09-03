import os
import json
import datetime
import logging
import copy  # 导入copy模块


class InfoPoolManager:
    def __init__(self, run_directory: str):
        """
        初始化Info Pool Manager。
        - 使用一个预先创建好的目录来存放所有运行数据。
        - 初始化一个空列表来存储完整的执行轨迹。
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.run_dir = run_directory

        if not os.path.isdir(self.run_dir):
            raise FileNotFoundError(f"运行目录 {self.run_dir} 未找到。")

        # 新增：用于存储整个运行过程的轨迹列表
        self.full_trace = []
        self.step_count = 0

        self.logger.info(f"信息池已关联到目录: {self.run_dir}")

    def record_step(self, step_data: dict):
        """
        记录一个完整步骤的数据。
        - 为该步骤创建一个独立的子文件夹。
        - 在子文件夹中保存截图、XML、简化布局和步骤详情JSON。
        - 将该步骤的信息追加到完整轨迹列表中。
        """
        self.step_count += 1
        step_folder_name = f"step_{self.step_count:03d}"
        step_dir = os.path.join(self.run_dir, step_folder_name)

        try:
            os.makedirs(step_dir, exist_ok=True)
            self.logger.info(f"正在记录第 {self.step_count} 步于: {step_dir}")
        except OSError as e:
            self.logger.error(f"为步骤 {self.step_count} 创建目录失败: {e}")
            return  # 如果目录创建失败，则不继续

        # 创建一个数据的深拷贝，用于追加到完整轨迹，避免后续修改影响
        trace_step_data = copy.deepcopy(step_data)

        # --- 将大文件保存到步骤子目录 ---

        # 保存截图
        if "screenshot_bytes" in step_data and step_data["screenshot_bytes"]:
            screenshot_path = os.path.join(step_dir, "screenshot.png")
            with open(screenshot_path, "wb") as f:
                f.write(step_data["screenshot_bytes"])
            # 更新JSON中的路径为相对路径
            trace_step_data["observation"]["screenshot_path"] = os.path.join(
                step_folder_name, "screenshot.png"
            )

        # 保存原始XML
        if "xml_content" in step_data and step_data["xml_content"]:
            xml_path = os.path.join(step_dir, "layout.xml")
            with open(xml_path, "w", encoding="utf-8") as f:
                f.write(step_data["xml_content"])
            trace_step_data["observation"]["xml_path"] = os.path.join(
                step_folder_name, "layout.xml"
            )

        # 新增：保存简化后的UI布局
        if (
            "simplified_elements_str" in step_data["observation"]
            and step_data["observation"]["simplified_elements_str"]
        ):
            simplified_path = os.path.join(step_dir, "simplified_layout.txt")
            with open(simplified_path, "w", encoding="utf-8") as f:
                f.write(step_data["observation"]["simplified_elements_str"])
            trace_step_data["observation"]["simplified_layout_path"] = os.path.join(
                step_folder_name, "simplified_layout.txt"
            )

        # 新增: 将完整的LLM对话保存到JSON文件
        if "llm_prompt" in step_data and "raw_llm_response" in step_data:
            dialogue_path = os.path.join(step_dir, "llm_dialogue.json")
            dialogue_content = {
                "prompt": step_data["llm_prompt"],
                "response": step_data["raw_llm_response"],
            }
            with open(dialogue_path, "w", encoding="utf-8") as f:
                json.dump(dialogue_content, f, indent=2, ensure_ascii=False)
            trace_step_data["llm_dialogue_path"] = os.path.join(
                step_folder_name, "llm_dialogue.json"
            )

        # --- 清理原始数据，准备写入JSON ---
        step_data.pop("screenshot_bytes", None)
        step_data.pop("xml_content", None)
        step_data.pop("llm_prompt", None)  # 从JSON中移除，因为它已经存为文件
        step_data.pop("raw_llm_response", None)  # 从JSON中移除

        trace_step_data.pop("screenshot_bytes", None)
        trace_step_data.pop("xml_content", None)
        trace_step_data.pop("llm_prompt", None)
        trace_step_data.pop("raw_llm_response", None)

        # 保存该步骤的详细JSON
        step_details_path = os.path.join(step_dir, "step_details.json")
        try:
            with open(step_details_path, "w", encoding="utf-8") as f:
                json.dump(step_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"步骤 {self.step_count} 详情已保存至 {step_details_path}")
        except Exception as e:
            self.logger.error(f"保存步骤 {self.step_count} JSON数据失败: {e}")

        # --- 将本步骤信息追加到完整轨迹中 ---
        self.full_trace.append(trace_step_data)

    def finalize_run(
        self,
        status: str,
        summary: str,
        run_start_time: datetime.datetime,  # 这个是带时区的时间
        task: str,
        token_usage: dict = None,  # 新增: 接收token使用情况
    ):
        """
        在任务结束时写入总结文件和完整的执行轨迹文件。
        """
        # --- 修正点：确保 run_end_time 与 run_start_time 的时区一致 ---
        # 使用 run_start_time 的时区信息来获取当前的结束时间，从而确保两者都是 offset-aware
        run_end_time = datetime.datetime.now(run_start_time.tzinfo)
        duration = run_end_time - run_start_time

        # 1. 写入包含最终状态和摘要的 summary.json 文件
        summary_data = {
            "run_start_time": run_start_time.isoformat(),
            "run_end_time": run_end_time.isoformat(),
            "duration_seconds": round(duration.total_seconds(), 2),
            "task_description": task,
            "final_status": status,
            "total_steps": self.step_count,
            "summary_text": summary,
            # 新增: 将token使用情况添加到总结中
            "token_usage": token_usage
            or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        summary_path = os.path.join(self.run_dir, "summary.json")
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"任务运行总结已保存: {summary_path}")
        except Exception as e:
            self.logger.error(f"保存运行总结失败: {e}")

        # 2. 新增：写入完整的执行轨迹文件 execution_trace.json
        trace_data = {
            "metadata": summary_data,  # 在轨迹文件中也包含元数据
            "trace": self.full_trace,
        }
        trace_path = os.path.join(self.run_dir, "execution_trace.json")
        try:
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"完整执行轨迹已保存: {trace_path}")
        except Exception as e:
            self.logger.error(f"保存完整执行轨迹失败: {e}")
