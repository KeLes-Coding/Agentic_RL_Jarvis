# agent_system/environments/env_package/jarvis/info_pool.py

import os
import json
import datetime
import io
from typing import List, Dict, Any
import yaml
import logging # <-- ✅ [日志] 新增

try:
    from PIL import Image
except ImportError:
    Image = None
    
# ======================= ✅ 1. 移除评估函数 ✅ =======================
# _evaluate_with_llm 函数已被移动到 envs.py
# =====================================================================

# --- ✅ [日志] 专用文件日志器 (用于 logger/INFO_POOL/info_pool_operations.log) ---
info_pool_file_logger = logging.getLogger("INFO_POOL_FILE")
info_pool_file_logger.setLevel(logging.INFO) # 捕获 INFO 及以上级别
info_pool_file_logger.propagate = False      # 防止重复记录到 root logger

# 仅在日志器没有处理器时才添加，以防止重复
if not info_pool_file_logger.handlers:
    try:
        log_dir = "logger/INFO_POOL"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "info_pool_operations.log")
        
        # 创建文件处理器 (追加模式)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - [INFO_POOL_FILE] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # 添加处理器
        info_pool_file_logger.addHandler(file_handler)
        info_pool_file_logger.info("--- InfoPool 专用文件日志器已初始化 ---")
        
    except Exception as e:
        # 使用标准日志器打印错误，因为文件日志器可能失败了
        print(f"[InfoPool] 无法创建专用文件日志器: {e}")
# --- 日志设置结束 ---


class InfoPoolManager:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.trajectory_data: List[Dict[str, Any]] = []
        self.step_count = -1  # 从-1开始，reset是第0步
        
        # --- ✅ [日志] 分配日志器实例 ---
        self.file_logger = info_pool_file_logger
        self.file_logger.info(f"--- InfoPoolManager 实例已创建 ---")
        self.file_logger.info(f"Log 目录: {self.log_dir}")
        # --- 结束 ---

        # --- 修改：定义 trace 文件路径并在初始化时创建空文件 ---
        self.trace_path = os.path.join(self.log_dir, "execution_trace.json")
        # 初始化一个空的JSON列表，确保文件存在且格式正确
        with open(self.trace_path, "w", encoding="utf-8") as f:
            json.dump([], f)

        # ======================= ✅ 新增: 初始化 Token 和置信度累加器 ✅ =======================
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.step_confidences: List[float] = []
        # =================================================================================

    def record_step(self, step_data: Dict[str, Any]):
        self.step_count += 1
        step_dir = os.path.join(self.log_dir, f"step_{self.step_count}")
        os.makedirs(step_dir, exist_ok=True)
        
        # --- ✅ [日志升级] 详细记录 I/O 路径与身份标识 ---
        try:
            # 获取路径的最后一段作为简易 UID 用于显示
            short_uid = os.path.basename(self.log_dir)
            self.file_logger.info(f"[record_step] Step {self.step_count} | UID: {short_uid}")
            
            # 检查关键 RL 标识符
            prompt_index = step_data.get('prompt_index', 'MISSING')
            prompt_vector_exists = 'prompt_vector' in step_data and step_data['prompt_vector'] is not None
            
            self.file_logger.info(f"    -> prompt_index: {prompt_index}")
            self.file_logger.info(f"    -> prompt_vector present: {prompt_vector_exists}")
            
            # [冲突预警] 如果 step_count 很大，可能是 append 到了旧文件？
            if self.step_count == 0 and os.path.exists(os.path.join(step_dir, "step_details.json")):
                 self.file_logger.warning(f"!!! CRITICAL WARNING: Step 0 file already exists in {step_dir}! Potential Overwrite/Collision!")

        except Exception as e:
            self.file_logger.error(f"[record_step] 日志记录失败: {e}")
        # --- 日志结束 ---

        # ======================= ✅ 新增: 累加 Token 和置信度 ✅ =======================
        if "token_usage" in step_data and isinstance(step_data.get("token_usage"), dict):
            self.total_prompt_tokens += step_data["token_usage"].get("prompt_tokens", 0)
            self.total_completion_tokens += step_data["token_usage"].get("completion_tokens", 0)
            self.total_tokens += step_data["token_usage"].get("total_tokens", 0)
        
        if "confidence_metrics" in step_data and isinstance(step_data.get("confidence_metrics"), dict):
            self.step_confidences.append(step_data["confidence_metrics"].get("average_confidence", 0.0))
        # =========================================================================

        # 准备要记录的步骤数据 (Macro Trace)
        current_step_record = {
            "step": self.step_count,
            "thought": step_data["thought"],
            "action": step_data["parsed_action"],
            "action_success": step_data["action_success"]
        }
        
        self.trajectory_data.append(current_step_record)

        # 同步写入 Trace
        try:
            with open(self.trace_path, "w", encoding="utf-8") as f:
                json.dump(self.trajectory_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.file_logger.error(f"[record_step] 写入 execution_trace.json 失败: {e}")

        # --- 保存每一步的详细文件 ---
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
                "raw_llm_response": step_data.get("raw_llm_response", "N/A"),
                "action_type": step_data.get("action_type", "unknown"),
                "action_success": step_data["action_success"],
                "action_status": step_data.get("action_status", ""),
                
                # [关键] 显式写入 log_dir_path，作为数据的身份证明
                "log_dir_path": self.log_dir,
                # [关键] 写入简易 UID，便于人工检查
                "traj_uid_hint": short_uid
            }

            if "token_usage" in step_data:
                details["token_usage"] = step_data["token_usage"]
            if "confidence_metrics" in step_data:
                details["confidence_metrics"] = step_data["confidence_metrics"]
            
            # --- Tensor Serialization ---
            if "rollout_log_probs" in step_data:
                try:
                    details["rollout_log_probs"] = step_data["rollout_log_probs"].cpu().tolist()
                except Exception:
                    details["rollout_log_probs"] = "Error: Not serializable"
            
            for key in ["input_ids", "attention_mask", "position_ids", "responses"]:
                if key in step_data:
                    try:
                        details[key] = step_data[key].cpu().tolist()
                    except Exception:
                        details[key] = "Error: Not serializable"
            
            if "multi_modal_inputs" in step_data:
                 details["multi_modal_inputs"] = step_data["multi_modal_inputs"]

            if "prompt_index" in step_data:
                details["prompt_index"] = step_data["prompt_index"]
            if "prompt_vector" in step_data:
                try:
                    details["prompt_vector"] = step_data["prompt_vector"].cpu().tolist()
                except Exception:
                    details["prompt_vector"] = "Error: Not serializable"

            # 奖励占位符
            details["reward_components"] = {
                "note": "Reward components will be populated here after calculation by dp_actor."
            }

            with open(os.path.join(step_dir, "step_details.json"), "w", encoding="utf-8") as f:
                json.dump(details, f, indent=4, ensure_ascii=False)

        except Exception as e:
            self.file_logger.error(f"[record_step] 保存步骤 {self.step_count} artifacts 失败: {e}")

    
    # ======================= ✅ 2. 修改 finalize_run 方法签名和逻辑 ✅ =======================
    def finalize_run(self, status: str, summary: str, run_start_time: datetime, task: str, task_completed: bool):
    # ======================================================================================
        end_time = datetime.datetime.now(datetime.timezone.utc)
        duration = end_time - run_start_time
        
        self.file_logger.info(f"--- [finalize_run] 轨迹结束 ---")
        self.file_logger.info(f"任务 (截断): {task[:100]}...")
        self.file_logger.info(f"状态: {status}, 最终摘要: {summary}")
        self.file_logger.info(f"是否完成: {task_completed}")

        # 评估逻辑已移至 envs.py, 这里直接使用传入的 task_completed 参数
        
        # --- 修改：execution_trace.json 已经是最新状态，这里的写入作为最终确认 ---
        # 这一步也可以移除，但保留可以作为一种保障机制
        try:
            with open(self.trace_path, "w", encoding="utf-8") as f:
                json.dump(self.trajectory_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.file_logger.error(f"[finalize_run] 写入 final execution_trace.json 失败: {e}")
            print(f"Error writing final execution_trace.json: {e}")

        # --- 保存 summary.json (这部分逻辑保持不变) ---
        summary_data = {
            "task": task,
            "status": status,
            "summary": summary,
            "task_completed": task_completed, # 直接记录传入的结果
            "start_time_utc": run_start_time.isoformat(),
            "end_time_utc": end_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "step_count": self.step_count,
            # ======================= ✅ 使用内部累加的 Token 和置信度数据 ✅ =======================
            "token_usage": {
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens
            },
            "confidence_metrics": {
                "average_confidence_over_trajectory": 
                    (sum(self.step_confidences) / len(self.step_confidences)) if self.step_confidences else 0.0
            }
            # ===============================================================================
        }
        summary_path = os.path.join(self.log_dir, "summary.json")
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=4, ensure_ascii=False)
            self.file_logger.info(f"[finalize_run] summary.json 已保存。")
        except Exception as e:
            self.file_logger.error(f"[finalize_run] 写入 summary.json 失败: {e}")
            print(f"Error writing summary.json: {e}")
        
        return summary_data # <--- ✅ [CCAPO] 返回 summary_data