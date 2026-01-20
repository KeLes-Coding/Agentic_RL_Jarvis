# CCAPO/inspector.py

import os
import json
import re
from openai import OpenAI

# 配置你的 Key 和 Base URL (如果使用第三方/中转)
API_KEY = "sk-65b9208aa589434db22f3863e772b213" 
BASE_URL = "https://api.deepseek.com" 

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def parse_training_log(log_path):
    """解析 TaskRunner 的标准输出日志，提取关键指标"""
    metrics = []
    if not os.path.exists(log_path): return metrics
    
    with open(log_path, 'r') as f:
        for line in f:
            if "global_seqlen/mean" in line: # 关键行特征
                # 简单正则提取 step 和 success_rate
                step_match = re.search(r'step:(\d+)', line)
                sr_match = re.search(r'episode/success_rate:([\d\.]+)', line)
                rwd_match = re.search(r'episode/reward/mean:([\d\.]+)', line)
                
                if step_match and sr_match:
                    metrics.append({
                        "step": int(step_match.group(1)),
                        "success_rate": float(sr_match.group(1)),
                        "mean_reward": float(rwd_match.group(1) if rwd_match else 0.0)
                    })
    return metrics

def parse_ccapo_log(log_dir):
    """解析 CCAPO 内部日志"""
    events = []
    # 读取 stdb_server.jsonl
    stdb_path = os.path.join(log_dir, "stdb_server.jsonl")
    if os.path.exists(stdb_path):
        with open(stdb_path, 'r') as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except: pass
    return events

def analyze_with_llm(training_metrics, ccapo_events):
    """调用 LLM 生成分析报告"""
    
    # 构造 Prompt Context
    context = f"""
    You are an expert in Reinforcement Learning, specifically analyzing the CCAPO (Consensus-based Context-Aware Policy Optimization) framework.
    
    I will provide you with two sets of data from a training run:
    1. **Training Metrics**: PPO metrics like Success Rate, Reward, etc.
    2. **CCAPO Internal Events**: Logs from the STDB (State-Trajectory Database), showing Anchor updates and Logic consensus.
    
    **Your Task**: Analyze the effectiveness of the CCAPO mechanism.
    
    **Specific Questions to Answer**:
    1. **Is the Execution Stream working?** (Are we finding better anchors? Is the step count of anchors decreasing?)
    2. **Is the STDB growing?** (Are we accumulating knowledge or just stagnating?)
    3. **Correlation**: Do updates in STDB correlate with increases in Success Rate?
    4. **Potential Issues**: Are there signs of collapse (e.g., Success Rate drops while Anchors update)?
    
    --- DATA START ---
    
    [Training Metrics (Last 5 steps)]:
    {json.dumps(training_metrics[-5:], indent=2)}
    
    [CCAPO Anchor Updates (Sample)]:
    {json.dumps([e for e in ccapo_events if e.get('event') == 'anchor_update'][-10:], indent=2)}
    
    --- DATA END ---
    
    Please output the report in Chinese (Markdown format).
    """
    
    response = client.chat.completions.create(
        model="deepseek-chat", # 或 gpt-3.5-turbo / deepseek-chat
        messages=[
            {"role": "system", "content": "You are a helpful AI research assistant."},
            {"role": "user", "content": context}
        ]
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    # 这里填入你实际的日志路径
    # 提示：运行训练时请使用: ./run_script.sh | tee training.log
    TRAIN_LOG = "training.log" 
    CCAPO_LOG_DIR = "experiments/ccapo_logs" # 假设这是 Server 写入的目录
    
    print(">>> Parsing logs...")
    t_metrics = parse_training_log(TRAIN_LOG)
    c_events = parse_ccapo_log(CCAPO_LOG_DIR)
    
    print(f"Found {len(t_metrics)} training steps and {len(c_events)} CCAPO events.")
    
    if len(t_metrics) > 0:
        print(">>> Generating AI Report...")
        report = analyze_with_llm(t_metrics, c_events)
        print("\n" + "="*30 + " AI ANALYSIS REPORT " + "="*30 + "\n")
        print(report)
        
        # 保存报告
        with open("latest_analysis_report.md", "w") as f:
            f.write(report)
    else:
        print("No metrics found. Did you pipe stdout to 'training.log'?")