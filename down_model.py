import os
from modelscope import snapshot_download

# ================= 配置区域 =================
# 目标下载路径 (AutoDL 的数据盘通常挂载在 /root/autodl-tmp)
target_root = "/root/autodl-tmp/models"

# 模型ID (请确保 ModelScope 上有完全一致的 ID)
# 常用候选项:
# Qwen/Qwen2.5-VL-3B-Instruct (如果是 Qwen2.5 3B 版本)
# Qwen/Qwen2.5-VL-7B-Instruct
# Qwen/Qwen2-VL-2B-Instruct (如果是 Qwen2 2B 版本)
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
# ===========================================

def download_model():
    # 1. 自动创建目录
    if not os.path.exists(target_root):
        print(f"正在创建目录: {target_root}")
        os.makedirs(target_root, exist_ok=True)
    else:
        print(f"目录已存在: {target_root}")

    print(f"开始从 ModelScope 下载 [{model_id}] ...")
    
    try:
        # 2. 执行下载
        # cache_dir 指定下载文件存放的根目录
        model_dir = snapshot_download(
            model_id, 
            cache_dir=target_root,
            revision='master' # 默认下载最新版
        )
        
        print("\n" + "="*50)
        print("下载成功！")
        print(f"模型保存路径: {model_dir}")
        print("="*50)
        
    except Exception as e:
        print("\n下载失败，请检查网络或模型ID是否正确。")
        print(f"错误信息: {e}")

if __name__ == "__main__":
    download_model()