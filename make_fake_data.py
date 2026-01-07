import pandas as pd
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', type=int, default=16)
    parser.add_argument('--val_size', type=int, default=4)
    args = parser.parse_args()

    print(f"Generating {args.train_size} train samples and {args.val_size} val samples...")
    
    # 训练集
    train_data = []
    for i in range(args.train_size):
        train_data.append({
            "prompt": [{"role": "user", "content": "Interact with the environment."}],
            "data_source": "alfworld",
            "ability": "text",
            "reward_model": {"style": "rule", "ground_truth": "N/A"},
            "extra_info": {"task_id": f"train_{i}"}
        })
    df_train = pd.DataFrame(train_data)
    
    # 验证集
    val_data = []
    for i in range(args.val_size):
        val_data.append({
            "prompt": [{"role": "user", "content": "Interact with the environment."}],
            "data_source": "alfworld",
            "ability": "text",
            "reward_model": {"style": "rule", "ground_truth": "N/A"},
            "extra_info": {"task_id": f"val_{i}"}
        })
    df_val = pd.DataFrame(val_data)
    
    # --- 🔥 [修改] 使用当前目录下的 data 文件夹 ---
    current_dir = os.getcwd()
    output_dir = os.path.join(current_dir, "data/verl-agent/text")
    os.makedirs(output_dir, exist_ok=True)
    # -------------------------------------------
    
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    df_train.to_parquet(train_path)
    df_val.to_parquet(test_path)
    
    print(f"✅ Data saved to:\n  {train_path}\n  {test_path}")

if __name__ == "__main__":
    main()