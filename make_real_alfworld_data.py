import os
import glob
import pandas as pd
import argparse
import logging
import random
import json
import hashlib

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_all_game_files(base_path, split_names=["train"]):
    """扫描指定 split 目录下的所有游戏文件"""
    all_files = []
    for split in split_names:
        search_pattern = os.path.join(base_path, split, "**", "game.tw-pddl")
        files = glob.glob(search_pattern, recursive=True)
        logger.info(f"Scanned {split}: found {len(files)} files.")
        all_files.extend(files)
    return all_files

def process_files(file_list, split_label):
    data_list = []
    for f_path in file_list:
        full_dir = os.path.dirname(f_path)
        
        # 构造相对路径作为 ID
        try:
            rel_path = full_dir.split("json_2.1.1/")[-1] 
        except IndexError:
            rel_path = full_dir 

        # --- 🔥 [核心修复] 构造符合 Chat Template 的 prompt ---
        # 即使我们在 Env Reset 时会覆盖这个 prompt，
        # 我们也必须给 Tokenizer 一个合法的结构，否则它会在数据预处理阶段报错。
        chat_prompt = [
            {
                "role": "user",
                "content": rel_path  # 这里放 ID 没问题，环境管理器会提取 meta info
            }
        ]
        # --------------------------------------------------

        data_list.append({
            "prompt": chat_prompt,     # List[Dict] 格式，解决 string indices error
            "prompt_index": rel_path,  # 原始 ID，用于 STDB
            "game_path": full_dir,     
            "ability": "alfworld",
            "split": split_label,
            "data_source": "alfworld"  # 解决 data_source KeyError
        })
    return data_list

def get_config_hash(args):
    """计算配置的哈希值"""
    config_dict = {
        "data_root": args.data_root,
        "total_samples": args.total_samples,
        "train_ratio": args.train_ratio,
        "seed": args.seed
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode('utf-8')).hexdigest(), config_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=os.path.expanduser("~/.cache/alfworld/json_2.1.1"))
    parser.add_argument("--output_dir", type=str, default="data/verl-agent/text")
    
    # 采样控制
    parser.add_argument("--total_samples", type=int, default=-1, help="提取的总数据量。-1 表示全量。")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集占比。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--force", action="store_true", help="强制重新生成，忽略缓存。")
    
    args = parser.parse_args()

    # --- 1. 检查缓存 ---
    os.makedirs(args.output_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, "dataset_meta.json")
    train_path = os.path.join(args.output_dir, "train.parquet")
    
    current_hash, current_config = get_config_hash(args)
    
    if os.path.exists(meta_path) and os.path.exists(train_path) and not args.force:
        try:
            with open(meta_path, 'r') as f:
                saved_meta = json.load(f)
            if saved_meta.get('config_hash') == current_hash:
                logger.info("✅ Dataset config unchanged. Skipping regeneration.")
                return 
        except: pass
    
    # --- 2. 生成数据 ---
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"ALFWorld data not found at {args.data_root}")

    raw_files = get_all_game_files(args.data_root, split_names=["train"])
    if not raw_files:
        logger.error("No game files found!")
        return

    # 排序 + Shuffle
    raw_files.sort()
    random.seed(args.seed)
    random.shuffle(raw_files)

    # 截取
    total_available = len(raw_files)
    num_to_take = args.total_samples
    if num_to_take == -1 or num_to_take > total_available:
        num_to_take = total_available
    
    selected_files = raw_files[:num_to_take]

    # 切分
    num_train = int(num_to_take * args.train_ratio)
    if num_train == 0 and num_to_take > 0: num_train = 1
    
    train_files = selected_files[:num_train]
    test_files = selected_files[num_train:]

    logger.info(f"Processing -> Train: {len(train_files)} | Test: {len(test_files)}")

    # 保存
    df_train = pd.DataFrame(process_files(train_files, "train"))
    df_train.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    
    if test_files:
        df_test = pd.DataFrame(process_files(test_files, "test"))
        df_test.to_parquet(os.path.join(args.output_dir, "test.parquet"))

    # 保存元数据
    meta_info = {
        "config_hash": current_hash,
        "config": current_config,
        "generated_at": pd.Timestamp.now().isoformat()
    }
    with open(meta_path, 'w') as f:
        json.dump(meta_info, f, indent=2)
    logger.info(f"Dataset generated and saved to {args.output_dir}")

if __name__ == "__main__":
    main()