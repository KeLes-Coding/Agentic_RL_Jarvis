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
        # logger.info(f"Scanned {split}: found {len(files)} files.")
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

        # 构造符合 Chat Template 的 prompt
        chat_prompt = [
            {
                "role": "user",
                "content": rel_path 
            }
        ]

        data_list.append({
            "prompt": chat_prompt,
            "prompt_index": rel_path,
            "game_path": full_dir,
            "ability": "alfworld",
            "split": split_label,
            "data_source": "alfworld"
        })
    return data_list

def extract_task_type(game_file_path: str, data_root: str) -> str:
    """
    从 game 文件路径中提取任务类型（task type）。
    约定：data_root 下通常是 <split>/<task_type>/.../game.tw-pddl
    """
    full_dir = os.path.dirname(game_file_path)
    try:
        rel_dir = os.path.relpath(full_dir, data_root)
    except Exception:
        rel_dir = full_dir

    parts = rel_dir.split(os.sep)
    if not parts:
        return "unknown"

    # 常见结构：train/<task_type>/...
    if parts[0] in {"train", "valid", "test"}:
        return parts[1] if len(parts) > 1 else "unknown"

    # 兜底：没有 split 前缀时，取第一段
    return parts[0]

def balanced_sample(file_pool, k, rng, data_root):
    """
    尽量按 task_type 均匀抽样 k 条（round-robin），并保持确定性（由 rng 控制）。
    返回：selected(list)
    """
    if k <= 0:
        return []

    task_to_files = {}
    for fp in file_pool:
        t = extract_task_type(fp, data_root)
        task_to_files.setdefault(t, []).append(fp)

    # 每个 task 内部打乱
    for t in task_to_files:
        rng.shuffle(task_to_files[t])

    # task 列表顺序也由 rng 决定（在 seed 固定时保持确定性）
    tasks = sorted(task_to_files.keys())
    rng.shuffle(tasks)

    selected = []
    # round-robin 抽取，直到凑够 k 或者所有 task 耗尽
    while len(selected) < k:
        progressed = False
        for t in tasks:
            if len(selected) >= k:
                break
            if task_to_files[t]:
                selected.append(task_to_files[t].pop())
                progressed = True
        if not progressed:
            break

    return selected

def get_config_hash(args):
    """
    计算配置的哈希值。
    必须包含所有影响数据集划分的参数，确保任何变动都能触发重新生成。
    """
    config_dict = {
        "data_root": args.data_root,
        "seed": args.seed,
        # 采样策略（新增：均匀按任务类型抽样）
        "sampling_strategy": "balanced_by_task_type_round_robin_v1",
        # 核心：将两种模式的参数都放入字典
        "mode_params": {
            "total_samples": args.total_samples,
            "train_ratio": args.train_ratio,
            "explicit_train_size": args.train_size,  # 关键：纳入显式大小
            "explicit_val_size": args.val_size       # 关键：纳入显式大小
        }
    }
    # 按照 Key 排序转字符串，确保确定性
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode('utf-8')).hexdigest(), config_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=os.path.expanduser("~/.cache/alfworld/json_2.1.1"))
    parser.add_argument("--output_dir", type=str, default="data/verl-agent/text")

    # === 模式 A: 比例采样 (Legacy) ===
    parser.add_argument("--total_samples", type=int, default=-1, help="[Ratio Mode] 总数据量，-1为全量")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="[Ratio Mode] 训练集占比")

    # === 模式 B: 显式指定大小 (New) ===
    parser.add_argument("--train_size", type=int, default=0, help="[Explicit Mode] 显式指定训练集数量")
    parser.add_argument("--val_size", type=int, default=0, help="[Explicit Mode] 显式指定验证集数量")

    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--force", action="store_true", help="强制重新生成")

    args = parser.parse_args()

    # --- 1. 检查缓存 (决定是否跳过) ---
    os.makedirs(args.output_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, "dataset_meta.json")
    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")

    # 计算当前的指纹
    current_hash, current_config = get_config_hash(args)

    need_regenerate = True
    if os.path.exists(meta_path) and os.path.exists(train_path) and not args.force:
        try:
            with open(meta_path, 'r') as f:
                saved_meta = json.load(f)
            # 对比指纹
            if saved_meta.get('config_hash') == current_hash:
                logger.info(f"✅ Config Check: Hash matched ({current_hash[:8]}). Using cached dataset.")
                need_regenerate = False
            else:
                logger.info(
                    f"⚠️  Config Check: Hash mismatch! (Saved: {saved_meta.get('config_hash')[:8]} vs Current: {current_hash[:8]}). Regenerating..."
                )
        except Exception as e:
            logger.warning(f"⚠️  Config Check: Error reading meta ({e}). Regenerating...")
    else:
        logger.info("ℹ️  No cache found or force update. Generating...")

    if not need_regenerate:
        return

    # --- 2. 开始生成数据 ---
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"ALFWorld data not found at {args.data_root}")

    # 扫描文件
    logger.info("Scanning files...")
    raw_files = get_all_game_files(args.data_root, split_names=["train"])
    if not raw_files:
        logger.error("No game files found!")
        return

    # 为确定性：先排序，再用固定 seed 的 rng 控制抽样过程
    raw_files.sort()
    rng = random.Random(args.seed)
    rng.shuffle(raw_files)

    total_available = len(raw_files)

    train_files = []
    test_files = []

    # ================= 核心判定逻辑 =================
    # 判定优先级：如果 train_size 或 val_size 被设置(>0)，则强制进入 Explicit Mode
    if args.train_size > 0 or args.val_size > 0:
        logger.info(f"🔵 Mode: EXPLICIT SIZE (Train: {args.train_size}, Val: {args.val_size})")

        req_train = args.train_size
        req_val = args.val_size

        # 边界检查
        if req_train + req_val > total_available:
            logger.warning(f"⚠️  Requested {req_train + req_val} > Available {total_available}. Truncating Train set first.")
            if req_val > total_available:
                req_val = total_available
                req_train = 0
            else:
                req_train = total_available - req_val

        # 先均匀抽 train，再从剩余里均匀抽 val，保证不重叠
        train_files = balanced_sample(raw_files, req_train, rng, args.data_root)
        remaining = [fp for fp in raw_files if fp not in set(train_files)]
        test_files = balanced_sample(remaining, req_val, rng, args.data_root)

    else:
        logger.info(f"🟣 Mode: RATIO SAMPLING (Total: {args.total_samples}, Ratio: {args.train_ratio})")

        num_to_take = args.total_samples
        if num_to_take == -1 or num_to_take > total_available:
            num_to_take = total_available

        num_train = int(num_to_take * args.train_ratio)
        if num_train == 0 and num_to_take > 0:
            num_train = 1  # 至少保证训练集有1条
        num_val = max(0, num_to_take - num_train)

        # 在全池里做“按任务类型尽量均匀”的抽样：先 train 再 val（不重叠）
        train_files = balanced_sample(raw_files, num_train, rng, args.data_root)
        remaining = [fp for fp in raw_files if fp not in set(train_files)]
        test_files = balanced_sample(remaining, num_val, rng, args.data_root)
    # =================================================

    logger.info(f"Result -> Train: {len(train_files)} | Val: {len(test_files)}")

    # 保存 Parquet
    df_train = pd.DataFrame(process_files(train_files, "train"))
    df_train.to_parquet(train_path)

    if test_files:
        df_test = pd.DataFrame(process_files(test_files, "test"))
        df_test.to_parquet(test_path)
    else:
        # 如果没有测试集，清理旧文件防止混淆
        if os.path.exists(test_path):
            os.remove(test_path)

    # 保存元数据 (更新 Hash)
    meta_info = {
        "config_hash": current_hash,
        "config": current_config,
        "generated_at": pd.Timestamp.now().isoformat(),
        "stats": {
            "train_len": len(train_files),
            "val_len": len(test_files)
        }
    }
    with open(meta_path, 'w') as f:
        json.dump(meta_info, f, indent=2)

    logger.info(f"✅ Dataset generated and saved to {args.output_dir}")

if __name__ == "__main__":
    main()