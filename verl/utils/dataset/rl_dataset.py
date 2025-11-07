# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os
import re
from collections import defaultdict
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

# ❗️ 新增：导入 sentence_transformers，并处理导入失败
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, *dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.
    - ❗️ (新增) Optionally vectorizes prompts for similarity search.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get('use_shm', False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False

        # ❗️ 新增：初始化 prompt 向量化器
        self.vectorize_prompts = True  # 默认开启
        self.vectorizer_model_name = "all-MiniLM-L6-v2" # 默认一个轻量级模型
        self.vectorizer = None
        
        if self.vectorize_prompts:
            if _SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    # all-MiniLM-L6-v2 是一个常用的轻量级句向量模型
                    self.vectorizer = SentenceTransformer(self.vectorizer_model_name) 
                    logger.info(f"成功初始化 Prompt 向量化器: {self.vectorizer_model_name}")
                except Exception as e:
                    logger.error(f"加载 SentenceTransformer 模型 '{self.vectorizer_model_name}' 失败: {e}")
                    self.vectorize_prompts = False
            else:
                logger.error(
                    "未找到 'sentence-transformers' 库。"
                    "请通过 `pip install sentence-transformers` 安装以启用 prompt 向量化。"
                )
                self.vectorize_prompts = False # 自动禁用该功能

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe.filter(
                lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataframe)}")

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        # 从HuggingFace dataset中获取原始数据行
        original_row: dict = self.dataframe[item]

        # ❗️ 关键修复：创建一个副本，以防后续操作修改原始字典。
        row_dict = original_row.copy()
        
        # 最终要返回的样本，我们从这里开始构建
        final_item = {}

        # ❗️ [CCAPO 修正]：在此处，从 original_row 向量化核心任务
        if self.vectorizer is not None:
            try:
                # 根据你的 parquet 样本: prompt 是 [{'content': '...', 'role': 'user'}]
                # 我们提取第一个 'content' 作为任务字符串
                task_content_string = original_row[self.prompt_key][0]['content']
                prompt_vector = self.vectorizer.encode(task_content_string, convert_to_tensor=True)
                final_item["prompt_vector"] = prompt_vector
            except Exception as e:
                logger.warning(f"Failed to vectorize prompt for item {item} (task: {original_row[self.prompt_key]}): {e}")
                # ❗️ [FIX] 创建一个零向量而不是 None，以确保 collate_fn 正常工作
                try:
                    dim = self.vectorizer.get_sentence_embedding_dimension()
                    final_item["prompt_vector"] = torch.zeros(dim, dtype=torch.float32)
                except Exception as e2:
                    logger.error(f"无法获取向量器维度: {e2}. 将 prompt_vector 设置为 None.")
                    # 最终回退（如果维度都拿不到），但这在 collate_fn 中仍可能失败
                    final_item["prompt_vector"] = None
        
        # (如果 self.vectorizer is None, 则 final_item 中不会有 "prompt_vector" 键)

        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row_dict:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")
            
            # ❗️ 关键修复：将多模态相关数据添加到 final_item
            final_item["multi_modal_data"] = multi_modal_data
            final_item["multi_modal_inputs"] = dict(model_inputs)
            final_item["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        # ❗️ 关键修复：将张量数据添加到 final_item
        final_item["input_ids"] = input_ids[0]
        final_item["attention_mask"] = attention_mask[0]
        final_item["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        final_item["raw_prompt_ids"] = raw_prompt_ids
        
        # ❗️ 关键修复：从原始数据行 `original_row` 中安全地复制所有非张量元数据
        for key, value in original_row.items():
            if key not in final_item and not isinstance(value, (torch.Tensor, np.ndarray)):
                 # 特别是 prompt，我们使用处理过的 messages
                if key == self.prompt_key:
                    if self.return_raw_chat:
                        final_item["raw_prompt"] = messages
                else:
                    final_item[key] = value

        # get prompts with chat template
        if self.return_full_prompt:
            final_item["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = original_row.get("extra_info", {}).get("index", 0)
        tools_kwargs = original_row.get("extra_info", {}).get("tools_kwargs", {})
        need_tools_kwargs = original_row.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, original_row["data_source"])
        
        # 确保 index 和 tools_kwargs 也被添加
        final_item["index"] = index
        final_item["tools_kwargs"] = tools_kwargs
        
        # print("final_item keys:", final_item.keys())
        # print(f"Processed item {item}: index={final_item['index']}, input_ids shape={final_item['input_ids'].shape}")
        return final_item

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            
            # ❗️ 新增：防止 vectorizer 实例被序列化
            if "vectorizer" in state:
                del state["vectorizer"]
                
            return state

        return self.__dict__.copy()

    # ❗️ 新增：__setstate__ 用于在反序列化时重新加载 vectorizer
    def __setstate__(self, state):
        self.__dict__.update(state)
        # 重新初始化 vectorizer
        if self.vectorize_prompts:
            try:
                from sentence_transformers import SentenceTransformer
                self.vectorizer = SentenceTransformer(self.vectorizer_model_name)
                logger.info(f"成功从 state 重新初始化 Prompt 向量化器: {self.vectorizer_model_name}")
            except ImportError:
                logger.error(
                    "未找到 'sentence-transformers' 库。"
                    "请通过 `pip install sentence-transformers` 安装以启用 prompt 向量化。"
                )
                self.vectorizer = None
                self.vectorize_prompts = False
        else:
            self.vectorizer = None