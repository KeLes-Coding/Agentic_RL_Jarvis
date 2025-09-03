import base64
import logging
import os
import json
import re
from typing import List, Dict, Any, Tuple

# 动态导入
try:
    import openai
    import httpx  # openai和anthropic需要httpx来设置代理
except ImportError:
    openai, httpx = None, None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import anthropic
except ImportError:
    anthropic = None

from . import prompts


def _fix_json_string(json_string: str) -> str:
    """
    尝试修复可能包含前后缀或 markdown 标记的不规范JSON字符串。
    例如, ` ```json\n{"key": "value"}\n``` `
    """
    # 查找被 ` ``` ` 包围的JSON块
    match = re.search(r"```(json)?\s*(\{.*?\})\s*```", json_string, re.DOTALL)
    if match:
        return match.group(2)
    # 如果没有找到，就假设整个字符串是JSON，只是可能前后有空格
    return json_string.strip()


class LLMClient:
    """
    LLMClient负责与不同的大型语言模型服务进行交互。
    支持多种LLM提供商（如OpenAI, Gemini, Claude），并能处理代理设置。
    """

    def __init__(self, config: Dict[str, Any], proxy_config: Dict[str, Any] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.api_mode = config.get("api_mode", "openai")
        self.fix_json_enabled = config.get("fix_json_enabled", True)

        provider_config = config.get("providers", {}).get(self.api_mode, {})
        self.model = provider_config.get("model")
        self.timeout = provider_config.get("timeout", 120)

        # 处理代理配置
        self.proxies = None
        if proxy_config and proxy_config.get("enabled", False):
            server = (
                proxy_config.get("server")
                or "[http://127.0.0.1:7890](http://127.0.0.1:7890)"
            )
            self.proxies = {"http://": server, "https://": server}
            self.logger.info(f"Using proxy server: {server}")

        self._initialize_client(provider_config)

    def _initialize_client(self, provider_config: Dict[str, Any]):
        """根据api_mode初始化对应的API客户端，并注入代理。"""
        api_key = provider_config.get("api_key") or os.getenv(
            f"{self.api_mode.upper()}_API_KEY"
        )
        if not api_key:
            raise ValueError(f"API key for {self.api_mode} not found.")

        http_client = (
            httpx.Client(proxies=self.proxies) if self.proxies and httpx else None
        )

        if self.api_mode == "openai":
            if not openai:
                raise ImportError("OpenAI SDK not installed.")
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=provider_config.get("base_url"),
                http_client=http_client,
            )
        elif self.api_mode == "claude":
            if not anthropic:
                raise ImportError("Anthropic SDK not installed.")
            self.client = anthropic.Anthropic(api_key=api_key, http_client=http_client)
        elif self.api_mode == "gemini":
            if not genai:
                raise ImportError("Google Generative AI SDK not installed.")
            # Gemini SDK 不直接支持http_client，我们通过环境变量的方式设置代理
            if self.proxies:
                proxy_server = self.proxies.get("https://")  # 通常https代理地址就够了
                os.environ["HTTPS_PROXY"] = proxy_server
                os.environ["HTTP_PROXY"] = proxy_server
                self.logger.warning(
                    "Set system-wide proxy environment variables for Gemini."
                )
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model)
        else:
            raise ValueError(f"Unsupported API mode: {self.api_mode}")

        self.logger.info(
            f"LLM Client initialized in '{self.api_mode}' mode for model '{self.model}'."
        )

    def _prepare_image_payload(self, image_bytes: bytes) -> Dict[str, Any]:
        """将图片字节转换为不同LLM提供商所需的格式。"""
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        media_type = "image/png"
        if self.api_mode == "openai":
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{encoded_image}"},
            }
        elif self.api_mode == "gemini":
            return {"inline_data": {"mime_type": media_type, "data": image_bytes}}
        elif self.api_mode == "claude":
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": encoded_image,
                },
            }
        return {}

    def query(
        self, text_prompt: str, images: List[bytes] = None
    ) -> Tuple[Dict[str, Any], str, Dict[str, int]]:
        """
        向LLM发送查询请求。
        - 整合文本和图片（如果支持VLM）。
        - 调用相应SDK的API。
        - 解析并返回LLM的响应、原始响应和token使用情况。
        """
        self.logger.info("Querying LLM...")
        images = images or []
        content = [{"type": "text", "text": text_prompt}]
        for img_bytes in images:
            content.insert(0, self._prepare_image_payload(img_bytes))

        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        raw_response = ""

        try:
            if self.api_mode == "openai":
                messages = [
                    {"role": "system", "content": prompts.SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ]
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                )
                raw_response = response.choices[0].message.content
                if response.usage:
                    token_usage["prompt_tokens"] = response.usage.prompt_tokens
                    token_usage["completion_tokens"] = response.usage.completion_tokens
                    token_usage["total_tokens"] = response.usage.total_tokens

            elif self.api_mode == "gemini":
                generation_config = genai.types.GenerationConfig(
                    response_mime_type="application/json", temperature=0.1
                )
                response = self.client.generate_content(
                    contents=content,
                    generation_config=generation_config,
                    system_instruction=prompts.SYSTEM_PROMPT,
                    request_options={"timeout": self.timeout},
                )
                raw_response = response.text
                if response.usage_metadata:
                    token_usage["prompt_tokens"] = (
                        response.usage_metadata.prompt_token_count
                    )
                    token_usage["completion_tokens"] = (
                        response.usage_metadata.candidates_token_count
                    )
                    token_usage["total_tokens"] = (
                        response.usage_metadata.total_token_count
                    )

            elif self.api_mode == "claude":
                response = self.client.messages.create(
                    model=self.model,
                    system=prompts.SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=1024,
                    temperature=0.1,
                    timeout=self.timeout,
                )
                raw_response = response.content[0].text
                if response.usage:
                    token_usage["prompt_tokens"] = response.usage.input_tokens
                    token_usage["completion_tokens"] = response.usage.output_tokens
                    token_usage["total_tokens"] = (
                        response.usage.input_tokens + response.usage.output_tokens
                    )
            else:
                raise ValueError(
                    f"Query method not implemented for API mode: {self.api_mode}"
                )

            self.logger.info(f"LLM raw response: {raw_response}")

            try:
                # 第一次尝试直接解析
                parsed_response = json.loads(raw_response)
            except json.JSONDecodeError:
                if self.fix_json_enabled:
                    self.logger.warning("Failed to parse JSON, attempting to fix...")
                    fixed_str = _fix_json_string(raw_response)
                    parsed_response = json.loads(fixed_str)  # 再次尝试解析
                else:
                    raise  # 如果禁用修复，则直接抛出异常

            return parsed_response, raw_response, token_usage

        except Exception as e:
            self.logger.error(
                f"LLM API call or JSON parsing failed: {e}", exc_info=True
            )
            # 即使失败，也返回一个标准的错误结构体和空的token信息
            error_response = {
                "thought": "Error: API call or JSON processing failed.",
                "action": f"error(details='{str(e)}')",
            }
            # 返回错误、原始响应（如果有的话）和空的token
            return error_response, raw_response, token_usage
