import os
import base64
import io
import time


class APIClient:
    """
    统一的多模态 API 客户端，支持：
    - openai（如 gpt-4o 系列）
    - gemini（Google Generative AI）
    - openrouter（OpenAI 兼容协议，需设置 base_url）
    """

    def __init__(self, api_type="openai", api_key=None, model_name=None, base_url=None):
        self.api_type = (api_type or "openai").lower()
        self.api_key = api_key or os.environ.get(f"{self._env_key_prefix()}_API_KEY")
        self.model_name = model_name
        self.base_url = base_url

        if not self.api_key:
            raise ValueError(
                f"请设置 {self._env_key_prefix()}_API_KEY 环境变量或通过参数传入 api_key"
            )

        if self.api_type in ("openai", "openrouter", "qwen"):
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("请安装 openai 库: pip install openai")

            client_kwargs = {"api_key": self.api_key}
            if self.api_type == "openrouter":
                # OpenRouter 使用 OpenAI 兼容接口，需要设置 base_url
                client_kwargs["base_url"] = self.base_url or os.environ.get(
                    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
                )
                # 兼容默认模型名（可在 CLI 传入覆盖）
                self.model_name = model_name or os.environ.get(
                    "OPENROUTER_MODEL", "openrouter/auto"
                )
            elif self.api_type == "qwen":
                # 阿里云 DashScope 的 OpenAI 兼容模式
                client_kwargs["base_url"] = self.base_url or os.environ.get(
                    "OPENAI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
                )
                # 默认多模态视觉模型
                self.model_name = model_name or os.environ.get(
                    "QWEN_API_MODEL", "qwen-vl-max"
                )
            else:
                self.model_name = model_name or os.environ.get("OPENAI_MODEL", "gpt-4o")

            self._client_kind = "openai"
            self.client = OpenAI(**client_kwargs)

        elif self.api_type == "gemini":
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError("请安装 google-generativeai 库: pip install google-generativeai")
            genai.configure(api_key=self.api_key)
            self._client_kind = "gemini"
            self.client = genai
            self.model_name = model_name or os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")

        else:
            raise ValueError(f"不支持的 API 类型: {self.api_type}")

    def _env_key_prefix(self):
        if self.api_type == "openrouter":
            return "OPENROUTER"
        if self.api_type == "qwen":
            # DashScope 官方使用 DASHSCOPE_API_KEY
            return "DASHSCOPE"
        return self.api_type.upper()

    def _pil_to_base64(self, image_pil):
        buf = io.BytesIO()
        image_pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def query(self, image_pil, question, max_retries=3):
        last_err = None
        for attempt in range(max_retries):
            try:
                if self._client_kind == "openai":
                    return self._query_openai_compatible(image_pil, question)
                elif self._client_kind == "gemini":
                    return self._query_gemini(image_pil, question)
            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    print(f"API调用失败，重试 {attempt + 1}/{max_retries}: {e}")
                    time.sleep(2 ** attempt)
        print(f"API调用最终失败: {last_err}")
        return "Error: API调用失败"

    def _query_openai_compatible(self, image_pil, question):
        base64_image = self._pil_to_base64(image_pil)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"},
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content

    def _query_gemini(self, image_pil, question):
        model = self.client.GenerativeModel(self.model_name)
        response = model.generate_content([question, image_pil])
        return getattr(response, "text", "") or ""


