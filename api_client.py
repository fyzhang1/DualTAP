import os
import base64
import io
import time


class APIClient:
    def __init__(self, api_type="openai", api_key=None, model_name=None, base_url=None):
        self.api_type = (api_type or "openai").lower()
        self.api_key = api_key or os.environ.get(f"{self._env_key_prefix()}_API_KEY")
        self.model_name = model_name
        self.base_url = base_url

        if not self.api_key:
            raise ValueError(
                f"please set {self._env_key_prefix()}_API_KEY"
            )

        if self.api_type in ("openai", "openrouter", "qwen"):
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("pip install openai")

            client_kwargs = {"api_key": self.api_key}
            if self.api_type == "openrouter":

                client_kwargs["base_url"] = self.base_url or os.environ.get(
                    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
                )

                self.model_name = model_name or os.environ.get(
                    "OPENROUTER_MODEL", "openrouter/auto"
                )
            elif self.api_type == "qwen":

                client_kwargs["base_url"] = self.base_url or os.environ.get(
                    "OPENAI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
                )

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
                raise ImportError("pip install google-generativeai")
            genai.configure(api_key=self.api_key)
            self._client_kind = "gemini"
            self.client = genai
            self.model_name = model_name or os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")

        else:
            raise ValueError(f"{self.api_type}")

    def _env_key_prefix(self):
        if self.api_type == "openrouter":
            return "OPENROUTER"
        if self.api_type == "qwen":

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
                    print(f"please retry{attempt + 1}/{max_retries}: {e}")
                    time.sleep(2 ** attempt)
        print(f"{last_err}")
        return "Error"

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


