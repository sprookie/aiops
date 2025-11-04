import os
from typing import Optional

from langchain_openai import ChatOpenAI


DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


def get_llm(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = DEFAULT_DEEPSEEK_MODEL,
    temperature: float = 0.2,
):
    """构建DeepSeek LLM（OpenAI兼容），用于LangChain v1.0。

    优先读取环境变量：
      - DEEPSEEK_API_KEY 或 OPENAI_API_KEY
      - OPENAI_BASE_URL 或 DEEPSEEK_API_BASE

    若未配置，将回退到用户提供的API Key与默认Base URL。
    """
    api_key = (
        api_key
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "缺少API Key。请在环境变量或.env中设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY。"
        )
    base_url = (
        base_url
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("DEEPSEEK_API_BASE")
        or DEFAULT_BASE_URL
    )

    # ChatOpenAI在langchain-openai中支持openai_api_key与openai_api_base参数
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base=base_url,
    )
    return llm