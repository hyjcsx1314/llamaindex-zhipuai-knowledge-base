import asyncio
import copy
import json
import os
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    MessageRole,
)
from llama_index.core.llms.callbacks import llm_chat_callback
from llama_index.embeddings.zhipuai import ZhipuAIEmbedding
from llama_index.llms.zhipuai import ZhipuAI


ENV_FILE = Path(__file__).resolve().parent / ".env"


def _next_or_none(iterator: Any) -> Any:
    """在线程中安全读取同步流的下一个分片。"""
    try:
        return next(iterator)
    except StopIteration:
        return None


class SafeZhipuAI(ZhipuAI):
    """
    修复 llama-index-llms-zhipuai 0.4.1 的流式工具调用兼容问题。

    主要处理：
    1. 不再因为 delta.content is None 而丢弃 tool_calls 分片；
    2. 跨流式分片累积工具名称和 JSON 参数；
    3. 将 tool_calls=None 统一转换成空列表；
    4. 模型声明 finish_reason=tool_calls、但未解析到工具时，
       返回明确错误文字，避免 UI 显示空白或看起来一直卡住。
    """

    TOOL_CALL_ERROR_MESSAGE: ClassVar[str] = (
        "知识库工具调用失败：模型请求调用工具，"
        "但工具名称或参数未能完整解析。请重新提问。"
    )

    @staticmethod
    def _normalize_tool_calls(response: ChatResponse) -> List[Any]:
        """确保 additional_kwargs.tool_calls 始终是列表。"""
        additional_kwargs = response.message.additional_kwargs or {}
        tool_calls = additional_kwargs.get("tool_calls") or []
        additional_kwargs["tool_calls"] = tool_calls
        response.message.additional_kwargs = additional_kwargs
        return tool_calls

    @staticmethod
    def _extract_finish_reason(response: ChatResponse) -> Optional[str]:
        """兼容 SDK 对象和字典形式的 raw response。"""
        raw = getattr(response, "raw", None)

        choices = getattr(raw, "choices", None)
        if choices:
            return getattr(choices[0], "finish_reason", None)

        if isinstance(raw, dict):
            raw_choices = raw.get("choices") or []
            if raw_choices:
                choice = raw_choices[0]
                if isinstance(choice, dict):
                    return choice.get("finish_reason")

        return None

    @staticmethod
    def _set_response_text(response: ChatResponse, text: str) -> None:
        """把兜底错误写入助手消息，保证 UI 能显示。"""
        try:
            response.message.content = text
            return
        except Exception:
            pass

        blocks = getattr(response.message, "blocks", None)
        if blocks and hasattr(blocks[0], "text"):
            blocks[0].text = text

    @staticmethod
    def _merge_tool_call_chunks(
        accumulated: Dict[int, Any],
        chunks: Any,
    ) -> None:
        """
        按工具调用 index 合并智谱流式分片。

        保留 SDK 原始对象，确保后续 get_tool_calls_from_response()
        仍可通过 tool_call.id、tool_call.function.name 和
        tool_call.function.arguments 访问数据。
        """
        for chunk in chunks or []:
            index = getattr(chunk, "index", 0)
            if index is None:
                index = 0

            if index not in accumulated:
                target = copy.deepcopy(chunk)

                function = getattr(target, "function", None)
                if function is not None:
                    if getattr(function, "name", None) is None:
                        function.name = ""
                    if getattr(function, "arguments", None) is None:
                        function.arguments = ""

                accumulated[index] = target
                continue

            target = accumulated[index]

            chunk_id = getattr(chunk, "id", None)
            if chunk_id:
                target.id = chunk_id

            chunk_type = getattr(chunk, "type", None)
            if chunk_type:
                target.type = chunk_type

            source_function = getattr(chunk, "function", None)
            if source_function is None:
                continue

            target_function = getattr(target, "function", None)
            if target_function is None:
                target.function = copy.deepcopy(source_function)
                target_function = target.function

            source_name = getattr(source_function, "name", None) or ""
            target_name = getattr(target_function, "name", None) or ""

            # 工具名称通常只出现在一个分片；避免重复拼接完整名称。
            if source_name:
                if not target_name:
                    target_function.name = source_name
                elif source_name != target_name and not target_name.endswith(source_name):
                    target_function.name = target_name + source_name

            source_arguments = getattr(source_function, "arguments", None) or ""
            target_arguments = getattr(target_function, "arguments", None) or ""

            if source_arguments:
                target_function.arguments = target_arguments + source_arguments

    @staticmethod
    def _ordered_tool_calls(accumulated: Dict[int, Any]) -> List[Any]:
        return [accumulated[index] for index in sorted(accumulated)]

    def _validate_chat_with_tools_response(
        self,
        response: ChatResponse,
        *args: Any,
        **kwargs: Any,
    ) -> ChatResponse:
        """先归一化 tool_calls，再调用父类校验。"""
        self._normalize_tool_calls(response)
        return super()._validate_chat_with_tools_response(
            response,
            *args,
            **kwargs,
        )

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> List[Any]:
        """
        解析工具调用，并在模型声明要调用工具但参数缺失时返回可读错误。
        """
        tool_calls = self._normalize_tool_calls(response)
        finish_reason = self._extract_finish_reason(response)

        if finish_reason == "tool_calls" and not tool_calls:
            self._set_response_text(
                response,
                self.TOOL_CALL_ERROR_MESSAGE,
            )
            return []

        try:
            return super().get_tool_calls_from_response(
                response,
                error_on_no_tool_call=error_on_no_tool_call,
                **kwargs,
            )
        except (ValueError, TypeError, AttributeError, json.JSONDecodeError):
            if finish_reason == "tool_calls":
                self._set_response_text(
                    response,
                    (
                        "知识库工具调用失败：工具参数格式不完整，"
                        "无法执行检索。请重新提问。"
                    ),
                )
                return []
            raise

    @llm_chat_callback()
    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseGen:
        """同步流式聊天：同时累积文本和工具调用分片。"""
        messages_dict = self._convert_to_llm_messages(messages)

        def gen() -> ChatResponseGen:
            raw_response = self._client.chat.completions.create(
                model=self.model,
                messages=messages_dict,
                stream=True,
                tools=kwargs.get("tools"),
                tool_choice=kwargs.get("tool_choice"),
                stop=kwargs.get("stop"),
                timeout=self.timeout,
                extra_body=self.model_kwargs,
            )

            response_text = ""
            accumulated_tool_calls: Dict[int, Any] = {}
            current_role: Any = MessageRole.ASSISTANT

            for chunk in raw_response:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.delta

                delta_role = getattr(delta, "role", None)
                if delta_role is not None:
                    current_role = delta_role

                content_delta = getattr(delta, "content", None) or ""
                response_text += content_delta

                self._merge_tool_call_chunks(
                    accumulated_tool_calls,
                    getattr(delta, "tool_calls", None),
                )

                yield ChatResponse(
                    message=ChatMessage(
                        content=response_text,
                        role=current_role,
                        additional_kwargs={
                            "tool_calls": self._ordered_tool_calls(
                                accumulated_tool_calls
                            )
                        },
                    ),
                    delta=content_delta,
                    raw=chunk,
                )

        return gen()

    @llm_chat_callback()
    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        """异步流式聊天：在线程中读取智谱同步流并累积工具调用分片。"""
        messages_dict = self._convert_to_llm_messages(messages)

        async def gen() -> ChatResponseAsyncGen:
            raw_response = self._client.chat.completions.create(
                model=self.model,
                messages=messages_dict,
                stream=True,
                tools=kwargs.get("tools"),
                tool_choice=kwargs.get("tool_choice"),
                stop=kwargs.get("stop"),
                timeout=self.timeout,
                extra_body=self.model_kwargs,
            )

            response_text = ""
            accumulated_tool_calls: Dict[int, Any] = {}
            current_role: Any = MessageRole.ASSISTANT

            while True:
                chunk = await asyncio.to_thread(
                    _next_or_none,
                    raw_response,
                )
                if chunk is None:
                    break

                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.delta

                delta_role = getattr(delta, "role", None)
                if delta_role is not None:
                    current_role = delta_role

                content_delta = getattr(delta, "content", None) or ""
                response_text += content_delta

                self._merge_tool_call_chunks(
                    accumulated_tool_calls,
                    getattr(delta, "tool_calls", None),
                )

                yield ChatResponse(
                    message=ChatMessage(
                        content=response_text,
                        role=current_role,
                        additional_kwargs={
                            "tool_calls": self._ordered_tool_calls(
                                accumulated_tool_calls
                            )
                        },
                    ),
                    delta=content_delta,
                    raw=chunk,
                )

        return gen()


def init_settings() -> None:
    """初始化智谱 LLM 和 Embedding。"""
    load_dotenv(ENV_FILE)

    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"没有读取到 ZHIPU_API_KEY，请检查：{ENV_FILE}"
        )

    Settings.llm = SafeZhipuAI(
        model=os.getenv("MODEL", "glm-4-flash"),
        api_key=api_key,
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
    )

    Settings.embed_model = ZhipuAIEmbedding(
        model=os.getenv("EMBEDDING_MODEL", "embedding-3"),
        api_key=api_key,
        dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1024")),
        timeout=int(os.getenv("EMBEDDING_TIMEOUT", "60")),
    )
