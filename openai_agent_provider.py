from __future__ import annotations

import asyncio
import os
import json

from dotenv import load_dotenv
from openai import AsyncOpenAI

from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner
from datetime import datetime
from contextvars import ContextVar
from typing import Optional

# 載入 .env 檔案
load_dotenv()

from agents import (
    Agent,
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    function_tool,
    set_tracing_disabled,
)

BASE_URL = os.getenv("EXAMPLE_BASE_URL") or ""
API_KEY = os.getenv("EXAMPLE_API_KEY") or ""
MODEL_NAME = os.getenv("EXAMPLE_MODEL_NAME") or ""

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError(
        "Please set EXAMPLE_BASE_URL, EXAMPLE_API_KEY, EXAMPLE_MODEL_NAME via env var or code."
    )

# 建立自訂 OpenAI client 與 provider
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_tracing_disabled(disabled=True)


class CustomModelProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        return OpenAIChatCompletionsModel(model=model_name or MODEL_NAME, openai_client=client)


CUSTOM_MODEL_PROVIDER = CustomModelProvider()

# 用於在單一請求生命週期中，把工具呼叫事件送進共用的輸出佇列，讓 SSE 串流能一併輸出
_tool_stream_queue: ContextVar[Optional[asyncio.Queue]] = ContextVar("tool_stream_queue", default=None)


def _tool_marker_nowait(tool_name: str) -> None:
    """Emit minimal tool marker: <tools>{tool_name}</tools> (non-blocking)."""
    q: Optional[asyncio.Queue] = _tool_stream_queue.get()
    if q is not None:
        try:
            q.put_nowait(f"<tools>{tool_name}</tools>")
        except Exception:
            pass


async def _tool_marker_async(tool_name: str) -> None:
    """Emit minimal tool marker: <tools>{tool_name}</tools> (async)."""
    q: Optional[asyncio.Queue] = _tool_stream_queue.get()
    if q is not None:
        await q.put(f"<tools>{tool_name}</tools>")


# ---- function tools ----

@function_tool
def get_weather(city: str):
    print(f"[debug] getting weather tool for {city}")
    _tool_marker_nowait("get_weather")
    return f"The weather in {city} is sunny."


@function_tool
async def google_search_pse_with_contents(q: str):
    """以 Google CSE 搜尋並回傳結果摘要 (async 版，避免 event loop 重入錯誤)。"""
    print(f"[debug] performing Google search with contents for: {q}", flush=True)
    await _tool_marker_async(f"google_search_pse_with_contents: {q}")
    try:
        from google_search_pse import Tools  # type: ignore
    except Exception as e:
        msg = f"google_search_pse module not available: {e}"
        print(f"[error] {msg}", flush=True)
        return [{"error": msg}]

    tools = Tools()
    tools.valves.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    tools.valves.GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

    if not tools.valves.GOOGLE_API_KEY or not tools.valves.GOOGLE_CSE_ID:
        msg = "Missing GOOGLE_API_KEY / GOOGLE_CSE_ID env vars"
        print(f"[error] {msg}", flush=True)
        return [{"error": msg}]

    tools.valves.ENGINE_RETURNED_PAGES_NO = 5
    tools.valves.SCRAPPED_PAGES_NO = 5
    tools.valves.PAGE_CONTENT_WORDS_LIMIT = 800

    try:
        raw = await tools.search_web(q)
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("Unexpected search_web return format")
    except Exception as e:
        print(f"[error] search failed: {e}", flush=True)
        return [{"error": str(e)}]

    print(
        f"共 {len(data)} 筆內容，第一筆字數: {len(data[0]['content']) if data else 0}",
        flush=True,
    )
    result_summary = []
    for item in data:
        title = item.get('title')
        url = item.get('url')
        snippet = item.get('snippet')
        content = item.get('content') or ""
        preview = content[:200]
        print(f"Title: {title}", flush=True)
        print(f"URL: {url}", flush=True)
        print(f"Snippet: {snippet}", flush=True)
        print(f"Content: {preview}...", flush=True)
        print("-" * 80, flush=True)
        result_summary.append({
            "title": title,
            "url": url,
            "snippet": snippet,
            "content_preview": preview,
        })
    return result_summary


@function_tool
def google_search(query: str):
    print(f"[debug] performing Google search for: {query}")
    _tool_marker_nowait("google_search")
    from googlesearch import search  # import library

    results = search(query, advanced=True)  # 執行程式

    answer = ""
    url = ""
    for result in results:  # 顯示
        answer += ", " + result.description
        url += ", " + result.url
    return f"Search results for '{query}' , information are :{answer} and corresponding to {url}"


@function_tool
def calculate_sum(values: list[float]) -> float:
    print(f"[debug] calculating sum for: {values}")
    _tool_marker_nowait("calculate_sum")
    return sum(values)


@function_tool
def get_current_time():
    print("[debug] getting current time")
    _tool_marker_nowait(f"get_current_time:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return f"The current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


# ---- local demo (optional) ----
async def main():
    agent = Agent(
        name="Assistant",
        instructions="""You only respond in 繁體中文. 查詢前，請先確認目前時間。
        Must call tools for calculations or to get real-time information.""",
        tools=[get_current_time, google_search_pse_with_contents, calculate_sum],
    )

    result = Runner.run_streamed(
        agent,
        input="brad pitt有什麼新電影，台中上映的場次?",
        run_config=RunConfig(model_provider=CUSTOM_MODEL_PROVIDER),
        max_turns=30,
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)


# ---- OpenAI-compatible FastAPI server ----
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn


async def stream_agent_response(user_input: str, max_turns: int = 30):
    """Async generator yielding both model deltas and tool markers (<tools>name</tools>)."""
    agent = Agent(
        name="Assistant",
        instructions="""You only respond in 繁體中文. 查詢前，請先確認目前時間。
        數學相關符號與數學式請一律包含 $$...$$ 來輸出。
        對於複雜加總請務必使用 calculate_sum 工具計算結果後再回覆。
        """,
        tools=[get_current_time, google_search_pse_with_contents, calculate_sum],
    )

    out_q: asyncio.Queue = asyncio.Queue()
    token = _tool_stream_queue.set(out_q)

    async def produce_model_stream():
        result = Runner.run_streamed(
            agent,
            input=user_input,
            run_config=RunConfig(model_provider=CUSTOM_MODEL_PROVIDER),
            max_turns=max_turns,
        )
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                await out_q.put(event.data.delta)
        await out_q.put(None)

    try:
        producer_task = asyncio.create_task(produce_model_stream())
        while True:
            item = await out_q.get()
            if item is None:
                break
            yield item
        await producer_task
    finally:
        _tool_stream_queue.reset(token)


app = FastAPI()

# Allow cross-origin requests (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    messages = body.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return JSONResponse({"error": "Missing messages"}, status_code=400)

    user_message = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    if not user_message:
        return JSONResponse({"error": "Missing user message"}, status_code=400)

    stream = bool(body.get("stream", False))
    max_turns = int(body.get("max_turns", 15))
    model_name = body.get("model") or MODEL_NAME

    if stream:
        async def sse_generator():
            pre = {
                "id": "cmpl-stream",
                "object": "chat.completion.chunk",
                "model": model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(pre, ensure_ascii=False)}\n\n"

            async for delta in stream_agent_response(user_message, max_turns=max_turns):
                payload = {
                    "id": "cmpl-stream",
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [
                        {"index": 0, "delta": {"content": delta}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            done = {
                "id": "cmpl-stream",
                "object": "chat.completion.chunk",
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    # non-stream: aggregate into one message
    full = []
    async for delta in stream_agent_response(user_message, max_turns=max_turns):
        full.append(delta)
    content = "".join(full)

    response = {
        "id": "cmpl-nonstream",
        "object": "chat.completion",
        "model": model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }
    return JSONResponse(response)


if __name__ == "__main__":
    # Use IPv6 host '::' so localhost resolving to ::1 works; also accepts IPv4 on most systems
    uvicorn.run("openai_agent_provider:app", host="::", port=3001, reload=False, log_level="info")