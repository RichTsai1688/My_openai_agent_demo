from __future__ import annotations

import asyncio
import os
import json

from dotenv import load_dotenv
from openai import AsyncOpenAI

from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner , handoff
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

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# 建立自訂 tracing
set_tracing_disabled(disabled=True)



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

# @function_tool
# def get_weather(city: str):
#     print(f"[debug] getting weather tool for {city}")
#     _tool_marker_nowait("get_weather")
#     return f"The weather in {city} is sunny."


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
def calculate_sum(values: list[float]) -> float:
    print(f"[debug] calculating sum for: {values}")
    _tool_marker_nowait("calculate_sum")
    return sum(values)


@function_tool
def get_current_time():
    print("[debug] getting current time")
    _tool_marker_nowait(f"get_current_time:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return f"The current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

@function_tool
def get_email_list(keyword: str) -> str:
    """
    Retrieve a formatted list of email contacts that match a given keyword.

    If keyword is empty or contains only whitespace, the function returns all contacts.
    Otherwise, it performs a case-insensitive substring search on contact names and
    returns any matching contacts. If no matches are found, a message indicating
    no matches is returned.

    Args:
        keyword (str): The keyword to search for within contact names. If empty or
            whitespace-only, all contacts will be returned.

        str: A formatted string listing the matching contacts or all contacts, or a
            message stating that no matching contacts were found.
    """
    print(f"[debug] querying email for keyword: {keyword}")
    contacts = {
        "陳永嘉": "Joe081488@gmail.com",
        "林立宬": "lee9207212@gmail.com"
    }
    if not keyword.strip():
        # If no keyword, return all contacts
        result = "\n".join([f"{name}: {email}" for name, email in contacts.items()])
        _tool_marker_nowait(f"Get email from all contacts:\n{result}")
        print(f"[debug] all contacts:\n{result}", flush=True)
        return f"All contacts:\n{result}"
    else:
        # Search for contacts whose name contains the keyword (case-insensitive)
        matches = {name: email for name, email in contacts.items() if keyword.lower() in name.lower()}
        if matches:
            result = "\n".join([f"{name}: {email}" for name, email in matches.items()])
            _tool_marker_nowait(f"Get email from {result}")
            print(f"[debug] matching contacts:\n{result}", flush=True)
            return f"Matching contacts:\n{result}"
        else:
            _tool_marker_nowait(f"No matching contacts for keyword: {keyword}")
            print(f"[debug] no matching contacts for keyword: {keyword}", flush=True)
            return "No matching contacts found"

@function_tool
def send_email(to: str, subject: str, body: str) -> str:
    """
    Send an email to the specified recipient(s) with the given subject and body.

    Args:
        to (str): Comma or semicolon-separated email addresses of the recipient(s).
        subject (str): Subject line of the email.
        body (str): Main content of the email.
    """
    print(f"[debug] sending email to: {to}, subject: {subject}", flush=True)
    _tool_marker_nowait("send_email")

    gmail_user = os.getenv("GMAIL_USER")
    gmail_app_password = os.getenv("GMAIL_APP_PASSWORD")  # App Password required for Gmail

    if not gmail_user or not gmail_app_password:
        return "Error: Missing GMAIL_USER or GMAIL_APP_PASSWORD environment variables."

    # Support multiple recipients separated by comma/semicolon
    recipients = [r.strip() for r in to.replace(";", ",").split(",") if r.strip()]
    if not recipients:
        return "Error: No valid recipient."

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = gmail_user
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context, timeout=20) as server:
            server.login(gmail_user, gmail_app_password)
            server.send_message(msg)
        return f"Email sent to {', '.join(recipients)} with subject '{subject}'."
    except smtplib.SMTPAuthenticationError as e:
        detail = getattr(e, "smtp_error", b"").decode(errors="ignore") if hasattr(e, "smtp_error") else str(e)
        return f"Error: SMTP authentication failed. {detail}"
    except Exception as e:
        return f"Error: Failed to send email. {e}"

# ---- local demo (optional) ----
async def main():
    agent = Agent(
        name="Assistant",
        instructions="""You only respond in 繁體中文. 查詢前，請先確認目前時間。
        數學相關符號與數學式請一律包含 $$...$$ 來輸出。
        請記得使用者輸入的步驟，請拆分todos逐一處理，確認每個步驟皆完成。
        如果是查詢請用google_search_pse_with_contents工具來查詢並回覆結果.
        如果是使用者有需要寄email的需求時, 你可以呼叫工具來達成寄信的目的.
        If user request email, you can call tools for email purpose.
        """,
        model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
        tools=[get_email_list, send_email, get_current_time, google_search_pse_with_contents],
        #handoffs=[handoff(emailagent, "當使用者有需要寄email的需求時")],
    )

    result = Runner.run_streamed(
        agent,
        input="請問現在有哪些工具可以使用",
        max_turns=15,
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)


# ---- OpenAI-compatible FastAPI server ----
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import os
import smtplib
import ssl
from email.message import EmailMessage


async def stream_agent_response(user_input: str, max_turns: int = 30):
    """Async generator yielding both model deltas and tool markers (<tools>name</tools>)."""
    
    emailagent = Agent(
        name="Email Assistant",
        instructions="""請訊息送件人的email, 然後將內容送給對方，副本給自己。
        """,
        model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client), 
        tools=[get_email_list, send_email],
    )

    # https://openai.github.io/openai-agents-python/handoffs/
    # 這裡的 handoff 是在主 Agent 裡面呼叫另一個 Agent
    # 當主 Agent 判斷使用者有需要寄email的需求時,
    #     Handoff inputs
    # In certain situations, you want the LLM to provide some data when it calls a handoff. For example, imagine a handoff to an "Escalation agent". You might want a reason to be provided, so you can log it.


    # from pydantic import BaseModel

    # from agents import Agent, handoff, RunContextWrapper

    # class EscalationData(BaseModel):
    #     reason: str

    # async def on_handoff(ctx: RunContextWrapper[None], input_data: EscalationData):
    #     print(f"Escalation agent called with reason: {input_data.reason}")

    # agent = Agent(name="Escalation agent")

    # handoff_obj = handoff(
    #     agent=agent,
    #     on_handoff=on_handoff,
    #     input_type=EscalationData,
    # )
    agent = Agent(
        name="Entrance Assistant",
        instructions="""You only respond in 繁體中文. 查詢前，請先確認目前時間。
        數學相關符號與數學式請一律包含 $$...$$ 來輸出。
        請記得使用者輸入的步驟，請拆分todos逐一處理，確認每個步驟皆完成。
        如果是查詢請用google_search_pse_with_contents工具來查詢並回覆結果.
        如果是使用者有需要寄email的需求時, 你可以handoffs給emailagent來達成寄信的目的.
        """,
        model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
        tools=[get_current_time, google_search_pse_with_contents],
        handoffs=[emailagent],
    )

    

    out_q: asyncio.Queue = asyncio.Queue()
    token = _tool_stream_queue.set(out_q)

    async def produce_model_stream():
        result = Runner.run_streamed(
            agent,
            input=user_input,
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
    uvicorn.run("multi_openai_agent:app", host="::", port=3002, reload=False, log_level="info")
    # asyncio.run(main())
    # get_email_list("永嘉")