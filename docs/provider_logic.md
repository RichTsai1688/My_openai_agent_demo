# Provider 與串流邏輯說明（簡要）

本文說明本專案中「自訂模型 Provider」、「Agent 串流」與「工具標記」的運作方式。

## 1) CustomModelProvider

- 檔案：`openai_agent_provider.py`
- 重點：以 `AsyncOpenAI` + `OpenAIChatCompletionsModel` 封裝成 `ModelProvider`，讓 `Runner` 可以透過統一介面呼叫任意 LLM 後端。

流程：
1. 由 `.env` 讀取 `EXAMPLE_BASE_URL`、`EXAMPLE_API_KEY`、`EXAMPLE_MODEL_NAME`。
2. 建立 `AsyncOpenAI(base_url, api_key)` client。
3. `CustomModelProvider.get_model(model_name)` 回傳 `OpenAIChatCompletionsModel`，會使用前述 client 與模型名稱。
4. 執行時把 `RunConfig(model_provider=CUSTOM_MODEL_PROVIDER)` 傳給 `Runner.run_streamed(...)`。

好處：
- 你可以替換成不同的 LLM 服務（OpenAI 相容 API 皆可），不需改 Agent/工具邏輯。

## 2) 串流執行（Runner.run_streamed）

- `stream_agent_response(user_input, max_turns)` 用 `Runner.run_streamed(agent, ...)` 啟動對話。
- 透過 `async for event in result.stream_events():` 收到模型回傳的片段（`ResponseTextDeltaEvent`）。
- 這些片段（delta）會被放進一個 `asyncio.Queue`，供 FastAPI SSE 端點持續輸出。

## 3) 工具標記（<tools>tool_name</tools>）

- 在每個 `@function_tool` 的入口（工具開始執行時）呼叫 `_tool_marker_nowait("tool_name")` 或 `_tool_marker_async("tool_name")`。
- 這些標記以純文字插入同一條輸出串流中，格式：`<tools>tool_name</tools>`。
- 模型文字與工具標記共享同一個 `Queue`，因此前端或客戶端可以在讀取串流時同時知道「此刻 Agent 正在使用哪個工具」。
- 工具的「回傳值」仍回到 Agent 內部邏輯使用，不會因為標記而改變；標記只是觀測用途。

## 4) FastAPI OpenAI 兼容端點

- 端點：`POST /v1/chat/completions`
- 參數：遵循 OpenAI 格式（`messages`、`model`、`stream` 等）。
- 若 `stream: true`，回傳 SSE，chunk 會包含：
  - 模型文字：一般字串內容
  - 工具標記：`<tools>tool_name</tools>`（可由前端客戶端解析）

## 5) 如何切換/擴充 Provider

- 切換模型：
  - 呼叫 API 時改 `model`，或修改 `.env` 的 `EXAMPLE_MODEL_NAME`。
- 切換 Provider（不同 LLM 服務）：
  - 修改 `.env` 的 `EXAMPLE_BASE_URL` 與 `EXAMPLE_API_KEY`。
  - 如需更動認證或路由行為，可在 `CustomModelProvider` 或 `AsyncOpenAI` 初始化時調整。

## 6) 常見問題

- 串流出現連線問題（::1）：
  - 伺服器已綁定 `::`（IPv6/IPv4），前端若仍失敗，改用 `http://127.0.0.1:3001/v1`。
- 想看工具的更多資料？
  - 目前僅輸出工具名稱最小化標記；若要擴充，可在工具內新增額外標記，但建議保持精簡以利前端處理。
