# OpenAI 兼容串流伺服器（簡易版）

## 1) 設定 .env

```
EXAMPLE_BASE_URL=...      # LLM 伺服器位址
EXAMPLE_API_KEY=...       # LLM API Key
EXAMPLE_MODEL_NAME=...    # 模型名稱
# 若使用 Google 搜尋工具，需提供：
GOOGLE_API_KEY=...
GOOGLE_CSE_ID=...
```

## 2) 安裝

```
pip install -r requirements.txt
```

## 3) 啟動伺服器

```
python openai_agent_provider.py
```

預設端點：`http://localhost:3001/v1/chat/completions`

## 4) 測試

- 非串流
```
curl -s http://localhost:3001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model":"EXAMPLE_MODEL_NAME",
        "messages":[{"role":"user","content":"台中天氣如何？"}],
        "stream": false
      }' | jq .
```

- 串流（SSE）
```
curl -N http://localhost:3001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model":"EXAMPLE_MODEL_NAME",
        "messages":[{"role":"user","content":"brad pitt有什麼新電影，台中上映的場次?"}],
        "stream": true
      }'
```

- Python 測試（已支援解析工具標記）
```
python test_openai_client.py --stream --prompt "台中天氣如何？"
```

說明：
- 串流時除了模型文字，還會穿插最小化工具標記：`<tools>tool_name</tools>`。
- 已啟用 CORS；若 localhost 解析到 ::1 連不上，改用 `http://127.0.0.1:3001/v1`。
