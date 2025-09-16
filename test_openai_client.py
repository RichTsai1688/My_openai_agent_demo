#!/usr/bin/env python3
import os
import argparse
from openai import OpenAI
import sys
import json


def get_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def _print_parsed_tools_in_text(s: str) -> None:
    """Parse and print any <tools>...</tools> blocks with minimal info (tool name only)."""
    start_tag, end_tag = "<tools>", "</tools>"
    i = 0
    while True:
        start = s.find(start_tag, i)
        if start == -1:
            # print remaining normal text
            print(s[i:], end="", flush=True)
            break
        # print text before the tools block
        print(s[i:start], end="", flush=True)
        end = s.find(end_tag, start)
        if end == -1:
            # no closing tag; print remainder as normal text
            print(s[start:], end="", flush=True)
            break
        inner = s[start + len(start_tag): end]
        # minimal marker carries only tool name
        tool_name = inner.strip()
        if tool_name:
            print(file=sys.stderr)
            print(f"[TOOLS] {tool_name}", file=sys.stderr, flush=True)
        # do not echo the marker on stdout
        i = end + len(end_tag)


def run_non_stream(client: OpenAI, model: str, prompt: str) -> None:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    content = resp.choices[0].message.content or ""
    _print_parsed_tools_in_text(content)
    print()


def run_stream(client: OpenAI, model: str, prompt: str) -> None:
    # Keep a small rolling buffer to handle tags spanning chunk boundaries
    start_tag, end_tag = "<tools>", "</tools>"
    tail_keep = len(start_tag) - 1  # 6
    buf = ""

    for chunk in client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ):
        delta = chunk.choices[0].delta
        if delta is not None and getattr(delta, "content", None):
            buf += delta.content
            while True:
                start = buf.find(start_tag)
                if start == -1:
                    # no tools start tag; flush most of the buffer but keep tail to catch split tags
                    if len(buf) > tail_keep:
                        print(buf[:-tail_keep], end="", flush=True)
                        buf = buf[-tail_keep:]
                    break
                # print normal text before tools block
                if start > 0:
                    print(buf[:start], end="", flush=True)
                end = buf.find(end_tag, start)
                if end == -1:
                    # incomplete tools block; keep from start and wait for next chunk
                    buf = buf[start:]
                    break
                inner = buf[start + len(start_tag): end]
                tool_name = inner.strip()
                if tool_name:
                    print(file=sys.stderr)
                    print(f"[TOOLS] {tool_name}", file=sys.stderr, flush=True)
                buf = buf[end + len(end_tag):]
    # flush remaining buffer
    if buf:
        print(buf, end="", flush=True)
    print()


def main():
    parser = argparse.ArgumentParser(description="Test OpenAI-compatible server")
    parser.add_argument("--prompt", default="brad pitt有什麼新電影，台中上映的場次?", help="User message")
    parser.add_argument("--model", default=os.getenv("TEST_MODEL", "EXAMPLE_MODEL_NAME"), help="Model name")
    parser.add_argument("--base-url", dest="base_url", default=os.getenv("TEST_BASE_URL", "http://localhost:3001/v1"), help="Server base url")
    parser.add_argument("--api-key", dest="api_key", default=os.getenv("TEST_API_KEY", "dummy"), help="API key (dummy accepted)")
    parser.add_argument("--stream", action="store_true", help="Use streaming response")
    args = parser.parse_args()

    client = get_client(args.base_url, args.api_key)

    if args.stream:
        run_stream(client, args.model, args.prompt)
    else:
        run_non_stream(client, args.model, args.prompt)


if __name__ == "__main__":
    main()
