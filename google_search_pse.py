"""
title: Web Search using Google Custom Search Engine
Last Updated: 20250415
"""

import os
import requests
from datetime import datetime
import json
from requests import get
from bs4 import BeautifulSoup
import concurrent.futures
from html.parser import HTMLParser
from urllib.parse import urlparse, urljoin
import re
import unicodedata
from pydantic import BaseModel, Field
import asyncio
from typing import Callable, Any


class HelpFunctions:
    def __init__(self):
        pass

    def get_base_url(self, url):
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        return base_url

    def generate_excerpt(self, content, max_length=200):
        return content[:max_length] + "..." if len(content) > max_length else content

    def format_text(self, original_text):
        soup = BeautifulSoup(original_text, "html.parser")
        formatted_text = soup.get_text(separator=" ", strip=True)
        formatted_text = unicodedata.normalize("NFKC", formatted_text)
        formatted_text = re.sub(r"\s+", " ", formatted_text)
        formatted_text = formatted_text.strip()
        formatted_text = self.remove_emojis(formatted_text)
        return formatted_text

    def remove_emojis(self, text):
        return "".join(c for c in text if not unicodedata.category(c).startswith("So"))

    def process_search_result(self, result, valves):
        title_site = self.remove_emojis(result["title"])
        url_site = result[
            "link"
        ]  # Changed from "url" to "link" (Google CSE field name)
        snippet = result.get("snippet", "")  # Changed from "content" to "snippet"

        # Check if the website is in the ignored list
        if valves.IGNORED_WEBSITES:
            base_url = self.get_base_url(url_site)
            if any(
                ignored_site.strip() in base_url
                for ignored_site in valves.IGNORED_WEBSITES.split(",")
            ):
                return None

        try:
            response_site = requests.get(url_site, timeout=20)
            response_site.raise_for_status()
            html_content = response_site.text
            soup = BeautifulSoup(html_content, "html.parser")
            content_site = self.format_text(soup.get_text(separator=" ", strip=True))
            truncated_content = self.truncate_to_n_words(
                content_site, valves.PAGE_CONTENT_WORDS_LIMIT
            )
            return {
                "title": title_site,
                "url": url_site,
                "content": truncated_content,
                "snippet": self.remove_emojis(snippet),
            }
        except requests.exceptions.RequestException as e:
            return None

    def truncate_to_n_words(self, text, token_limit):
        tokens = text.split()
        truncated_tokens = tokens[:token_limit]
        return " ".join(truncated_tokens)


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def emit(self, description="Unknown State", status="in_progress", done=False):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )


class Tools:
    class Valves(BaseModel):
        # Changed from SEARXNG to Google CSE configuration
        GOOGLE_API_KEY: str = Field(
            default="",
            description="Your Google API key for Custom Search Engine",
        )
        GOOGLE_CSE_ID: str = Field(
            default="",
            description="Your Google Custom Search Engine ID",
        )
        IGNORED_WEBSITES: str = Field(
            default="",
            description="Comma-separated list of websites to ignore",
        )
        ENGINE_RETURNED_PAGES_NO: int = Field(
            default=10,
            description="The number of Search Engine Results to Return. Google Max limit of 10 otherwise error.",
        )
        SCRAPPED_PAGES_NO: int = Field(
            default=10,
            description="Total pages scapped. Ideally greater than one of the returned pages",
        )
        PAGE_CONTENT_WORDS_LIMIT: int = Field(
            default=5000,
            description="Limit words content for each page.",
        )
        CITATION_LINKS: bool = Field(
            default=False,
            description="If True, send custom citations with links",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }

    async def search_web(
        self,
        query: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Search the web using Google Custom Search Engine and get the content of the relevant pages.
        :params query: Web Query used in search engine.
        :return: The content of the pages in json format.
        """
        functions = HelpFunctions()
        emitter = EventEmitter(__event_emitter__)
        await emitter.emit(f"Initiating web search for: {query}")

        # Enforce Google API Max return of 10 pages.
        if self.valves.ENGINE_RETURNED_PAGES_NO > 10:
            self.valves.ENGINE_RETURNED_PAGES_NO = 10
        # Ensure RETURNED_SCRAPPED_PAGES_NO does not exceed SCRAPPED_PAGES_NO
        if self.valves.ENGINE_RETURNED_PAGES_NO > self.valves.SCRAPPED_PAGES_NO:
            self.valves.ENGINE_RETURNED_PAGES_NO = self.valves.SCRAPPED_PAGES_NO

        # Google CSE API parameters
        params = {
            "q": query,
            "key": self.valves.GOOGLE_API_KEY,
            "cx": self.valves.GOOGLE_CSE_ID,
            "num": self.valves.ENGINE_RETURNED_PAGES_NO,  # Google's parameter for number of results
        }

        try:
            await emitter.emit("Sending request to Google CSE")
            resp = requests.get(
                "https://www.googleapis.com/customsearch/v1",  # Google CSE endpoint
                params=params,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("items", [])  # Google returns results in "items" field
            limited_results = results[: self.valves.SCRAPPED_PAGES_NO]
            await emitter.emit(f"Retrieved {len(limited_results)} search results")
        except requests.exceptions.RequestException as e:
            await emitter.emit(
                status="error",
                description=f"Error during search: {str(e)}",
                done=True,
            )
            return json.dumps({"error": str(e)})

        results_json = []
        if limited_results:
            await emitter.emit(f"Processing search results")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        functions.process_search_result, result, self.valves
                    )
                    for result in limited_results
                ]
                for future in concurrent.futures.as_completed(futures):
                    result_json = future.result()
                    if result_json:
                        try:
                            json.dumps(result_json)
                            results_json.append(result_json)
                        except (TypeError, ValueError):
                            continue
                    if len(results_json) >= self.valves.SCRAPPED_PAGES_NO:
                        break

            results_json = results_json[: self.valves.SCRAPPED_PAGES_NO]
            if self.valves.CITATION_LINKS and __event_emitter__:
                for result in results_json:
                    await __event_emitter__(
                        {
                            "type": "citation",
                            "data": {
                                "document": [result["content"]],
                                "metadata": [{"source": result["url"]}],
                                "source": {"name": result["title"]},
                            },
                        }
                    )

        await emitter.emit(
            status="complete",
            description=f"Web search completed. Retrieved content from {len(results_json)} pages",
            done=True,
        )
        return json.dumps(results_json, ensure_ascii=False)

    async def get_website(
        self, url: str, __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Web scrape the website provided and get the content of it.
        :params url: The URL of the website.
        :return: The content of the website in json format.
        """
        functions = HelpFunctions()
        emitter = EventEmitter(__event_emitter__)
        await emitter.emit(f"Fetching content from URL: {url}")
        results_json = []
        try:
            response_site = requests.get(url, headers=self.headers, timeout=120)
            response_site.raise_for_status()
            html_content = response_site.text
            await emitter.emit("Parsing website content")
            soup = BeautifulSoup(html_content, "html.parser")
            page_title = soup.title.string if soup.title else "No title found"
            page_title = unicodedata.normalize("NFKC", page_title.strip())
            page_title = functions.remove_emojis(page_title)
            title_site = page_title
            url_site = url
            content_site = functions.format_text(
                soup.get_text(separator=" ", strip=True)
            )
            truncated_content = functions.truncate_to_n_words(
                content_site, self.valves.PAGE_CONTENT_WORDS_LIMIT
            )
            result_site = {
                "title": title_site,
                "url": url_site,
                "content": truncated_content,
                "excerpt": functions.generate_excerpt(content_site),
            }
            results_json.append(result_site)
            if self.valves.CITATION_LINKS and __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "citation",
                        "data": {
                            "document": [truncated_content],
                            "metadata": [{"source": url_site}],
                            "source": {"name": title_site},
                        },
                    }
                )
            await emitter.emit(
                status="complete",
                description="Website content retrieved and processed successfully",
                done=True,
            )
        except requests.exceptions.RequestException as e:
            results_json.append(
                {
                    "url": url,
                    "content": f"Failed to retrieve the page. Error: {str(e)}",
                }
            )
            await emitter.emit(
                status="error",
                description=f"Error fetching website content: {str(e)}",
                done=True,
            )
        return json.dumps(results_json, ensure_ascii=False)


# ---------------- Convenience Wrapper Function ---------------- #
def google_search_pse(query: str) -> str:
    """同步呼叫 Google Custom Search 並回傳格式化字串。

    1. 自動讀取環境變數 GOOGLE_API_KEY 與 GOOGLE_CSE_ID。
    2. 取回搜尋結果 (使用上方 Tools().search_web)。
    3. 彙整前幾筆結果的內容摘要與其 URL。

    回傳格式:
        "Search results for '{query}' , information are :{answer} and corresponding to {url_list}"

    參數
    ------
    query: str
        要搜尋的查詢字串。
    """
    tools = Tools()
    # 從環境變數載入金鑰
    tools.valves.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    tools.valves.GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

    if not tools.valves.GOOGLE_API_KEY or not tools.valves.GOOGLE_CSE_ID:
        raise ValueError(
            "Missing GOOGLE_API_KEY or GOOGLE_CSE_ID environment variables. Set them before calling google_search_pse()."
        )

    # 可視需要調整可返回/擷取數量
    tools.valves.ENGINE_RETURNED_PAGES_NO = 5
    tools.valves.SCRAPPED_PAGES_NO = 5
    tools.valves.PAGE_CONTENT_WORDS_LIMIT = 300

    async def _run():
        return await tools.search_web(query)

    # 執行 async 搜尋
    raw = asyncio.run(_run())

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return f"Search results for '{query}' , information are :<parse error> and corresponding to <unknown>"

    if not isinstance(data, list) or not data:
        return f"Search results for '{query}' , information are :<no results> and corresponding to <none>"

    # 取前 3 筆摘要
    top_items = data[:3]
    answer_parts = []
    url_parts = []
    for item in top_items:
        content = item.get("content", "")
        # 簡短摘要 (前 200 字元)
        snippet = content[:200].strip()
        answer_parts.append(snippet)
        url_parts.append(item.get("url", ""))

    answer = " | ".join(answer_parts)
    urls = ", ".join(url_parts)
    return f"Search results for '{query}' , information are :{answer} and corresponding to {urls}"


def google_search_pse_with_contents(query: str):
    """回傳包含每個搜尋結果完整(截斷後)內容的資料結構。

    Returns: list[dict]
        Each dict: {title, url, snippet, content}
    """
    tools = Tools()
    tools.valves.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    tools.valves.GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

    if not tools.valves.GOOGLE_API_KEY or not tools.valves.GOOGLE_CSE_ID:
        raise ValueError("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID.")
    tools.valves.ENGINE_RETURNED_PAGES_NO = 5
    tools.valves.SCRAPPED_PAGES_NO = 5
    tools.valves.PAGE_CONTENT_WORDS_LIMIT = 800  # 多給一點字數
    loop = asyncio.get_event_loop()
    raw = loop.run_until_complete(tools.search_web(query))
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    # 保留 title/url/snippet/content
    cleaned = []
    for item in data:
        cleaned.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("snippet"),
                "content": item.get("content"),
            }
        )
    return cleaned


if __name__ == "__main__":
    # 範例呼叫
    q = "brad pitt新電影 上印日期"
    try:
        # result_text = google_search_pse(q)
        # print(result_text)
        # 額外：取得每筆完整內容
        full_data = google_search_pse_with_contents(q)
        print(
            f"共 {len(full_data)} 筆內容，第一筆字數: {len(full_data[0]['content']) if full_data else 0}"
        )
        for item in full_data:
            print(f"Title: {item['title']}")
            print(f"URL: {item['url']}")
            print(f"Snippet: {item['snippet']}")
            print(f"Content: {item['content'][:200]}...")  # Print first 200 characters of content
            print("-" * 80)
        
    except ValueError as e:
        print(f"Config error: {e}")
