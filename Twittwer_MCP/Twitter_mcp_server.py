from __future__ import annotations
import os
import re
from typing import Any, Dict
from dotenv import load_dotenv
load_dotenv()
import httpx
from fastmcp import FastMCP

mcp = FastMCP("TwitterMCP", log_level="WARNING")

X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN") or ""
BASE = "https://api.x.com/2/tweets/search/recent"


def _require_bearer() -> None:
    if not X_BEARER_TOKEN:
        raise RuntimeError("Missing X_BEARER_TOKEN (Bearer token required for v2 recent search).")


@mcp.tool()
async def hashtags_for_place(
    place: str,
    topic: str = "",
    max_results: int = 50,
    lang: str = "en",
) -> Dict[str, Any]:
    """
    Returns top hashtags from recent tweets mentioning a place (+ optional topic).
    Uses X API v2 recent search (Bearer token).
    """
    _require_bearer()

    place_q = f'"{place}"'
    topic_q = f"({topic})" if topic.strip() else ""
    query = " ".join([place_q, topic_q, f"lang:{lang}", "-is:retweet"]).strip()

    headers = {"Authorization": f"Bearer {X_BEARER_TOKEN}"}
    params = {"query": query, "max_results": min(int(max_results), 100)}

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(BASE, headers=headers, params=params)

    if resp.status_code != 200:
        return {"status": "error", "code": resp.status_code, "message": resp.text[:1500], "query": query}

    data = resp.json()
    tweets = data.get("data") or []

    freq: Dict[str, int] = {}
    for t in tweets:
        for tag in re.findall(r"#\w+", t.get("text", "")):
            tag = tag.lower()
            freq[tag] = freq.get(tag, 0) + 1

    top = sorted(
        [{"tag": k, "count": v} for k, v in freq.items()],
        key=lambda x: x["count"],
        reverse=True
    )[:20]

    return {
        "status": "success",
        "place": place,
        "topic": topic,
        "query": query,
        "hashtags": top,
        "tweet_count": len(tweets),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
