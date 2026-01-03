import asyncio
import os
import sys
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
load_dotenv()
SERVER_PATH = r"C:\Users\santa\Desktop\twitter_mcp\twitter_mcp_server.py"

SERVERS = {
    "twitter-mcp": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [SERVER_PATH],
        "env": {
            "API_KEY": os.getenv("API_KEY") or "",
            "API_SECRET_KEY": os.getenv("API_SECRET_KEY") or "",
            "ACCESS_TOKEN": os.getenv("ACCESS_TOKEN") or "",
            "ACCESS_TOKEN_SECRET": os.getenv("ACCESS_TOKEN_SECRET") or "",
            "X_BEARER_TOKEN": os.getenv("BEARER_TOKEN") or "",
        },
    }
}

async def main():
    print("ENV CHECK:",
          bool(os.getenv("API_KEY")),
          bool(os.getenv("API_SECRET_KEY")),
          bool(os.getenv("ACCESS_TOKEN")),
          bool(os.getenv("ACCESS_TOKEN_SECRET")))

    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()

    print("\nTOOLS FOUND:")
    for t in tools:
        print("-", t.name)

    # Prefer hashtags tool if present (your current need)
    hashtag_tool = next((t for t in tools if t.name.endswith("hashtags_for_place")), None)
    search_tool = next((t for t in tools if t.name.endswith("search_tweets")), None)

    if hashtag_tool:
        res = await hashtag_tool.ainvoke({
            "place": "manali",
            "topic": "all",
            "max_results": 1,
            "lang": "en",
        })
        print("\nRESULT:\n", res)
        return

    if search_tool:
        res = await search_tool.ainvoke({
            "q": "San Francisco traffic OR accident OR fire",
            "count": 1,
            "result_type": "recent",
            "lang": "en",
        })
        print("\nRESULT:\n", res)
        return

    raise RuntimeError("No matching tool found. Expected 'hashtags_for_place' or 'search_tweets'.")

if __name__ == "__main__":
    asyncio.run(main())
