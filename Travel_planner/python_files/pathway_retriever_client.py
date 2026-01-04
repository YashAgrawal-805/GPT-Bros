import requests

class PathwayRetrieverClient:
    def __init__(self, base_url="http://localhost:8765"):
        self.base_url = base_url.rstrip("/")

    def search(self, query: str, k: int = 5):
        # endpoint depends on which Pathway template you use
        # common ones: /v1/retrieve (template) or /v1/query (RAG REST API)
        resp = requests.post(
            f"{self.base_url}/v1/retrieve",
            json={"query": query, "k": k},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
