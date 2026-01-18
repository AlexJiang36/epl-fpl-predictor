import httpx

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

def fetch_bootstrap() -> dict:
    """Fetch FPL bootstrap-static JSON."""
    with httpx.Client(timeout=30) as client:
        r = client.get(BOOTSTRAP_URL)
        r.raise_for_status()
        return r.json()