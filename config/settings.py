import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
SEC_EDGAR_USER_AGENT: str = os.getenv("SEC_EDGAR_USER_AGENT", "")
GROQ_MODEL_SIGNAL: str = os.getenv("GROQ_MODEL_SIGNAL", "llama-3.3-70b-versatile")
GROQ_MODEL_REPORT: str = os.getenv("GROQ_MODEL_REPORT", "llama-3.1-8b-instant")

_missing = [
    name
    for name, val in [
        ("GROQ_API_KEY", GROQ_API_KEY),
        ("SEC_EDGAR_USER_AGENT", SEC_EDGAR_USER_AGENT),
    ]
    if not val
]
if _missing:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(_missing)}. "
        "Copy .env.example to .env and fill in the values."
    )
