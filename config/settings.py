import logging
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

# Optional — LLM agents warn at call time if this is None; dashboard imports safely.
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY") or None
if GROQ_API_KEY is None:
    log.warning(
        "GROQ_API_KEY is not set — LLM agents (signal_finder, report_writer) "
        "will fail if invoked. Set it in .env for local development."
    )

# Not validated here; edgar_downloader raises at run() time if this is empty.
SEC_EDGAR_USER_AGENT: str = os.getenv("SEC_EDGAR_USER_AGENT", "")

GROQ_MODEL_SIGNAL: str = os.getenv("GROQ_MODEL_SIGNAL", "llama-3.3-70b-versatile")
GROQ_MODEL_REPORT: str = os.getenv("GROQ_MODEL_REPORT", "llama-3.1-8b-instant")
