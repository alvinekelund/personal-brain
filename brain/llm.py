"""Minimal Gemini REST client over stdlib urllib — no SDK required.

The google-genai package isn't always installable (offline/sandboxed envs), so we
talk to the generativelanguage REST API directly. Only the standard library is used.
"""
import json
import os
import ssl
import time
import urllib.error
import urllib.request

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"
RETRIES = 2          # extra attempts on transient failures
BACKOFF = 1.5        # seconds, multiplied by attempt number
RATE_LIMIT_BACKOFF = 20.0   # seconds for a 429 when the API names no retryDelay
RATE_LIMIT_MAX_WAIT = 65.0  # cap on a server-suggested retryDelay

# Wall-clock budget per logical call, retries and 429 waits included. Without it
# one `brain add` could sit for 10+ minutes while retries and quota waits
# compounded across the dozen calls an ingest makes (L-061); a scheduled task
# would rather fail one fact fast than miss its window. Override per process
# with BRAIN_LLM_BUDGET / BRAIN_EMBED_BUDGET (seconds).
GENERATE_BUDGET = 120.0
EMBED_BUDGET = 45.0


class BudgetExceeded(OSError):
    """The call (attempts + waits) would outlive its wall-clock budget."""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _sleep_within(seconds: float, deadline, budget, attempt: int):
    """Sleep before the next attempt — unless that wait alone would blow the
    budget, in which case give up now: sleeping would only delay the failure."""
    if deadline is not None and time.monotonic() + seconds > deadline:
        raise BudgetExceeded(
            f"gave up after attempt {attempt + 1}/{RETRIES + 1}: a {seconds:.0f}s retry "
            f"wait would exceed the {budget:.0f}s budget")
    time.sleep(seconds)


def _retry_delay(body: str, attempt: int) -> float:
    """Seconds to wait after a 429. Honors the RetryInfo the API sends
    (e.g. "retryDelay": "18s"); falls back to a flat backoff that outlasts
    a per-minute quota window."""
    try:
        for detail in json.loads(body)["error"]["details"]:
            if detail.get("@type", "").endswith("RetryInfo"):
                return min(float(detail["retryDelay"].rstrip("s")), RATE_LIMIT_MAX_WAIT)
    except (ValueError, KeyError, TypeError, AttributeError):
        pass
    return RATE_LIMIT_BACKOFF * (attempt + 1)


def ssl_context() -> ssl.SSLContext:
    """TLS trust for every API call. The python.org framework build ships with an
    empty CA store, so a plain urlopen fails CERTIFICATE_VERIFY_FAILED; certifi's
    bundle fixes that when installed, else fall back to the system default."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _request(req, timeout, budget=None):
    """POST and parse JSON, retrying transient failures: 5xx, dropped
    connections, timeouts, and 429 rate limits (waiting out the quota window —
    free-tier keys allow only a handful of requests per minute). Other 4xx
    (bad key/request) fail fast — no point retrying.

    `timeout` bounds one socket wait; `budget` (seconds) bounds the whole call:
    each attempt's timeout is cut to what is left, and a retry wait that would
    overrun it raises BudgetExceeded instead of sleeping.
    """
    deadline = None if budget is None else time.monotonic() + budget
    for attempt in range(RETRIES + 1):
        per_try = timeout
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BudgetExceeded(f"{budget:.0f}s budget exhausted after {attempt} attempt(s)")
            per_try = min(timeout, remaining)
        try:
            with urllib.request.urlopen(req, timeout=per_try, context=ssl_context()) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < RETRIES:
                wait = _retry_delay(e.read().decode(errors="replace"), attempt)
                try:
                    _sleep_within(wait, deadline, budget, attempt)
                except BudgetExceeded:
                    e.close()
                    raise
                continue
            if e.code < 500 or attempt == RETRIES:
                raise
        except OSError:  # URLError, ConnectionError (incl. RemoteDisconnected), timeout
            if attempt == RETRIES:
                raise
        _sleep_within(BACKOFF * (attempt + 1), deadline, budget, attempt)


def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY in your environment or in "
            "~/.personal-brain/.env (or the project .env)."
        )
    return key


def have_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def embed(text: str, model: str = EMBED_MODEL, timeout: float = 30.0,
          budget: float | None = None) -> list:
    """Return the embedding vector for `text` via the Gemini embeddings API.
    `budget` caps the whole call (retries + waits); default BRAIN_EMBED_BUDGET."""
    if budget is None:
        budget = _env_float("BRAIN_EMBED_BUDGET", EMBED_BUDGET)
    body = {"model": f"models/{model}", "content": {"parts": [{"text": text}]}}
    req = urllib.request.Request(
        f"{API_ROOT}/{model}:embedContent",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key()},
    )
    try:
        data = _request(req, timeout, budget=budget)
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Gemini embeddings error {e.code}: {e.read().decode(errors='replace')[:300]}"
        ) from None
    except BudgetExceeded as e:
        raise RuntimeError(f"Gemini embeddings call over budget: {e}") from None
    except OSError as e:
        raise RuntimeError(f"Could not reach Gemini embeddings API: {e}") from None
    try:
        return data["embedding"]["values"]
    except (KeyError, TypeError):
        raise RuntimeError(f"Unexpected embeddings response: {json.dumps(data)[:200]}")


def parse_json(raw: str):
    """Parse JSON from an LLM response, tolerating ``` fences and surrounding prose.

    Falls back to the outermost {...} block so a chatty or partial response can't
    crash callers. Single source of truth for parsing model output.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def generate(
    prompt: str,
    system: str = "",
    model: str = DEFAULT_MODEL,
    response_json: bool = False,
    timeout: float | None = None,
    budget: float | None = None,
) -> str:
    """Call Gemini generateContent and return the response text.

    response_json=True asks the model to emit raw JSON (responseMimeType),
    which removes the need to strip ``` fences for structured extraction.
    `budget` caps the whole call (retries + waits); default BRAIN_LLM_BUDGET.
    `timeout` is one attempt's socket wait and defaults to the budget: a slow
    but progressing generation (a dense 4 KB chunk can take 80 s) must not be
    killed at 60 s and restarted from scratch — retries are for failures.
    """
    if budget is None:
        budget = _env_float("BRAIN_LLM_BUDGET", GENERATE_BUDGET)
    if timeout is None:
        timeout = budget
    body: dict = {"contents": [{"parts": [{"text": prompt}]}]}
    if system:
        body["system_instruction"] = {"parts": [{"text": system}]}
    if response_json:
        body["generationConfig"] = {"responseMimeType": "application/json"}

    req = urllib.request.Request(
        f"{API_ROOT}/{model}:generateContent",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key(),  # header, not URL — keeps the key out of logs
        },
    )
    try:
        data = _request(req, timeout, budget=budget)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:400]
        hint = ""
        if e.code == 400 and "API_KEY_INVALID" in detail:
            hint = (
                "\nHint: Google rejected the configured GEMINI_API_KEY. Check it at "
                "https://aistudio.google.com/apikey and update ~/.personal-brain/.env"
            )
        elif e.code == 429:
            hint = (
                "\nHint: rate limited even after waiting and retrying — a free-tier "
                "key allows only a few requests per minute. Wait a minute and retry, "
                "or use a key from a billed project."
            )
        raise RuntimeError(f"Gemini API error {e.code}: {detail}{hint}") from None
    except BudgetExceeded as e:
        raise RuntimeError(
            f"Gemini call over budget: {e}. Raise BRAIN_LLM_BUDGET (seconds) to allow longer waits."
        ) from None
    except OSError as e:
        raise RuntimeError(f"Could not reach Gemini API: {e}") from None

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Gemini response shape: {json.dumps(data)[:400]}")
