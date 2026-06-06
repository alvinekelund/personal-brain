"""Minimal Gemini REST client over stdlib urllib — no SDK required.

The google-genai package isn't always installable (offline/sandboxed envs), so we
talk to the generativelanguage REST API directly. Only the standard library is used.
"""
import json
import os
import urllib.error
import urllib.request

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-2.5-flash"


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
    timeout: float = 60.0,
) -> str:
    """Call Gemini generateContent and return the response text.

    response_json=True asks the model to emit raw JSON (responseMimeType),
    which removes the need to strip ``` fences for structured extraction.
    """
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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:400]
        hint = ""
        if e.code == 400 and "API_KEY_INVALID" in detail:
            hint = (
                "\nHint: Google rejected the configured GEMINI_API_KEY. Check it at "
                "https://aistudio.google.com/apikey and update ~/.personal-brain/.env"
            )
        raise RuntimeError(f"Gemini API error {e.code}: {detail}{hint}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach Gemini API: {e.reason}") from None

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Gemini response shape: {json.dumps(data)[:400]}")
