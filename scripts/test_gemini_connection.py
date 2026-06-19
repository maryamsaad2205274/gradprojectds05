"""
test_gemini_connection.py
=========================
Standalone script that verifies the Gemini API key stored in .env is valid
and that a minimal text request succeeds.

Safety rules enforced:
  - The API key is never printed, logged, or exposed in any output.
  - No DentAlign logic, models, DB, or Flask routes are imported or modified.
  - This file is purely a connection probe and can be deleted afterwards.

Run from the project root:
    python test_gemini_connection.py
"""

import os
import sys

# ── 1. Load .env (must be beside this file / beside app.py) ──────────────────
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ENV  = _HERE / ".env"

if not _ENV.is_file():
    print(f"[ERROR] .env file not found at: {_ENV}")
    print("  Create it and add:  GEMINI_API_KEY=<your_key>")
    sys.exit(1)

from dotenv import load_dotenv
load_dotenv(dotenv_path=_ENV, override=False)

# ── 2. Read and validate key — never print its value ─────────────────────────
# Validation rules:
#   - Key must exist and be non-empty.
#   - Key must not equal the exact placeholder string.
#   - No prefix check (e.g. "AIza") — Google AI Studio now issues keys
#     starting with "AQ" and the format may change again in future.
#   - No length check — key length is not part of the public contract.
# The complete, unmodified value is passed directly to genai.Client().

api_key = os.getenv("GEMINI_API_KEY")

if not api_key or not api_key.strip():
    print("[ERROR] GEMINI_API_KEY is not set or is empty in .env.")
    sys.exit(1)

if api_key.strip() == "PASTE_MY_PRIVATE_KEY_HERE":
    print("[ERROR] GEMINI_API_KEY still contains the placeholder value.")
    print("  Open .env and replace the placeholder with your real API key.")
    sys.exit(1)

print("[OK] API key loaded from .env (value hidden).")

# ── 3. Initialise the google-genai SDK ────────────────────────────────────────
try:
    from google import genai
    from google.genai import errors as genai_errors
except ImportError:
    print("[ERROR] google-genai is not installed.")
    print("  Run:  pip install google-genai")
    sys.exit(1)

try:
    client = genai.Client(api_key=api_key)
except Exception as exc:
    print(f"[ERROR] Could not create Gemini client: {exc}")
    sys.exit(1)

# ── 4. Model candidates — tried in order, no listing call needed ──────────────
# Skipping client.models.list() avoids consuming an extra quota slot.
# Models are tried in order; 404 means the model is unavailable for this key,
# 429 means rate-limited — both cause a fallback to the next candidate.
CANDIDATE_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
]

selected_model = None   # filled when a model responds successfully

# ── 5. Send a minimal text request (tries each candidate on 429 / 404) ────────
PROMPT = 'Reply with exactly this text and nothing else: Gemini API connected successfully'

import time

reply = None

for attempt_model in CANDIDATE_MODELS:
    print(f"\n[INFO] Sending test prompt to {attempt_model} …")
    try:
        response = client.models.generate_content(
            model=attempt_model,
            contents=PROMPT,
        )
        reply = response.text.strip()
        selected_model = attempt_model   # record which model actually responded
        break  # success — stop trying

    except genai_errors.APIError as exc:
        msg = str(exc)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            print(f"  [WARN] Rate-limit on {attempt_model} — trying next model …")
            time.sleep(4)
            continue   # try next model
        if "NOT_FOUND" in msg or "404" in msg:
            print(f"  [WARN] {attempt_model} not available — trying next model …")
            continue   # try next model
        # Non-rate-limit API error — report and stop
        code = getattr(exc, "code", "unknown")
        if "API_KEY_INVALID" in msg or "401" in msg:
            print("[ERROR] Invalid API key (HTTP 401). Check your GEMINI_API_KEY.")
        elif "PERMISSION_DENIED" in msg or "403" in msg:
            print("[ERROR] Permission denied (HTTP 403). Key may lack model access.")
        else:
            print(f"[ERROR] Gemini API error (code={code}): {msg}")
        sys.exit(1)

    except OSError as exc:
        print(f"[ERROR] Network error — could not reach the Gemini API: {exc}")
        sys.exit(1)

    except Exception as exc:
        print(f"[ERROR] Unexpected error: {type(exc).__name__}: {exc}")
        sys.exit(1)

if reply is None:
    print("\n[ERROR] All models returned rate-limit errors (HTTP 429).")
    print("  Free-tier quota exhausted for this minute.")
    print("  Wait ~60 seconds and run the script again.")
    print("  Quota dashboard: https://aistudio.google.com/app/quota")
    sys.exit(1)

# ── 6. Show result ────────────────────────────────────────────────────────────
print(f"\n[RESPONSE] {reply}")

EXPECTED = "Gemini API connected successfully"
if EXPECTED.lower() in reply.lower():
    print("\n[SUCCESS] Gemini API connection verified.")
else:
    print("\n[WARNING] The model responded, but the reply did not match the expected phrase.")
    print(f"  Expected (approx): {EXPECTED!r}")
    print("  The API is reachable, but the model may have paraphrased the response.")
    print("  Connection is functional.")
