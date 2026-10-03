import os
import re
import json
import time
import html
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
from google import genai
from google.genai import types

# =============================================================================
# OmniExtract AI
# Evidence-grounded web intelligence in one Streamlit file.
#
# Pipeline:
#   Natural language
#        ↓
#   Schema planner
#        ↓
#   Gemini + Google Search grounding (with retry + model fallback)
#        ↓
#   Structured extraction + evidence
#        ↓
#   Deterministic validation (source, reachability, quote-on-page check)
#        ↓
#   Deduplication
#        ↓
#   Evidence-derived confidence
#        ↓
#   Dataset + source inspection + exports
#
# IMPORTANT:
# - No fabricated fallback records.
# - Confidence is computed by this application, not supplied by the LLM.
# - Every returned record must contain a source URL and evidence quote.
# =============================================================================

st.set_page_config(
    page_title="OmniExtract AI | Evidence Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
.stApp {
    background: #0d1117;
    color: #c9d1d9;
}
.main-card, .source-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 18px;
}
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}
.metric-card h4 {
    color: #8b949e;
    margin-bottom: 4px;
}
.metric-card h2 {
    margin-top: 0;
}
.stButton > button {
    width: 100%;
    border-radius: 7px;
    font-weight: 700;
}
.small-muted {
    color: #8b949e;
    font-size: 0.9rem;
}
.badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 999px;
    border: 1px solid #30363d;
    font-size: 0.78rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
for key, default in {
    "history": [],
    "current_results": None,
    "last_successful_results": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Tried in order. If one is busy (503) or missing (404), the next is used.
# Check these names in Google AI Studio and edit the list if needed.
MODELS = [
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
RETRIES_PER_MODEL = 3

MAX_RECORDS = 8
REQUEST_TIMEOUT = 10
MAX_PAGE_CHARS = 600_000
URL_CHECK_WORKERS = 6

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; OmniExtractAI/1.0)"}

api_key = os.getenv("GEMINI_API_KEY") or st.sidebar.text_input(
    "Gemini API Key", type="password"
)


def get_client():
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def log_message(logs, log_area, message):
    logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")
    log_area.code("\n".join(logs), language="text")


def clean_url(value):
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if not value:
        return ""
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return value


def domain_from_url(url):
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def normalize_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).lower()


EMPTY_MARKERS = {"", "none", "null", "n/a", "na", "unknown", "nan"}


def is_filled(value):
    """True only for real values. None / 'null' / 'N/A' count as empty."""
    if value is None:
        return False
    if isinstance(value, float) and value != value:  # NaN
        return False
    if isinstance(value, str):
        return value.strip().lower() not in EMPTY_MARKERS
    return True


def display_value(value):
    return str(value) if is_filled(value) else "—"


def extract_json(text):
    """Handle valid JSON plus occasional fenced JSON from a model."""
    if not text:
        raise ValueError("Empty model response.")

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to recover the first JSON object/array.
        starts = [p for p in (text.find("["), text.find("{")) if p >= 0]
        if not starts:
            raise
        start = min(starts)
        candidate = text[start:]
        decoder = json.JSONDecoder()
        value, _ = decoder.raw_decode(candidate)
        return value


def get_grounding_sources(response):
    """Extract URLs/titles from Gemini grounding metadata."""
    sources = []

    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return sources

        metadata = getattr(candidates[0], "grounding_metadata", None)
        if not metadata:
            return sources

        chunks = getattr(metadata, "grounding_chunks", None) or []

        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if not web:
                continue

            uri = clean_url(getattr(web, "uri", ""))
            title = getattr(web, "title", "") or domain_from_url(uri)

            if uri:
                sources.append({"url": uri, "title": title})

    except Exception:
        # Grounding data should enrich the result, not crash the application.
        pass

    unique = {}
    for source in sources:
        unique[source["url"]] = source
    return list(unique.values())


# -----------------------------------------------------------------------------
# Grounding match (exact URL, or same domain)
# -----------------------------------------------------------------------------
REDIRECT_HOSTS = ("vertexaisearch.cloud.google.com",)


def build_grounding_index(grounded_sources):
    """
    Gemini often returns redirect links as the URL and puts the real domain
    in the title. So we keep both exact URLs and domains.
    """
    exact = set()
    domains = set()

    for source in grounded_sources:
        url = clean_url(source.get("url", ""))
        if url:
            exact.add(url.rstrip("/"))
            domain = domain_from_url(url)
            if domain and domain not in REDIRECT_HOSTS:
                domains.add(domain)

        title = str(source.get("title", "") or "").strip().lower()
        title = title.replace("www.", "")
        if title and "." in title and " " not in title:
            domains.add(title)

    return exact, domains


def domain_matches(domain, grounded_domains):
    if not domain:
        return False
    for known in grounded_domains:
        if (
            domain == known
            or domain.endswith("." + known)
            or known.endswith("." + domain)
        ):
            return True
    return False


def grounding_match_type(source_url, exact, domains):
    """Returns 'exact', 'domain' or None."""
    if source_url.rstrip("/") in exact:
        return "exact"
    if domain_matches(domain_from_url(source_url), domains):
        return "domain"
    return None


# -----------------------------------------------------------------------------
# Source fetching + quote verification
# -----------------------------------------------------------------------------
def fetch_page(url):
    """Fetch once: reachability + page text for quote verification."""
    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
            headers=HTTP_HEADERS,
        )
        ok = 200 <= response.status_code < 400
        text = ""
        content_type = response.headers.get("Content-Type", "").lower()
        if ok and any(t in content_type for t in ("text", "html", "xml", "json")):
            text = response.text[:MAX_PAGE_CHARS]
        return {"ok": ok, "status": f"HTTP {response.status_code}", "text": text}
    except requests.RequestException as exc:
        return {"ok": False, "status": type(exc).__name__, "text": ""}


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _page_to_text(raw_html):
    raw_html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw_html)
    raw_html = re.sub(r"(?s)<[^>]+>", " ", raw_html)
    return html.unescape(raw_html)


def quote_in_page(quote, page_html):
    """
    True  = quote found on page (exact, or at least 85% of its words).
    False = page was readable but quote not found.
    None  = could not check (no page text, or quote too short).
    """
    if not page_html:
        return None

    quote_tokens = _tokens(quote)
    if len(quote_tokens) < 3:
        return None

    page_text = _page_to_text(page_html)
    page_tokens = _tokens(page_text)
    if not page_tokens:
        return None

    # Exact phrase match on normalised word streams.
    if " ".join(quote_tokens) in " ".join(page_tokens):
        return True

    # Looser match: most of the quote's words appear on the page.
    page_set = set(page_tokens)
    hits = sum(1 for token in quote_tokens if token in page_set)
    return hits / len(quote_tokens) >= 0.85


def deterministic_confidence(record, match_type, url_reachable, quote_verified):
    """
    Evidence-derived score (max 99).
    This is deliberately deterministic and capped below 100%.
    It is NOT a claim that the factual content is guaranteed true.

      valid URL ................ 15
      grounding: exact 25, same-domain 15
      evidence quote present ... 15
      source reachable now ..... 10
      quote found on the page .. 25
      field completeness ....... up to 10
    """
    score = 0

    if clean_url(record.get("source_url")):
        score += 15

    if match_type == "exact":
        score += 25
    elif match_type == "domain":
        score += 15

    if len(str(record.get("evidence_quote", "")).strip()) >= 20:
        score += 15

    if url_reachable:
        score += 10

    if quote_verified is True:
        score += 25

    required = record.get("_fields", [])
    if required:
        populated = sum(is_filled(record.get(field)) for field in required)
        score += round(10 * populated / len(required))

    return min(score, 99)


def deduplicate_records(records, fields):
    """Deterministic duplicate removal using requested fields."""
    seen = set()
    output = []

    for record in records:
        key_parts = [normalize_text(record.get(field, "")) for field in fields]
        key = tuple(key_parts)

        # If all requested fields are empty, don't collapse unrelated records.
        if not any(key_parts):
            output.append(record)
            continue

        if key not in seen:
            seen.add(key)
            output.append(record)

    return output


# -----------------------------------------------------------------------------
# API failure classification + retry/fallback
# -----------------------------------------------------------------------------
class GeminiQuotaError(RuntimeError):
    """Raised when Gemini is rate-limited, overloaded or unavailable."""


def error_code(exc):
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def is_overloaded(exc):
    """Quota, rate limit, or server overload (429 / 500 / 503 / 504)."""
    if error_code(exc) in (429, 500, 503, 504):
        return True

    message = str(exc).lower()
    markers = (
        "resource_exhausted",
        "resource exhausted",
        "quota exceeded",
        "quota_exceeded",
        "rate limit",
        "rate_limit",
        "too many requests",
        "requests per minute",
        "requests per day",
        "tokens per minute",
        "tokens per day",
        "unavailable",
        "high demand",
        "overloaded",
    )
    return any(marker in message for marker in markers)


is_quota_error = is_overloaded


def raise_if_quota_error(exc):
    if is_overloaded(exc):
        raise GeminiQuotaError(
            "Gemini is rate-limited or overloaded right now. "
            "All fallback models were tried. Wait a few minutes and retry, "
            "or use an API key/project with available quota."
        ) from exc
    raise exc


def generate_with_retry(client, logs, log_area, **kwargs):
    """
    Try each model in MODELS. Per model: retry with backoff on overload.
    Missing model (404) -> skip to next model.
    Any other error -> raise immediately.
    Returns (response, model_used).
    """
    last_exc = None

    for model in MODELS:
        for attempt in range(1, RETRIES_PER_MODEL + 1):
            try:
                response = client.models.generate_content(model=model, **kwargs)
                if model != MODELS[0]:
                    log_message(logs, log_area, f"Using fallback model: {model}")
                return response, model

            except Exception as exc:
                last_exc = exc

                if error_code(exc) == 404:
                    log_message(logs, log_area, f"{model} not found. Next model.")
                    break

                if not is_overloaded(exc):
                    raise

                if attempt < RETRIES_PER_MODEL:
                    wait = 2 ** attempt
                    log_message(
                        logs,
                        log_area,
                        f"{model} busy (try {attempt}/{RETRIES_PER_MODEL}). "
                        f"Waiting {wait}s...",
                    )
                    time.sleep(wait)
                else:
                    log_message(
                        logs,
                        log_area,
                        f"{model} still busy after {RETRIES_PER_MODEL} tries.",
                    )

    raise last_exc


# -----------------------------------------------------------------------------
# Stage 1: schema planning
# -----------------------------------------------------------------------------
def plan_schema(client, user_prompt, logs, log_area):
    log_message(logs, log_area, "Planning target entity and extraction schema...")

    plan_prompt = f"""
You are the schema planner for a web-intelligence system.

User request:
{user_prompt}

Return ONLY JSON:
{{
  "entity": "short entity name",
  "fields": ["field1", "field2", "field3", "field4"],
  "search_focus": "short description of what evidence should be searched for"
}}

Rules:
- Choose 4 to 6 useful fields.
- Do not include source_url, evidence_quote, evidence_title, or confidence_score
  in fields; the application adds those.
- Fields must be concise snake_case identifiers.
- Optimize the schema for factual web research.
"""

    response, _ = generate_with_retry(
        client,
        logs,
        log_area,
        contents=plan_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )

    plan = extract_json(response.text)

    if isinstance(plan, list):
        plan = plan[0] if plan else {}

    if not isinstance(plan, dict):
        raise ValueError("Schema planner returned an invalid object.")

    fields = plan.get("fields")
    if not isinstance(fields, list):
        raise ValueError("Schema planner returned no fields.")

    fields = [
        re.sub(r"[^a-zA-Z0-9_]", "_", str(field)).strip("_").lower()
        for field in fields
    ]
    fields = list(dict.fromkeys(field for field in fields if field))

    if not fields:
        raise ValueError("No usable schema fields were generated.")

    return {
        "entity": str(plan.get("entity", "Target Entity")),
        "fields": fields[:6],
        "search_focus": str(plan.get("search_focus", "")),
    }


# -----------------------------------------------------------------------------
# Stage 2: actual grounded web retrieval + extraction
# -----------------------------------------------------------------------------
def grounded_extract(client, user_prompt, plan, logs, log_area):
    fields = plan["fields"]

    log_message(
        logs,
        log_area,
        "Activating Gemini Google Search grounding for live web retrieval...",
    )

    field_text = ", ".join(fields)

    extraction_prompt = f"""
You are the evidence extraction engine inside OmniExtract AI.

USER REQUEST:
{user_prompt}

TARGET ENTITY:
{plan["entity"]}

TARGET FIELDS:
{field_text}

SEARCH FOCUS:
{plan["search_focus"]}

Use Google Search grounding to find current public web evidence.

Return ONLY a JSON array containing up to {MAX_RECORDS} records.
No text before or after the array. No markdown.

Each record MUST have:
- every requested field
- source_url
- evidence_quote
- evidence_title

CRITICAL EVIDENCE RULES:
1. Do NOT invent facts from memory.
2. Do NOT invent URLs.
3. source_url MUST be one of the URLs actually returned by your web search.
4. evidence_quote must be a short, faithful quote or close factual excerpt
   from the cited source that supports the record.
5. If a field cannot be supported by the searched sources, use null rather
   than guessing.
6. Prefer primary sources, official company pages, official filings,
   reputable publications, or original announcements.
7. Do not assign confidence scores. The application computes them.
8. Return only records for which there is meaningful source evidence.
"""

    # NOTE: no response_mime_type here. Many Gemini models refuse
    # "JSON mode" and the Google Search tool in the same request.
    # extract_json() already handles plain / fenced JSON.
    response, model_used = generate_with_retry(
        client,
        logs,
        log_area,
        contents=extraction_prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )

    log_message(logs, log_area, f"Extraction model: {model_used}")

    records = extract_json(response.text)

    if isinstance(records, dict):
        records = records.get("records", [])

    if not isinstance(records, list):
        raise ValueError("Extraction engine did not return a JSON array.")

    grounded_sources = get_grounding_sources(response)

    log_message(
        logs,
        log_area,
        f"Grounding returned {len(grounded_sources)} source(s).",
    )

    exact_urls, grounded_domains = build_grounding_index(grounded_sources)

    # ---- First pass: cheap checks, no network ------------------------------
    candidates = []
    rejected = 0

    for raw in records[:MAX_RECORDS]:
        if not isinstance(raw, dict):
            rejected += 1
            continue

        record = dict(raw)

        source_url = clean_url(record.get("source_url", ""))
        record["source_url"] = source_url
        record["evidence_quote"] = str(record.get("evidence_quote", "") or "").strip()
        record["evidence_title"] = str(record.get("evidence_title", "") or "").strip()
        record["_fields"] = fields

        if not source_url or not record["evidence_quote"]:
            rejected += 1
            log_message(
                logs,
                log_area,
                "Rejected record: missing source URL or evidence quote.",
            )
            continue

        match_type = grounding_match_type(source_url, exact_urls, grounded_domains)

        # Never accept a source that search did not actually return.
        if match_type is None:
            rejected += 1
            log_message(
                logs,
                log_area,
                "Rejected record: source not in grounding metadata "
                f"({domain_from_url(source_url)}).",
            )
            continue

        record["_match"] = match_type
        candidates.append(record)

    # ---- Second pass: fetch each unique source once, in parallel -----------
    unique_urls = list(dict.fromkeys(r["source_url"] for r in candidates))
    pages = {}

    if unique_urls:
        log_message(
            logs,
            log_area,
            f"Checking {len(unique_urls)} source page(s) and verifying quotes...",
        )
        workers = min(URL_CHECK_WORKERS, len(unique_urls))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for url, page in zip(unique_urls, pool.map(fetch_page, unique_urls)):
                pages[url] = page

    cleaned = []

    for record in candidates:
        page = pages.get(record["source_url"], {"ok": False, "status": "n/a", "text": ""})

        quote_verified = quote_in_page(record["evidence_quote"], page["text"])

        record["_url_reachable"] = page["ok"]
        record["_url_status"] = page["status"]
        record["_quote_verified"] = quote_verified

        record["confidence_score"] = deterministic_confidence(
            record,
            record["_match"],
            page["ok"],
            quote_verified,
        )

        cleaned.append(record)

    return cleaned, grounded_sources, rejected


# -----------------------------------------------------------------------------
# Stage 3: validation + deduplication
# -----------------------------------------------------------------------------
def quote_label(value):
    if value is True:
        return "verified"
    if value is False:
        return "not found"
    return "not checked"


def validate_and_deduplicate(records, fields, logs, log_area):
    log_message(logs, log_area, "Running deterministic integrity checks...")

    valid = []
    rejected = 0

    for record in records:
        populated = sum(is_filled(record.get(field)) for field in fields)

        # Require at least half the requested fields to be populated.
        minimum = max(1, len(fields) // 2)

        if populated < minimum:
            rejected += 1
            log_message(
                logs,
                log_area,
                "Rejected record: insufficient field coverage.",
            )
            continue

        if not record.get("source_url") or not record.get("evidence_quote"):
            rejected += 1
            continue

        valid.append(record)

    before = len(valid)
    valid = deduplicate_records(valid, fields)
    duplicates = before - len(valid)

    log_message(
        logs,
        log_area,
        f"Deduplication removed {duplicates} duplicate record(s).",
    )

    # Remove internal columns before presenting the dataframe.
    public_records = []

    for record in valid:
        public_record = {field: record.get(field) for field in fields}
        public_record.update(
            {
                "source_url": record.get("source_url", ""),
                "evidence_title": record.get("evidence_title", ""),
                "evidence_quote": record.get("evidence_quote", ""),
                "grounding_match": record.get("_match", ""),
                "source_reachable": bool(record.get("_url_reachable")),
                "quote_verified": quote_label(record.get("_quote_verified")),
                "confidence_score": record.get("confidence_score", 0),
            }
        )
        public_records.append(public_record)

    return public_records, {"rejected": rejected, "duplicates": duplicates}


# -----------------------------------------------------------------------------
# Full workflow
# -----------------------------------------------------------------------------
def run_agentic_workflow(user_prompt, progress_bar, status_text, log_area, logs):
    try:
        return _run_agentic_workflow(
            user_prompt, progress_bar, status_text, log_area, logs
        )
    except Exception as exc:
        raise_if_quota_error(exc)


def _run_agentic_workflow(user_prompt, progress_bar, status_text, log_area, logs):
    client = get_client()

    if not client:
        raise RuntimeError(
            "Gemini API key is missing or could not initialize the client."
        )

    # Stage 1
    status_text.text("Stage 1/4 · Understanding request & designing schema")
    progress_bar.progress(15)

    plan = plan_schema(client, user_prompt, logs, log_area)

    log_message(
        logs,
        log_area,
        f"Schema: {plan['entity']} → {', '.join(plan['fields'])}",
    )

    # Stage 2
    status_text.text("Stage 2/4 · Searching the live web & collecting evidence")
    progress_bar.progress(40)

    records, sources, rejected_extract = grounded_extract(
        client,
        user_prompt,
        plan,
        logs,
        log_area,
    )

    if not records:
        raise RuntimeError(
            "No sufficiently grounded records were returned. "
            "Try a narrower query or run the search again."
        )

    # Stage 3
    status_text.text("Stage 3/4 · Validating evidence & removing duplicates")
    progress_bar.progress(70)

    final_records, stats = validate_and_deduplicate(
        records,
        plan["fields"],
        logs,
        log_area,
    )

    if not final_records:
        raise RuntimeError(
            "The search returned evidence, but no records passed validation."
        )

    df = pd.DataFrame(final_records)

    # Stage 4
    status_text.text("Stage 4/4 · Building auditable dataset")
    progress_bar.progress(90)

    ordered_columns = (
        plan["fields"]
        + [
            "source_url",
            "evidence_title",
            "evidence_quote",
            "grounding_match",
            "source_reachable",
            "quote_verified",
            "confidence_score",
        ]
    )
    df = df[[column for column in ordered_columns if column in df.columns]]

    avg_confidence = (
        float(df["confidence_score"].mean())
        if "confidence_score" in df.columns and not df.empty
        else 0
    )

    quotes_verified = int((df["quote_verified"] == "verified").sum())

    metrics = {
        "records": len(df),
        "sources": len(sources),
        "average_confidence": avg_confidence,
        "validated": len(df),
        "quotes_verified": quotes_verified,
        "duplicates_removed": stats["duplicates"],
        "rejected": rejected_extract + stats["rejected"],
    }

    progress_bar.progress(100)
    status_text.text("Workflow completed · Evidence-grounded dataset ready")
    log_message(
        logs,
        log_area,
        f"Completed: {len(df)} validated record(s), "
        f"{quotes_verified} quote(s) verified on page, "
        f"{metrics['rejected']} rejected, "
        f"{len(sources)} grounded source(s), "
        f"average evidence score {avg_confidence:.1f}%.",
    )

    return df, plan, metrics, sources


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("⚡ OmniExtract AI")
st.sidebar.markdown(
    '<span class="badge">LIVE WEB + EVIDENCE MODE</span>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

st.sidebar.subheader("Example workflows")

preset = st.sidebar.radio(
    "Load example:",
    [
        "Custom Prompt",
        "AI companies hiring in Europe",
        "Recent FinTech Series A rounds",
        "Developer-tool SaaS companies",
    ],
)

prompt_map = {
    "AI companies hiring in Europe":
        "Find AI companies hiring for senior engineering roles in Europe. "
        "Return company, role, location, and other useful details. "
        "Use current public evidence.",
    "Recent FinTech Series A rounds":
        "Find FinTech startups that announced Series A funding recently. "
        "Return company, funding amount, lead investor, announcement date, "
        "and other useful details. Use primary or reputable sources.",
    "Developer-tool SaaS companies":
        "Find developer-tool SaaS companies with evidence of recent growth, "
        "funding, or developer-focused products. Return useful structured "
        "company information with supporting sources.",
}

default_prompt = prompt_map.get(preset, "")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Confidence is computed from evidence and source checks. "
    "It is not an LLM-generated truth score."
)


# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------
st.title("AI-Powered Evidence Intelligence")
st.caption(
    "Turn natural-language research questions into structured datasets "
    "with traceable web evidence."
)

with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    user_prompt = st.text_area(
        "Describe the dataset you need",
        value=default_prompt,
        placeholder=(
            "Example: Find Indian AI startups that raised Series A in 2026 "
            "with company, amount, investors, date and source evidence."
        ),
        height=110,
    )

    col_run, col_clear = st.columns([4, 1])

    with col_run:
        run_btn = st.button(
            "🚀 Research & Build Dataset",
            type="primary",
            use_container_width=True,
        )

    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state.current_results = None
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------
if run_btn and user_prompt.strip():
    st.subheader("⚙️ Live Research Pipeline")

    progress_bar = st.progress(0)
    status_text = st.empty()
    log_expander = st.expander("Execution trace", expanded=True)
    log_area = log_expander.empty()
    logs = []  # shared with the workflow so error lines are appended, not lost

    try:
        (
            df_results,
            schema_info,
            metrics,
            sources,
        ) = run_agentic_workflow(
            user_prompt,
            progress_bar,
            status_text,
            log_area,
            logs,
        )

        result = {
            "prompt": user_prompt,
            "df": df_results,
            "schema": schema_info,
            "metrics": metrics,
            "sources": sources,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "live",
        }

        st.session_state.current_results = result
        st.session_state.last_successful_results = result
        st.session_state.history.append(result)

    except GeminiQuotaError as exc:
        status_text.error("Gemini rate limit / overload.")
        log_message(logs, log_area, f"QUOTA/OVERLOAD: {exc}")

        st.error(
            "⚠️ Gemini is rate-limited or overloaded. "
            "Retries and fallback models were all tried."
        )
        st.info(
            "Wait a few minutes and run again, or use an API key/project "
            "with available quota."
        )

        cached = st.session_state.last_successful_results
        if cached:
            st.warning(
                "Showing the previous successful dataset as SESSION CACHE. "
                "No new web research was performed."
            )
            cached_copy = dict(cached)
            cached_copy["mode"] = "session-cache"
            st.session_state.current_results = cached_copy

    except Exception as exc:
        status_text.error("Research run failed safely.")
        log_message(logs, log_area, f"ERROR: {type(exc).__name__}: {exc}")

        # Do NOT fabricate records.
        cached = st.session_state.last_successful_results

        if cached:
            st.warning(
                "Live retrieval failed. Showing the previous successful dataset "
                "as a clearly labelled session cache. No fabricated fallback "
                "records were generated."
            )
            cached_copy = dict(cached)
            cached_copy["mode"] = "session-cache"
            st.session_state.current_results = cached_copy
        else:
            st.error(
                "No validated dataset is available. Fix the API/search issue "
                "or retry with a narrower query."
            )


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
if st.session_state.current_results:
    result = st.session_state.current_results
    df = result["df"]
    metrics = result.get("metrics", {})

    st.markdown("---")
    st.subheader("📊 Evidence Dataset")

    if result.get("mode") == "session-cache":
        st.info("SESSION CACHE · This is not a new live research result.")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(
            f'<div class="metric-card"><h4>Validated Records</h4>'
            f'<h2>{len(df)}</h2></div>',
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f'<div class="metric-card"><h4>Grounded Sources</h4>'
            f'<h2>{metrics.get("sources", "—")}</h2></div>',
            unsafe_allow_html=True,
        )

    with c3:
        avg = metrics.get("average_confidence", 0)
        st.markdown(
            f'<div class="metric-card"><h4>Evidence Score</h4>'
            f'<h2>{avg:.1f}%</h2></div>',
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f'<div class="metric-card"><h4>Quotes Verified</h4>'
            f'<h2>{metrics.get("quotes_verified", 0)}/{len(df)}</h2></div>',
            unsafe_allow_html=True,
        )

    with c5:
        st.markdown(
            f'<div class="metric-card"><h4>Duplicates Removed</h4>'
            f'<h2>{metrics.get("duplicates_removed", 0)}</h2></div>',
            unsafe_allow_html=True,
        )

    st.caption(
        f"{metrics.get('rejected', 0)} record(s) were rejected by validation. "
        "Evidence Score is an application-derived signal based on source "
        "grounding, evidence text, URL reachability, quote-on-page check and "
        "field completeness. It is not a guarantee of factual correctness."
    )

    # -------------------------------------------------------------------------
    # Dynamic filtering
    # -------------------------------------------------------------------------
    search_query = st.text_input(
        "🔍 Filter the dataset",
        placeholder="Search any field...",
    )

    if search_query:
        mask = (
            df.astype(str)
            .apply(
                lambda column: column.str.contains(
                    search_query,
                    case=False,
                    regex=False,
                    na=False,
                )
            )
            .any(axis=1)
        )
        filtered_df = df[mask]
    else:
        filtered_df = df

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
    )

    # -------------------------------------------------------------------------
    # Evidence inspection
    # -------------------------------------------------------------------------
    st.subheader("🔎 Evidence Inspector")

    if not filtered_df.empty:
        for idx, row in filtered_df.reset_index(drop=True).iterrows():
            label = row.get(result["schema"]["fields"][0])
            if not is_filled(label):
                label = f"Record {idx + 1}"

            with st.expander(f"{idx + 1}. {label}"):
                left, right = st.columns([2, 1])

                with left:
                    st.markdown("**Supporting evidence**")
                    st.write(row.get("evidence_quote") or "No quote returned.")

                    source_url = row.get("source_url", "")
                    if source_url:
                        st.markdown(
                            f"**Source:** [{row.get('evidence_title') or domain_from_url(source_url)}]"
                            f"({source_url})"
                        )

                with right:
                    st.metric(
                        "Evidence score",
                        f"{float(row.get('confidence_score', 0)):.0f}%",
                    )
                    reachable = row.get("source_reachable", False)
                    st.write(
                        "🟢 Source reachable"
                        if reachable
                        else "🟠 Source not reachable during verification"
                    )

                    verified = row.get("quote_verified", "not checked")
                    if verified == "verified":
                        st.write("🟢 Quote found on source page")
                    elif verified == "not found":
                        st.write("🟠 Quote not found on source page")
                    else:
                        st.write("⚪ Quote could not be checked")

                    st.caption(f"Grounding match: {row.get('grounding_match', '—')}")

                st.markdown("---")

                for field in result["schema"]["fields"]:
                    st.markdown(
                        f"**{field.replace('_', ' ').title()}:** "
                        f"{display_value(row.get(field))}"
                    )

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
    st.subheader("📥 Export")

    col_csv, col_json, _ = st.columns([1, 1, 2])

    with col_csv:
        st.download_button(
            "Download CSV",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="omniextract_evidence_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_json:
        st.download_button(
            "Download JSON",
            data=filtered_df.to_json(
                orient="records",
                indent=2,
                force_ascii=False,
            ),
            file_name="omniextract_evidence_dataset.json",
            mime="application/json",
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# History
# -----------------------------------------------------------------------------
if st.session_state.history:
    with st.expander("🕒 Research History"):
        for idx, item in enumerate(reversed(st.session_state.history)):
            mode = item.get("mode", "live")
            st.markdown(
                f"**Run #{len(st.session_state.history) - idx}** · "
                f"{item['timestamp']} · `{mode}`"
            )
            st.caption(item["prompt"])
