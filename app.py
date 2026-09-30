import os
import json
import time
import re
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
#   Gemini + Google Search grounding
#        ↓
#   Structured extraction + evidence
#        ↓
#   Deterministic validation
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
MODEL = "gemini-3.8-flash"
MAX_RECORDS = 8
REQUEST_TIMEOUT = 8

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
    """
    Extract URLs/titles from Gemini grounding metadata.
    The SDK exposes grounding metadata on the candidate in current
    generate_content responses.
    """
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

    # De-duplicate source URLs.
    unique = {}
    for source in sources:
        unique[source["url"]] = source
    return list(unique.values())


def validate_url(url):
    """Cheap reachability check; does not claim semantic verification."""
    url = clean_url(url)
    if not url:
        return False, "invalid URL"

    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
            headers={"User-Agent": "OmniExtractAI/1.0"},
        )
        ok = 200 <= response.status_code < 400
        return ok, f"HTTP {response.status_code}"
    except requests.RequestException as exc:
        return False, type(exc).__name__


def deterministic_confidence(record, source_was_grounded, url_reachable):
    """
    Evidence-derived score.
    This is deliberately deterministic and capped below 100%.
    It is NOT a claim that the factual content is guaranteed true.
    """
    score = 0

    # Source exists and has a valid URL.
    if clean_url(record.get("source_url")):
        score += 20

    # Gemini grounding actually returned this URL.
    if source_was_grounded:
        score += 30

    # Evidence quote is present.
    if len(str(record.get("evidence_quote", "")).strip()) >= 20:
        score += 25

    # Source is reachable now.
    if url_reachable:
        score += 15

    # Basic record completeness.
    required = record.get("_fields", [])
    if required:
        populated = sum(
            bool(str(record.get(field, "")).strip())
            for field in required
        )
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

    response = client.models.generate_content(
        model=MODEL,
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

    response = client.models.generate_content(
        model=MODEL,
        contents=extraction_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            tools=[
                types.Tool(
                    google_search=types.GoogleSearch()
                )
            ],
        ),
    )

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

    grounded_urls = {
        clean_url(source["url"]) for source in grounded_sources if source.get("url")
    }

    cleaned = []

    for raw in records[:MAX_RECORDS]:
        if not isinstance(raw, dict):
            continue

        record = dict(raw)

        # Normalize source fields.
        source_url = clean_url(record.get("source_url", ""))
        record["source_url"] = source_url
        record["evidence_quote"] = str(record.get("evidence_quote", "") or "").strip()
        record["evidence_title"] = str(record.get("evidence_title", "") or "").strip()
        record["_fields"] = fields

        # Only accept a source URL that actually appears in grounding metadata.
        source_was_grounded = source_url in grounded_urls

        # If the model returned a URL without exact normalization match,
        # try matching by canonical URL string.
        if not source_was_grounded and source_url:
            source_was_grounded = any(
                source_url.rstrip("/") == url.rstrip("/")
                for url in grounded_urls
            )

        record["_source_grounded"] = source_was_grounded

        if not source_url or not record["evidence_quote"]:
            log_message(
                logs,
                log_area,
                "Rejected record: missing source URL or evidence quote.",
            )
            continue

        # Never silently substitute a source URL from nowhere.
        if not source_was_grounded:
            log_message(
                logs,
                log_area,
                f"Rejected record: source was not present in grounding metadata ({domain_from_url(source_url)}).",
            )
            continue

        reachable, status = validate_url(source_url)
        record["_url_reachable"] = reachable
        record["_url_status"] = status

        record["confidence_score"] = deterministic_confidence(
            record,
            source_was_grounded,
            reachable,
        )

        cleaned.append(record)

    return cleaned, grounded_sources


# -----------------------------------------------------------------------------
# Stage 3: validation + deduplication
# -----------------------------------------------------------------------------
def validate_and_deduplicate(records, fields, logs, log_area):
    log_message(logs, log_area, "Running deterministic integrity checks...")

    valid = []

    for record in records:
        populated = sum(
            bool(str(record.get(field, "")).strip())
            for field in fields
        )

        # Require at least half the requested fields to be populated.
        minimum = max(1, len(fields) // 2)

        if populated < minimum:
            log_message(
                logs,
                log_area,
                "Rejected record: insufficient field coverage.",
            )
            continue

        if not record.get("source_url") or not record.get("evidence_quote"):
            continue

        valid.append(record)

    before = len(valid)
    valid = deduplicate_records(valid, fields)
    removed = before - len(valid)

    log_message(
        logs,
        log_area,
        f"Deduplication removed {removed} duplicate record(s).",
    )

    # Remove internal columns before presenting the dataframe.
    public_records = []

    for record in valid:
        public_record = {
            field: record.get(field)
            for field in fields
        }
        public_record.update(
            {
                "source_url": record.get("source_url", ""),
                "evidence_title": record.get("evidence_title", ""),
                "evidence_quote": record.get("evidence_quote", ""),
                "source_reachable": bool(record.get("_url_reachable")),
                "confidence_score": record.get("confidence_score", 0),
            }
        )
        public_records.append(public_record)

    return public_records


# -----------------------------------------------------------------------------
# API failure classification
# -----------------------------------------------------------------------------
class GeminiQuotaError(RuntimeError):
    """Raised when Gemini rejects a request because of quota/rate limiting."""


def is_quota_error(exc):
    """Detect common Gemini/API quota and rate-limit failures."""
    status_code = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)

    # Google APIs commonly use HTTP 429 / RESOURCE_EXHAUSTED for quota limits.
    if status_code == 429 or code == 429:
        return True

    message = str(exc).lower()
    quota_markers = (
        "resource_exhausted",
        "resource exhausted",
        "quota exceeded",
        "quota_exceeded",
        "rate limit",
        "rate_limit",
        "too many requests",
        "too_many_requests",
        "429",
        "requests per minute",
        "requests per day",
        "tokens per minute",
        "tokens per day",
        "capacity",
        "temporarily unavailable",
    )
    return any(marker in message for marker in quota_markers)


def raise_if_quota_error(exc):
    """Convert a quota/rate-limit exception into a user-facing exception."""
    if is_quota_error(exc):
        raise GeminiQuotaError(
            "Gemini API quota or rate limit was reached. "
            "Google is temporarily refusing requests because the API has "
            "hit its usage limit or is under high demand. "
            "Wait and retry, or use an API project/key with available quota."
        ) from exc

    raise exc


# -----------------------------------------------------------------------------
# Full workflow
# -----------------------------------------------------------------------------
def run_agentic_workflow(user_prompt, progress_bar, status_text, log_area):
    try:
        return _run_agentic_workflow(
            user_prompt, progress_bar, status_text, log_area
        )
    except Exception as exc:
        raise_if_quota_error(exc)


def _run_agentic_workflow(user_prompt, progress_bar, status_text, log_area):
    client = get_client()

    if not client:
        raise RuntimeError(
            "Gemini API key is missing or could not initialize the client."
        )

    logs = []

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

    records, sources = grounded_extract(
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

    final_records = validate_and_deduplicate(
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

    # Stable ordering.
    ordered_columns = (
        plan["fields"]
        + [
            "source_url",
            "evidence_title",
            "evidence_quote",
            "source_reachable",
            "confidence_score",
        ]
    )
    df = df[[column for column in ordered_columns if column in df.columns]]

    avg_confidence = (
        float(df["confidence_score"].mean())
        if "confidence_score" in df.columns and not df.empty
        else 0
    )

    metrics = {
        "records": len(df),
        "sources": len(sources),
        "average_confidence": avg_confidence,
        "validated": len(df),
        "duplicates_removed": len(records) - len(df),
    }

    progress_bar.progress(100)
    status_text.text("Workflow completed · Evidence-grounded dataset ready")
    log_message(
        logs,
        log_area,
        f"Completed: {len(df)} validated record(s), "
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
        status_text.error("Gemini quota / rate limit reached.")
        log_message(
            [],
            log_area,
            f"QUOTA/RATE-LIMIT: {exc}",
        )

        st.error(
            "⚠️ Gemini API quota or rate limit reached. "
            "Google is refusing the request because the API usage limit was "
            "reached or the service is experiencing high demand."
        )
        st.info(
            "Try again later, reduce request frequency, or use a Gemini API "
            "project/key with available quota."
        )

        # A previous result can still be displayed, but it is explicitly cached.
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
        log_message(
            [],
            log_area,
            f"ERROR: {type(exc).__name__}: {exc}",
        )

        # Do NOT fabricate records.
        # If there is a previous successful run, expose it explicitly as cache.
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

    c1, c2, c3, c4 = st.columns(4)

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
            f'<div class="metric-card"><h4>Duplicates Removed</h4>'
            f'<h2>{metrics.get("duplicates_removed", 0)}</h2></div>',
            unsafe_allow_html=True,
        )

    st.caption(
        "Evidence Score is an application-derived signal based on source "
        "grounding, evidence text, URL reachability and field completeness. "
        "It is not a guarantee of factual correctness."
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
            label = row.get(
                result["schema"]["fields"][0],
                f"Record {idx + 1}",
            )

            with st.expander(f"{idx + 1}. {label}"):
                left, right = st.columns([2, 1])

                with left:
                    st.markdown("**Supporting evidence**")
                    st.write(row.get("evidence_quote", "No quote returned."))

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

                st.markdown("---")

                for field in result["schema"]["fields"]:
                    st.markdown(
                        f"**{field.replace('_', ' ').title()}:** "
                        f"{row.get(field, '')}"
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
            file_name="omnifact_evidence_dataset.csv",
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
            file_name="omnifact_evidence_dataset.json",
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
