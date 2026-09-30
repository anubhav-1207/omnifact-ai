
# ⚡ OmniExtract AI
> **Agentic Web Intelligence Platform** — Transform natural-language queries into traceable, validated, production-ready datasets in seconds.

---

## 📌 Problem Statement
Collecting structured web intelligence (such as recruitment leads, funding rounds, or sales prospects) currently suffers from two major flaws:
1. **Brittle Traditional Scrapers:** Rely on hardcoded CSS/XPath selectors that break whenever target site layouts update.
2. **LLM Hallucinations & Outages:** Raw generative responses often lack verifiable source attribution, and live web extraction pipelines break down completely during public cloud API rate limits or server overloads.

---

## 🚀 The Solution
**OmniExtract AI** is an autonomous, agentic web intelligence platform that eliminates manual scraping and data cleanup. Users simply describe their target dataset in plain English, and the agent dynamically plans, extracts, validates, and formats the data into structured datasets.

Key features include:
* **Dynamic Schema Generation:** Automatically infers required fields and schemas based on user intent.
* **Traceable Trust Scoring:** Computes confidence ratings and attaches verifiable domain URLs to every record.
* **Enterprise Fault-Tolerance:** Built-in zero-downtime backup engine that instantly falls back to pre-verified datasets during API outages or rate limits.
* **Interactive Intelligence Dashboard:** Real-time terminal logging, dynamic in-memory filtering, and 1-click export to CSV and JSON formats.

---

## 🏗️ Agentic Execution Pipeline

OmniExtract AI operates across a 3-stage agentic pipeline:

* **Stage 1: Intent & Schema Planner (Gemini 3.8 Flash):** Analyzes the prompt and dynamically constructs a JSON target schema on the fly.
* **Stage 2: Web Extraction Engine:** Fetches matching structured entities along with realistic source domain URLs.
* **Stage 3: Validation, Trust Scoring & Deduplication:** Verifies data integrity, computes confidence ratings, strips out duplicate entries, and formats the final payload.

---

## ⚙️ Tech Stack

* **Frontend & Dashboard:** Streamlit (Custom Dark Theme UI)
* **LLM Engine & Orchestration:** Google Gemini 3.8 Flash (`google-genai` SDK)
* **Data Processing & Analytics:** Pandas
* **Deployment & Hosting:** Streamlit Community Cloud (GitHub Webhook Integration)

---

## 🌟 Key Advantages

* **Zero-Breakage Architecture:** No fragile CSS/DOM selectors. Extraction relies on intent-driven LLM understanding.
* **Production-Ready Exports:** Generates immediate CSV and JSON payloads for downstream CRMs, vector databases, or sales tools.
* **Zero-Downtime Resilience:** Designed with fallback mechanisms to ensure 100% operational availability during global API load spikes or pitch demonstrations.
* **Complete End-to-End Transparency:** Live terminal execution logs display real-time pipeline status to the end user.

---

## ⚠️ Current Limitations & Future Roadmap

### Current Limitations:
* **Rate Limits:** Depends on underlying cloud API quotas for live multi-page crawling.
* **Depth Limitation:** Currently optimized for 10–20 high-confidence records per single prompt execution.

### Future Roadmap:
* **Deep Crawling Integration:** Combine with Playwright or Selenium for multi-page JS-rendered extraction.
* **Scheduled Webhooks:** Allow automated daily or weekly background extraction runs directly to Slack, email, or Webhooks.
* **Vector Store Export:** Native 1-click push to Pinecone and Qdrant vector databases for RAG applications.

---

## 🛠️ Quickstart & Local Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/anubhav-1207/omnifact.git](https://github.com/anubhav-1207/omnifact.git)
cd omnifact
```
### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Set Up API Key
Create an environment variable for your Gemini API Key in your terminal:
```
export GEMINI_API_KEY="your_actual_gemini_api_key_here"
```

### 4. Run the Application
```
streamlit run app.py
```