# Clinical Trial Matching Assistant

An AI-powered system that matches patient profiles against actively recruiting clinical trials by parsing unstructured eligibility criteria and scoring semantic fit.

## What it does

1. **Ingests** live trial data from the [ClinicalTrials.gov API](https://clinicaltrials.gov/data-api/api) for a given condition
2. **Extracts** structured eligibility fields (diagnosis, age range, biomarkers, prior treatment requirements/exclusions) from messy free-text criteria using an LLM with structured output (LangChain + Pydantic)
3. **Scores** a patient profile against each trial using sentence-embedding similarity, boosted by exact biomarker overlap, and gated by hard filters (age, explicit treatment exclusions)
4. **Serves** results through a Flask REST API

## Tech stack

- **Flask** — REST API layer
- **LangChain** — LLM orchestration, structured output parsing
- **Ollama (llama3.1)** — local LLM for entity extraction, no API key required (swappable for Claude/GPT/Gemini — see Design Notes)
- **sentence-transformers** — embedding-based semantic similarity (`all-MiniLM-L6-v2`)
- **Pydantic** — schema validation for patient/trial data

## Project structure
linical-trial-matcher/
├── app.py # Flask app, REST endpoints
├── ingestion/
│ └── fetch_trials.py # Pulls trial data from ClinicalTrials.gov
├── pipeline/
│ ├── extract_criteria.py # LLM chain: unstructured text -> structured entities
│ └── match_score.py # Embedding similarity + biomarker boost + ranking
├── models/
│ └── schemas.py # Pydantic models for patient profile & trial criteria
└── requirements.txt


## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

This project uses [Ollama](https://ollama.com) to run the LLM locally — no API key needed:
```bash
ollama pull llama3.1
```

## Running it

```bash
python app.py
```

Then, in a separate terminal:
```bash
curl -X POST http://localhost:5000/api/match \
  -H "Content-Type: application/json" \
  -d '{
    "patient": {
      "diagnosis": ["non-small cell lung cancer"],
      "age": 58,
      "biomarkers": ["KRAS G12C"],
      "prior_treatments": []
    },
    "condition": "lung cancer"
  }'
```

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/patient/intake` | Validate and echo a patient profile |
| GET | `/api/trials/search?condition=X` | Search recruiting trials for a condition |
| POST | `/api/match` | Full pipeline: fetch trials, extract criteria, rank by match |

## Design notes

**Why a hard filter before semantic scoring?** Pure embedding similarity ranks trials by *topical* closeness, not by whether a patient actually satisfies specific eligibility rules. A trial could score highly on similarity just because it shares cancer-related vocabulary with the patient profile, even if the patient is explicitly excluded (e.g. by age or a disqualifying prior treatment). `hard_filter()` disqualifies trials on non-negotiable constraints before embeddings are used to rank what's left.

**Why a biomarker boost on top of embedding similarity?** In early testing, pure embedding similarity sometimes ranked a trial highly just because it shared general topical vocabulary (cancer terms, "advanced," "solid tumors") with the patient profile — even when a *different* trial required the patient's exact biomarker (e.g. KRAS G12C). Embeddings capture topical closeness well, but don't reliably weight exact-match criteria the way real clinical eligibility requires. `biomarker_boost()` adds an explicit score bonus for exact biomarker overlap between patient and trial, layered on top of the embedding score rather than hoping the embedding infers the significance of an exact term match. This is a simple, transparent, interpretable fix — a production system could extend the same pattern to other exact-match fields (e.g. specific mutation subtypes, prior therapy classes).

**Why a local LLM (Ollama)?** Keeps the project runnable by anyone without API costs or key management. The LangChain abstraction (`with_structured_output`) means swapping to a hosted model (Claude, GPT, Gemini) for production use is a one-line change — the extraction prompt and schema stay identical.

## Disclaimer

This is a portfolio/ project using public ClinicalTrials.gov data. It is not a clinical decision-support tool and should not be used to make real treatment decisions.

