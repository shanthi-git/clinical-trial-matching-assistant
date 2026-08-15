# Clinical Trial Matching Assistant

An AI-powered system that matches patient profiles against actively recruiting clinical trials by parsing unstructured eligibility criteria and scoring semantic fit.

## What it does

1. **Ingests** live trial data from the [ClinicalTrials.gov API](https://clinicaltrials.gov/data-api/api) for a given condition
2. **Extracts** structured eligibility fields (diagnosis, age range, biomarkers, prior treatment requirements/exclusions) from messy free-text criteria using an LLM with structured output (LangChain + Pydantic)
3. **Scores** a patient profile against each trial using sentence-embedding similarity, gated by hard filters (age, explicit treatment exclusions)
4. **Serves** results through a Flask REST API

## Tech stack

- **Flask** — REST API layer
- **LangChain** — LLM orchestration, structured output parsing
- **Ollama (llama3.1)** — local LLM for entity extraction, no API key required (swappable for Claude/GPT/Gemini — see Design Notes)
- **sentence-transformers** — embedding-based semantic similarity (`all-MiniLM-L6-v2`)
- **Pydantic** — schema validation for patient/trial data

## Project structure

\`\`\`
clinical-trial-matcher/
├── app.py                     # Flask app, REST endpoints
├── ingestion/
│   └── fetch_trials.py        # Pulls trial data from ClinicalTrials.gov
├── pipeline/
│   ├── extract_criteria.py    # LLM chain: unstructured text -> structured entities
│   └── match_score.py         # Embedding similarity + ranking
├── models/
│   └── schemas.py