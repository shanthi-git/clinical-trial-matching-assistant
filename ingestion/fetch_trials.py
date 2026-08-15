# ingestion/fetch_trials.py
import requests
import json
from datetime import datetime

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

def fetch_trials(condition: str, max_results: int = 20) -> list[dict]:
    """Fetch active trials for a given condition."""
    params = {
        "query.cond": condition,
        "filter.overallStatus": "RECRUITING",
        "pageSize": max_results,
        "fields": "NCTId,BriefTitle,EligibilityCriteria,Condition,MinimumAge,MaximumAge"
    }
    resp = requests.get(BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    studies = resp.json().get("studies", [])

    trials = []
    for s in studies:
        protocol = s.get("protocolSection", {})
        trials.append({
            "nct_id": protocol.get("identificationModule", {}).get("nctId"),
            "title": protocol.get("identificationModule", {}).get("briefTitle"),
            "eligibility_text": protocol.get("eligibilityModule", {}).get("eligibilityCriteria", ""),
            "conditions": protocol.get("conditionsModule", {}).get("conditions", []),
        })
    return trials

if __name__ == "__main__":
    trials = fetch_trials("lung cancer", max_results=3)
    print(f"Fetched {len(trials)} trials\n")
    for t in trials:
        print(f"NCT ID: {t['nct_id']}")
        print(f"Title: {t['title']}")
        print(f"Conditions: {t['conditions']}")
        print(f"Eligibility text (first 200 chars): {t['eligibility_text'][:200]}...")
        print("-" * 60)