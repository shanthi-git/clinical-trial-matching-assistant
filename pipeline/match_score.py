# pipeline/match_score.py
from sentence_transformers import SentenceTransformer, util
from models.schemas import PatientProfile, TrialCriteria

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def profile_to_text(p: PatientProfile) -> str:
    return f"Diagnosis: {', '.join(p.diagnosis)}. Age: {p.age}. " \
           f"Biomarkers: {', '.join(p.biomarkers)}. " \
           f"Prior treatments: {', '.join(p.prior_treatments)}."

def criteria_to_text(c: TrialCriteria) -> str:
    return f"Requires diagnosis: {', '.join(c.diagnosis)}. " \
           f"Age range: {c.min_age}-{c.max_age}. " \
           f"Biomarkers: {', '.join(c.biomarkers)}. " \
           f"Required treatments: {', '.join(c.prior_treatments_required)}. " \
           f"Excluded treatments: {', '.join(c.prior_treatments_excluded)}."

def hard_filter(patient: PatientProfile, criteria: TrialCriteria) -> bool:
    """Disqualify trials on hard constraints before scoring."""
    if criteria.min_age and patient.age < criteria.min_age:
        return False
    if criteria.max_age and patient.age > criteria.max_age:
        return False
    if set(patient.prior_treatments) & set(criteria.prior_treatments_excluded):
        return False
    return True

def score_trial(patient: PatientProfile, criteria: TrialCriteria) -> float:
    if not hard_filter(patient, criteria):
        return 0.0
    p_emb = embedder.encode(profile_to_text(patient), convert_to_tensor=True)
    c_emb = embedder.encode(criteria_to_text(criteria), convert_to_tensor=True)
    return float(util.cos_sim(p_emb, c_emb).item())

def rank_trials(patient: PatientProfile, trials_with_criteria: list[tuple[dict, TrialCriteria]]) -> list[dict]:
    scored = []
    for trial, criteria in trials_with_criteria:
        score = score_trial(patient, criteria)
        scored.append({**trial, "match_score": round(score, 3)})
    return sorted(scored, key=lambda t: t["match_score"], reverse=True)


if __name__ == "__main__":
    from ingestion.fetch_trials import fetch_trials
    from pipeline.extract_criteria import extract_structured_criteria

    trials = fetch_trials("lung cancer", max_results=3)

    patient = PatientProfile(
        diagnosis=["non-small cell lung cancer"],
        age=58,
        biomarkers=["KRAS G12C"],
        prior_treatments=[]
    )

    trials_with_criteria = []
    for t in trials:
        if not t["eligibility_text"]:
            continue
        criteria = extract_structured_criteria(t["eligibility_text"])
        trials_with_criteria.append((t, criteria))

    ranked = rank_trials(patient, trials_with_criteria)

    print(f"Patient: {patient.diagnosis}, age {patient.age}, biomarkers {patient.biomarkers}\n")
    for r in ranked:
        print(f"Score: {r['match_score']}  |  {r['nct_id']}  |  {r['title']}")