from flask import Flask, request, jsonify
from ingestion.fetch_trials import fetch_trials
from pipeline.extract_criteria import extract_structured_criteria
from pipeline.match_score import rank_trials
from models.schemas import PatientProfile

app = Flask(__name__)

@app.route("/api/patient/intake", methods=["POST"])
def patient_intake():
    data = request.get_json()
    profile = PatientProfile(**data)
    return jsonify(profile.model_dump()), 200

@app.route("/api/trials/search", methods=["GET"])
def trial_search():
    condition = request.args.get("condition")
    if not condition:
        return jsonify({"error": "condition query param required"}), 400
    trials = fetch_trials(condition)
    return jsonify(trials), 200

@app.route("/api/match", methods=["POST"])
def match():
    body = request.get_json()
    patient = PatientProfile(**body["patient"])
    condition = body.get("condition", patient.diagnosis[0])

    trials = fetch_trials(condition)
    trials_with_criteria = [
        (t, extract_structured_criteria(t["eligibility_text"]))
        for t in trials if t["eligibility_text"]
    ]
    ranked = rank_trials(patient, trials_with_criteria)
    return jsonify(ranked[:10]), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)