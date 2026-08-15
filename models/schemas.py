# models/schemas.py
from pydantic import BaseModel, Field
from typing import Optional

class TrialCriteria(BaseModel):
    diagnosis: list[str] = Field(description="Diagnoses required for eligibility")
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    biomarkers: list[str] = Field(default_factory=list)
    prior_treatments_required: list[str] = Field(default_factory=list)
    prior_treatments_excluded: list[str] = Field(default_factory=list)

class PatientProfile(BaseModel):
    diagnosis: list[str]
    age: int
    biomarkers: list[str] = Field(default_factory=list)
    prior_treatments: list[str] = Field(default_factory=list)