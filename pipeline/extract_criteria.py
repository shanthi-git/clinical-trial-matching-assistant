# pipeline/extract_criteria.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import TrialCriteria

llm = ChatOllama(model="llama3.1", temperature=0)
structured_llm = llm.with_structured_output(TrialCriteria)

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You extract structured eligibility data from clinical trial text. "
     "Identify diagnoses, age range, required biomarkers, and prior treatment "
     "requirements/exclusions. Only use information explicitly stated."),
    ("human", "Eligibility criteria:\n\n{eligibility_text}")
])

def extract_structured_criteria(eligibility_text: str) -> TrialCriteria:
    chain = EXTRACTION_PROMPT | structured_llm
    return chain.invoke({"eligibility_text": eligibility_text})

