from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class QueryClassification(BaseModel):
    """Binary classification of whether a query is related to Indian Law."""
    is_legal: bool = Field(description="True if the query is about Indian law, IPC, crimes, etc.")
    reasoning: str = Field(description="Brief explanation of why this was or wasn't classified as legal.")
    confidence_score: float = Field(description="Confidence score STRICTLY between 0 and 1.", ge=0, le=1)


class IntentClassification(BaseModel):
    """Classification of legal intent/category."""
    intent: Literal["fraud", "theft", "assault", "murder", "cybercrime", "extortion", "harassment", "property_dispute", "legal_advice", "other"]
    confidence: float = Field(description="Confidence score for the intent classification.", ge=0, le=1)
    detected_entities: List[str] = Field(description="List of key legal entities or objects mentioned (e.g., 'FIR', 'Cheque', 'Weapon').")
    target_statutes: List[Literal["bns", "bnss", "bsa", "ipc", "judgments"]] = Field(description="The primary legal collections to search.")


class QueryExpansion(BaseModel):
    """List of alternative queries for vector search."""
    queries: List[str] = Field(description="Exactly 10 alternative versions of the question.")


class LegalSectionDetail(BaseModel):
    statute: str = Field(description="The name of the Act (e.g., BNS, IPC, BNSS).")
    section_number: str
    section_name: str
    interpretation: str = Field(description="A clear, simplified explanation of the legal section.")
    scenario: str = Field(description="A realistic example or scenario where this section applies.")
    punishment: str
    legal_consequences: str
    is_bailable: Literal["Bailable", "Non-Bailable", "Not Specified"] = Field(description="Whether the offense is Bailable or Non-Bailable.")
    is_cognizable: Literal["Cognizable", "Non-Cognizable", "Not Specified"] = Field(description="Whether the offense is Cognizable or Non-Cognizable.")
    triable_by: str = Field(description="The type of Magistrate or Court that trials this offense.")
    is_compoundable: Literal["Compoundable", "Non-Compoundable", "Not Specified"] = Field(description="Whether the offense can be settled (Compoundable) or not.")


class LegalResponseSchema(BaseModel):
    acknowledgment: str
    summary: str = Field(description="A 2-3 sentence high-level summary of the legal position.")
    sections: List[LegalSectionDetail]
    related_sections: Optional[List[str]] = Field(description="List of related sections for further reading.")
    next_steps: List[str] = Field(description="A list of actionable next steps for the user.")
    urgency_level: Literal["Low", "Medium", "High"]
    limitations: str = Field(description="State what information was missing or not found in the context.")