import requests
from src.core.logger import logger
from src.core.schemas import LegalResponseSchema
from src.core.config import LLM_MODEL, MINI_AGENT_MODELS, VECTOR_EMBEDDINGS_MODEL, OLLAMA_BASE_URL


def check_ollama_availability():
    """Verifies targeted agent weights are active inside local background memory pools."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=4)
        if response.status_code != 200:
            return False, "Ollama service engine rejected connectivity checks."
        
        available_models = [m["name"] for m in response.json().get("models", [])]
        required_models = [LLM_MODEL, MINI_AGENT_MODELS, VECTOR_EMBEDDINGS_MODEL]
        missing_models = [m for m in required_models if m not in available_models and f"{m}:latest" not in available_models]
        
        if missing_models:
            return False, f"Missing models: {', '.join(missing_models)}. Execute `ollama pull <model>` to correct."
        return True, ""
    except Exception as e:
        return False, f"Ollama Connection Error: {str(e)}"


def format_legal_response(response: LegalResponseSchema) -> str:
        """Converts the structured legal response into a user-friendly markdown format."""
        try:
            # Urgency mapping based on schema output
            urgency_label = {
                "High": "STATUS: HIGH URGENCY",
                "Medium": "STATUS: MEDIUM URGENCY",
                "Low": "STATUS: LOW URGENCY",
                "Not Specified": "STATUS: GENERAL NOTICE"
            }.get(response.urgency_level, "STATUS: GENERAL NOTICE")

            parts = [
                f"## {urgency_label}\n\n",
                f"### Legal Executive Summary\n{response.summary}\n\n",
                "---"
            ]
            
            for sec in response.sections:
                sec_md = (
                    f"\n### Statutory Provision: {sec.statute} Section {sec.section_number}\n"
                    f"**Official Designation:** {sec.section_name}\n\n"
                    f"#### Detailed Legal Interpretation\n{sec.interpretation}\n\n"
                    f"**Illustrative Case/Scenario:** *{sec.scenario}*\n\n"
                    f"**Legal Details:**\n"
                    f"- **Punishment:** {sec.punishment}\n"
                    f"- **Consequences:** {sec.legal_consequences}\n"
                    f"- **Bailable:** `{sec.is_bailable}` | **Cognizable:** `{sec.is_cognizable}` | **Compoundable:** `{sec.is_compoundable}`\n"
                    f"- **Triable By:** {sec.triable_by}\n"
                    "---"
                )
                parts.append(sec_md)
                
            if response.related_sections:
                related = ", ".join(response.related_sections)
                parts.append(f"\n**Further Reading (Related):** {related}\n")
                
            if response.next_steps:
                steps = "\n".join([f"- {step}" for step in response.next_steps])
                parts.append(f"\n#### Recommended Next Steps\n{steps}\n")

            parts.append(f"\n\n> **Disclaimer:** *{response.limitations if response.limitations else 'This system provides context-grounded legal information based on IPC/BNS. It does not constitute professional legal counsel.'}*")
            
            return "".join(parts)
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}", exc_info=True)
            raise e