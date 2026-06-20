import re
from typing import Dict, Any

class NanoprompterEngine:
    def __init__(self):
        self.clinical_scaffolding = (
            "You are an expert clinical advisory AI. Your task is to analyze the provided clinical situation "
            "and provide evidence-based insights. Adhere strictly to medical communication standards.\n"
            "Formatting Requirement: Structure your clinical advisory using the standard SOAP note format "
            "(Subjective, Objective, Assessment, Plan).\n"
            "Safety Constraint: If any critical diagnostic details are missing, explicitly state them in the Plan."
        )

    def _redact_phi(self, raw_text: str) -> str:
        cleaned = raw_text
        cleaned = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_IDENTIFIER]', cleaned)
        cleaned = re.sub(r'\b(DOB|Date of Birth):\s*\d{2}/\d{2}/\d{4}\b', r'\1: [REDACTED_DATE]', cleaned, flags=re.IGNORECASE)
        return cleaned

    def optimize_prompt(self, user_prompt: str) -> str:
        safe_input = self._redact_phi(user_prompt)
        optimized = (
            f"{self.clinical_scaffolding}\n\n"
            f"--- CLINICAL CASE DETAILS ---\n"
            f"{safe_input}\n"
            f"-----------------------------\n\n"
            f"Advisory Output:"
        )
        return optimized

    def verify_output(self, llm_response: str, original_context: str) -> Dict[str, Any]:
        verification_results = {
            "passed": True,
            "violations": [],
            "cleaned_output": llm_response
        }
        required_headers = ["SUBJECTIVE", "OBJECTIVE", "ASSESSMENT", "PLAN"]
        for header in required_headers:
            if header not in llm_response.upper():
                verification_results["passed"] = False
                verification_results["violations"].append(f"Missing required clinical structural component: {header}")
        
        if "Mr." in llm_response or "Ms." in llm_response:
            if "Mr." not in original_context and "Ms." not in original_context:
                verification_results["passed"] = False
                verification_results["violations"].append("Potential hallucination: Model introduced gendered patient identifiers not present in source text.")

        return verification_results

    def run_pipeline(self, raw_input: str, mock_llm_callable) -> Dict[str, Any]:
        structured_prompt = self.optimize_prompt(raw_input)
        raw_llm_output = mock_llm_callable(structured_prompt)
        validation = self.verify_output(raw_llm_output, raw_input)
        return {
            "processed_prompt": structured_prompt,
            "raw_output": raw_llm_output,
            "safety_validation": validation
        }
