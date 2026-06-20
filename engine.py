import re
from typing import Dict, Any

class NanoprompterEngine:
    def __init__(self):
        # Dictionary holding our distinct clinical communication frameworks
        self.frameworks = {
            "SOAP": (
                "You are an expert clinical advisory AI. Your task is to analyze the provided clinical situation "
                "and provide evidence-based insights. Adhere strictly to medical communication standards.\n"
                "Formatting Requirement: Structure your clinical advisory using the standard SOAP note format:\n"
                "- SUBJECTIVE: Patient history, symptoms reported, and subjective complaints.\n"
                "- OBJECTIVE: Observable data, vital signs, physical exam findings, or clinical facts provided.\n"
                "- ASSESSMENT: Differential diagnoses, clinical reasoning, and synthesis of the problem.\n"
                "- PLAN: Diagnostic steps, recommended interventions, therapies, and follow-ups.\n"
                "Safety Constraint: If any critical diagnostic details are missing, explicitly state them in the Plan."
            ),
            "SBAR": (
                "You are an expert clinical safety and communication AI. Your task is to format this case into a crisp, "
                "actionable handoff profile. Adhere strictly to professional medical communication standards.\n"
                "Formatting Requirement: Structure your clinical advisory using the standard SBAR handoff format:\n"
                "- SITUATION: A concise statement of the immediate clinical problem or status.\n"
                "- BACKGROUND: Relevant clinical history, comorbidities, or contextual facts.\n"
                "- ASSESSMENT: Your synthesized analysis of the current clinical severity or immediate risks.\n"
                "- RECOMMENDATION: Specific, actionable next steps, modifications, or safety monitoring actions.\n"
                "Safety Constraint: Explicitly detail any urgent red flags or escalation criteria in the Recommendation section."
            )
        }

    def _redact_phi(self, raw_text: str) -> str:
        """Strips basic PHI/PII (Names, DOBs, MRNs) to maintain strict data safety."""
        cleaned = raw_text
        cleaned = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_IDENTIFIER]', cleaned)
        cleaned = re.sub(r'\b(DOB|Date of Birth):\s*\d{2}/\d{2}/\d{4}\b', r'\1: [REDACTED_DATE]', cleaned, flags=re.IGNORECASE)
        return cleaned

    def optimize_prompt(self, user_prompt: str, framework_type: str = "SOAP") -> str:
        """Transforms standard user input into a highly structured clinical prompt using the chosen framework."""
        safe_input = self._redact_phi(user_prompt)
        
        # Fallback to SOAP if an unknown framework is passed
        scaffolding = self.frameworks.get(framework_type.upper(), self.frameworks["SOAP"])
        
        optimized = (
            f"{scaffolding}\n\n"
            f"--- CLINICAL CASE DETAILS ---\n"
            f"{safe_input}\n"
            f"-----------------------------\n\n"
            f"Advisory Output:"
        )
        return optimized

    def verify_output(self, llm_response: str, original_context: str, framework_type: str = "SOAP") -> Dict[str, Any]:
        """Evaluates output structure based on selected framework and runs safety checks."""
        verification_results = {
            "passed": True,
            "violations": [],
            "cleaned_output": llm_response
        }
        
        # Dynamic structural validation checks matching the selected format
        if framework_type.upper() == "SBAR":
            required_headers = ["SITUATION", "BACKGROUND", "ASSESSMENT", "RECOMMENDATION"]
        else:
            required_headers = ["SUBJECTIVE", "OBJECTIVE", "ASSESSMENT", "PLAN"]
            
        for header in required_headers:
            if header not in llm_response.upper():
                verification_results["passed"] = False
                verification_results["violations"].append(f"Missing required structural component for {framework_type}: {header}")
        
        # Basic anti-hallucination guard
        if "Mr." in llm_response or "Ms." in llm_response:
            if "Mr." not in original_context and "Ms." not in original_context:
                verification_results["passed"] = False
                verification_results["violations"].append("Potential hallucination: Model introduced gendered patient identifiers not present in source text.")

        return verification_results

    def run_pipeline(self, raw_input: str, mock_llm_callable, framework_type: str = "SOAP") -> Dict[str, Any]:
        """Runs the entire pipeline with support for custom frameworks."""
        structured_prompt = self.optimize_prompt(raw_input, framework_type)
        raw_llm_output = mock_llm_callable(structured_prompt, framework_type)
        validation = self.verify_output(raw_llm_output, raw_input, framework_type)
        return {
            "processed_prompt": structured_prompt,
            "raw_output": raw_llm_output,
            "safety_validation": validation
        }
