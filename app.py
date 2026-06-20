from flask import Flask, render_template, request
from engine import NanoprompterEngine

app = Flask(__name__, template_folder=".") 
engine = NanoprompterEngine()

# Updated mock model to simulate dynamic structures based on context
def local_inference_model(prompt: str, framework_type: str) -> str:
    if framework_type.upper() == "SBAR":
        return (
            "SITUATION: Patient presents with acute, localized cluster-type symptoms.\n"
            "BACKGROUND: History of chronic headaches, non-responsive to baseline NSAID treatment.\n"
            "ASSESSMENT: Signs match severe variant criteria requiring abortive optimization protocols.\n"
            "RECOMMENDATION: Initiate immediate safe hydration tracking and schedule targeted clinical review."
        )
    else:
        return (
            "SUBJECTIVE: Patient reports localized cluster-type head symptoms.\n"
            "OBJECTIVE: Stable tracking parameters observed in available documentation.\n"
            "ASSESSMENT: Acute clinical migraine variations noted.\n"
            "PLAN: Proceed with core specialized optimization protocol."
        )

@app.route("/", methods=["GET", "POST"])
def home():
    processed_prompt = ""
    raw_output = ""
    validation_status = ""
    violations = []
    user_input = ""
    selected_framework = "SOAP"  # default setting

    if request.method == "POST":
        user_input = request.form.get("clinical_input", "")
        selected_framework = request.form.get("framework_type", "SOAP")
        
        if user_input:
            # Pass the selected framework right into the processing engine
            pipeline_result = engine.run_pipeline(user_input, local_inference_model, selected_framework)
            
            processed_prompt = pipeline_result["processed_prompt"]
            raw_output = pipeline_result["raw_output"]
            validation_status = "PASSED" if pipeline_result["safety_validation"]["passed"] else "FAILED"
            violations = pipeline_result["safety_validation"]["violations"]

    return render_template(
        "index.html",
        user_input=user_input,
        selected_framework=selected_framework,
        processed_prompt=processed_prompt,
        raw_output=raw_output,
        validation_status=validation_status,
        violations=violations
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
