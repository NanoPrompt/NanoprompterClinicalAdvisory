from flask import Flask, render_template, request
from engine import NanoprompterEngine

app = Flask(__name__, template_folder=".") 
engine = NanoprompterEngine()

def local_inference_model(prompt: str) -> str:
    return (
        "SUBJECTIVE: Patient presents with acute, localized cluster-type symptoms.\n"
        "OBJECTIVE: Baseline vitals stable. Visual field assessment clear.\n"
        "ASSESSMENT: Acute clinical migraine variations.\n"
        "PLAN: Initiate safety hydration protocol and schedule targeted specialist review."
    )

@app.route("/", methods=["GET", "POST"])
def home():
    processed_prompt = ""
    raw_output = ""
    validation_status = ""
    violations = []
    user_input = ""

    if request.method == "POST":
        user_input = request.form.get("clinical_input", "")
        if user_input:
            pipeline_result = engine.run_pipeline(user_input, local_inference_model)
            
            processed_prompt = pipeline_result["processed_prompt"]
            raw_output = pipeline_result["raw_output"]
            validation_status = "PASSED" if pipeline_result["safety_validation"]["passed"] else "FAILED"
            violations = pipeline_result["safety_validation"]["violations"]

    return render_template(
        "index.html",
        user_input=user_input,
        processed_prompt=processed_prompt,
        raw_output=raw_output,
        validation_status=validation_status,
        violations=violations
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
