from flask import Flask, render_template, request
import os
import json
import asyncio
import sys
from dotenv import load_dotenv
from main import run_analysis

load_dotenv()

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    df_html = None
    findings = None
    result_json_str = None
    openai_insights_str = None
    similarity_result_str = None

    if request.method == "POST":
        if 'pdf' not in request.files:
            findings = ["Error: No file uploaded. Please select a PDF file."]
        else:
            pdf = request.files["pdf"]
            if pdf.filename == '':
                findings = ["Error: No file selected. Please choose a PDF file."]
            elif pdf and pdf.filename.endswith(".pdf"):
                pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf.filename)
                pdf.save(pdf_path)
                abs_pdf_path = os.path.abspath(pdf_path)

                try:
                    namespace = run_analysis(abs_pdf_path)

                    df = namespace.get("df_results")
                    first = namespace.get("first_order_findings")
                    full_json = namespace.get("result_json")
                    openai_insights = namespace.get("OpenAI_insights")
                    similarity_result = namespace.get("similarity_result") 

                    if df is not None:
                        df_html = df.to_html(classes="table", index=False)
                    if isinstance(first, dict):
                        findings = [f"{k}: {v}" for k, v in first.items()]
                    if isinstance(full_json, dict):
                        result_json_str = json.dumps(full_json, indent=2)
                    if openai_insights is not None:  
                        openai_insights_str = json.dumps(openai_insights, indent=2)
                    if similarity_result is not None:
                        similarity_result_str = str(similarity_result)
                except Exception as e:
                    print(f"Error processing PDF: {e}")
                    findings = [f"Error processing PDF: {str(e)}"]
            else:
                findings = ["Error: Please upload a valid PDF file only."]

    return render_template(
        "index.html",
        table=df_html,
        findings=findings,
        result_json=result_json_str,
        openai_insights=openai_insights_str,
        similarity_result=similarity_result_str
    )

if __name__ == "__main__":
    app.run(debug=True, port=5004)
