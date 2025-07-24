# 🧪 Health Insights from Blood Report Flask App

This Flask application processes blood report PDFs and generates health insights using OpenAI GPT models. It automatically extracts, cleans, interprets, and analyzes medical lab reports from PDF files using rule-based logic and GPT-based reasoning.

---

## 🚀 What Does This Do?

Given a blood report PDF (e.g., from labs like Agilus, Thyrocare, SRL, etc.), this tool:

1. 🧾 **Extracts** test names, values, units, and reference ranges using OCR and regex.
2. 🧼 **Normalizes** data using `pandas` for reliable comparison.
3. ⚖️ **Flags abnormalities** by comparing against medical reference ranges.
4. 🧠 **Generates insights** using OpenAI GPT for deeper analysis.
5. 🔍 **Explains relationships** between different test results.
6. 🧾 **Exports** clean, structured output in JSON and web interface.

---

## 🛠️ Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the project root with your OpenAI API key:
```env
OPENAI_API_KEY=your_actual_api_key_here
```

### 3. Run the Application
```bash
python app.py
```

### 4. Access the Application
- Open your browser and go to `http://127.0.0.1:5004`
- Upload a PDF blood report to get health insights

---

## 🧱 Technologies Used

| Component     | Purpose                                           |
|--------------|---------------------------------------------------|
| `pdfplumber`  | Extracts structured/unstructured text from PDFs   |
| `pandas`      | Cleans and organizes lab test data                |
| `Flask`       | Web framework for upload and viewing interface    |
| `OpenAI API`  | GPT models for intelligent health insights        |

---

## 🔧 Troubleshooting

### ModuleNotFoundError Issues
If you encounter "ModuleNotFoundError: No module named 'openai'":
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Check that you're using the correct Python environment
3. Verify OpenAI package installation: `pip show openai`



### API Key Issues
- Verify your OpenAI API key is correctly set in the `.env` file
- Ensure the `.env` file is in the project root directory
- Check API key permissions and billing status on OpenAI dashboard

---

## 🔒 Security Notes

- API keys are stored in environment variables (`.env` file)
- The `.env` file is excluded from version control via `.gitignore`
- Never commit API keys to the repository
- Use environment variables for production deployments

---
- Remove noise, spacing, and inconsistencies.

### ✅ Step 4: Medical Flagging
- Each test result is compared against the reference range.
- Result is categorized as:
  - ✅ Normal
  - 🔺 High
  - 🔻 Low

### ✅ Step 5: Generate Insights
- 🔍 **First-order findings** – direct abnormalities.
- 🧠 **Second-order insights** – systemic patterns (e.g., metabolic syndrome).
- 🔄 **Causal hypotheses** – inferred relationships using GPT models.
- 🗣️ **Narrative summary** – readable health interpretation.

---

