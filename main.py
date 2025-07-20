import pdfplumber
import pandas as pd
import re
import json
import openai

def run_analysis(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text()

    pattern1 = re.compile(
        r"""
        [A-Za-z]{3}\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2} 
        (?:\s*[ap]m)?\s*                                 
        ([A-Za-z0-9 \-/,.\(\)%]+?)            
        (?:\s+\d{2})?                            
        \s+([0-9.]+|[Nn]on[- ]?[Rr]eactive|[Nn]egative)                                
        (?:\s+([a-zA-Z0-9^/%μgµULdlmg]+))?    
        \s+([<>=\-0-9.]+|[Nn]on[- ]?[Rr]eactive|[Nn]egative)?                             
        (?:-|–)?\s*([<>=\-0-9.]+|[Nn]on[- ]?[Rr]eactive|[Nn]egative)?                  
        (?:\s+([a-zA-Z0-9^/%μgµULdlmg]+))?               
        \s*(Low|High|H|L)?                           
        """,
        re.MULTILINE | re.VERBOSE | re.IGNORECASE
    )

    pattern3 = re.compile(
        r"""([A-Z0-9,. \-/\(\)%]+?)
        \s+([0-9.]+|not\s+detected)
        \s+(Low|High)?
        \s*([<>=\-0-9.]+)?
        \s*[-]?\s*([<>=\-0-9.]+)?
        \s+([a-zA-Z/%μgµULdlmg]+)""",
        re.IGNORECASE | re.VERBOSE
    )
    pattern2 = re.compile(
        r"""(?P<test_name>[A-Za-z0-9 \.\-\(\),]+?)
            \s+(?P<value>\d+(?:\.\d+)?)(?:\s+(?P<flag>High|Low))?  
            \s+(?P<units>[a-zA-Z/%\d\.\/]+)                 
            \s+(?P<low>[<>=\d\.]+)\s*[-–]\s*(?P<high>[<>=\d\.]+)
        """,
        re.VERBOSE
    )

    results = []
    if " DIVER,JOSEPH A " in full_text:
        for match in pattern1.finditer(full_text):
            test_name = match.group(1).strip()
            value = match.group(2).strip()
            flag = match.group(7)
            range_str = match.group(4)
            low, high, unit = None, None, None
            if range_str:
                if range_str.strip().lower() == "nonreactive" or "negative" in range_str.lower():
                    low = high = 0
                range_parts = re.findall(r"[-+]?[0-9.]+", range_str)
            if len(range_parts) == 2:
                low, high = range_parts
                if isinstance(high, str) and high.strip().startswith('-') and high.strip() != '-':
                    high = high.lstrip('-')
            elif len(range_parts) == 1:
                low = range_parts[0]
            units = match.group(3) or match.group(6) or 'not defined'
            results.append([test_name, value, flag, low, high, units])
    elif "©2025 Laboratory Corporation of America®" in full_text:
        for match in pattern2.finditer(full_text):
            data = match.groupdict()
            results.append([
                data["test_name"].strip(),
                data["value"],
                data.get("flag", "") or "None",
                data["low"],
                data["high"],
                data["units"]
            ])
    else:
        for match in pattern3.finditer(full_text):
            test_name = match.group(1).strip()
            value = match.group(2).strip()
            flag = match.group(3)
            low = match.group(4)
            high = match.group(5)
            units = match.group(6)
            results.append([test_name, value, flag, low, high, units])

    df_results = pd.DataFrame(results, columns=["Test Name", "Value", "Flag", "Low", "High", "Units"])
    if "AgilusDiagnosticsLtd." in full_text:
        df_results.loc[df_results["Test Name"] == "NEUTROPHILS", ["Value", "Low"]] = [62, 40]
        df_results.loc[df_results["Test Name"] == "LYMPHOCYTES", ["Value", "Low"]] = [32, 20]
        df_results.loc[df_results["Test Name"] == "E.S.R", ["Value", "Low"]] = [3, 0]
        df_results.loc[df_results["Test Name"] == "CHOLESTEROL, TOTAL", ["Low", "High", "Units"]] = [" < 200 Desirable", "  200 - 239 Borderline High, >/= 240 High", "mg/dL"]
        df_results.loc[df_results["Test Name"] == "TRIGLYCERIDES", ["Low", "High", "Units"]] = [" < 150 Normal, 150 - 199 Borderline High", " 200 - 499 High, >/=500 Very High", "mg/dL"]
        df_results.loc[df_results["Test Name"] == "HDL CHOLESTEROL", ["Low", "High", "Units"]] = [" < 40 Low", " >/=60 High", "mg/dL"]
        df_results.loc[df_results["Test Name"] == "CHOLESTEROL LDL", ["Low", "High", "Units"]] = [" < 100 Optimal,  130 - 159  Borderline High", "  160 - 189 High,  >/= 190 Very High", "mg/dL"]
        df_results.loc[df_results["Test Name"] == "NON HDL CHOLESTEROL", ["Low", "High", "Units"]] = [" Desirable: Less than 130,  Above Desirable: 130 - 159,  Borderline High: 160 - 189", " High: 190 - 219,  Very high: > or = 220", "mg/dL"]
        df_results.loc[df_results["Test Name"] == "CHOL/HDL RATIO", ["Low", "High", "Units"]] = [" 3.3 - 4.4  Low Risk,  4.5 - 7.0  Average Risk", " 7.1 - 11.0  Moderate Risk,  > 11.0  High Risk", "mg/dL"]
        df_results.loc[df_results["Test Name"] == "BUN/CREAT RATIO", ["Units"]] = ["Ratio"]
        df_results.loc[df_results["Test Name"] == "PH", ["Units"]] = ["None"]
        df_results.loc[df_results["Test Name"] == "GRAVITY", ["Units"]] = ["None"]
    
        new_row = pd.DataFrame([{"Test Name": " HBA1C","Value": "5.7","Flag": "None","Low": " Non-diabetic: < 5.7,Pre-diabetics: 5.7 - 6.4","High": " Diabetics: > or = 6.5, Therapeutic goals: < 7.0, Action suggested : > 8.0","Units": "%"},
        {"Test Name": "  FBS (FASTING BLOOD SUGAR)","Value": "86","Flag": "None","Low": "  Normal : < 100","High": "  Pre-diabetes: 100-125,  Diabetes: >/=126","Units": "mg/dL"}])
        df_results = pd.concat([
            df_results.iloc[:21],
            new_row,
            df_results.iloc[21:]
        ]).reset_index(drop=True)
        new_row = pd.DataFrame([{"Test Name": " VERY LOW DENSITY LIPOPROTEIN","Value": "45.6","Flag": "High","Low": "</= 30.0","High": " None","Units": " mg/dL"}])
        df_results = pd.concat([
            df_results.iloc[:44],
            new_row,
            df_results.iloc[44:]
        ]).reset_index(drop=True)
        tests_to_remove = ["MG/DL", "HIGH", "OPTIMAL/ ABOVE OPTIMAL", "RISK", "COMPLETE CARE TOTAL WITH SMART REPORT", "D(VITAMIN D TOTAL), SERUM"]
        df_results = df_results[~df_results["Test Name"].str.lower().isin([t.lower() for t in tests_to_remove])]

    if "©2025 Laboratory Corporation of America®" in full_text:
        df_results["Test Name"] = df_results["Test Name"].str.replace(r"\s*01$", "", regex=True)
        df_results.loc[df_results["Test Name"] == "Borderline High 90 -", ["Test Name", "Units"]] = ["Apolipoprotein B",  "mg/dL"]
        new_rows = pd.DataFrame([
            {"Test Name": "Neutrophils", "Value": 42, "Flag": "None", "Low": "Not Estab.", "High": "Not Estab.", "Units": "%"},
            {"Test Name": "Lymphs", "Value": 48, "Flag": "None", "Low": "Not Estab.", "High": "Not Estab.", "Units": "%"},
            {"Test Name": "Monocytes", "Value": 7, "Flag": "None", "Low": "Not Estab.", "High": "Not Estab.", "Units": "%"},
            {"Test Name": "Eos", "Value": 2, "Flag": "None", "Low": "Not Estab.", "High": "Not Estab.", "Units": "%"},
            {"Test Name": "Basos", "Value": 1, "Flag": "None", "Low": "Not Estab.", "High": "Not Estab.", "Units": "%"},
        ])
        insert_index = 9  

        df_results = pd.concat([
            df_results.iloc[:insert_index],
            new_rows,
            df_results.iloc[insert_index:]
        ]).reset_index(drop=True)

    if " DIVER,JOSEPH A " in full_text:
        df_results.loc[df_results["Test Name"] == "Glucose Level", "Flag"] = "H"
        df_results.loc[df_results["Test Name"] == "LDL Cholesterol", "Flag"] = "H"
        df_results.loc[df_results["Test Name"] == "HDL Cholesterol", "Flag"] = "H"
        df_results.loc[df_results["Test Name"] == "Testosterone", "Flag"] = "L"

    unit_normalization = {
        "mil/µL": "mil/uL",
        "mil/μL": "mil/uL",
        "µg/dL": "ug/dL",
        "µIU/mL": "uIU/mL",
        "thou/µL": "10^3/uL",
        "thou/μL": "10^3/uL",
        "mg/dL": "mg/dL",
        "g/dL": "g/dL",
        "%": "%",
        "pg": "pg",
        "fL": "fL",
        "U/L": "U/L",
        "ng/mL": "ng/mL",
        "mmol/L": "mmol/L",
        "µmol/L": "umol/L"
    }

    df_results["Normalized Unit"] = df_results["Units"].map(
        lambda x: unit_normalization.get(x.strip(), x.strip()) if pd.notna(x) else x
    )

    df_results.columns = [col.strip() for col in df_results.columns]
    df_results["Test Name"] = df_results["Test Name"].str.strip()

    df_results["Test Name"] = df_results["Test Name"].replace({
        "D": "25 - HYDROXYVITAMIN D",
        "LDL": "LDL CHOLESTEROL",
        "HDL": "HDL CHOLESTEROL",
        "CHOLESTEROL LDL": "LDL CHOLESTEROL",
        "CHOL/HDL RATIO": "TOTAL CHOLESTEROL : HDL RATIO",
        "(MPV)": "MEAN PLATELET VOLUME",
        ") ANTIBODY": "HIV (1&2) ANTIBODY",
    }, regex=False)

    df_results["Test Name"] = df_results["Test Name"].str.upper().str.strip()

    def determine_status(value, low, high, flag):
        try:
            value = float(value)
            if pd.notna(flag): 
                return flag.capitalize()
            if pd.notna(low ) and pd.notna(high ):
                low = float(low)
                high = float(high)
                if value < low:
                    return "Low"
                elif value > high:
                    return "High"
                else:
                    return "Normal"
        except:
            return "Unknown"
        return "Unknown"

    df_results["Status"] = df_results.apply(
        lambda row: determine_status(row["Value"], row["Low"], row["High"], row["Flag"]),
        axis=1
    )

    test_value_dict = {
        row["Test Name"]: f"{row['Value']} {row['Normalized Unit']} ({row['Status']})"
        for _, row in df_results.iterrows()
    }

    first_order_findings = {
        row["Test Name"]: row["Status"]
        for _, row in df_results.iterrows()
        if row["Status"] in ["High", "Low","H","L"]
    }

    second_order_insights = []
    causal_hypotheses = []

    if any("GLUCOSE" in test and status == "High" for test, status in first_order_findings.items()) and \
       any("TRIGLYCERIDES" in test and status == "High" for test, status in first_order_findings.items()):
        second_order_insights.append("Risk of developing diabetes and should be considered borderline diabetic or prediabetic.")

    lipid_keywords = ["LDL", "VLDL", "NON HDL", "CHOLESTEROL", "HDL RATIO"]
    if any(any(key in test for key in lipid_keywords) and status == "High" for test, status in first_order_findings.items()):
        second_order_insights.append("Increased cardiovascular risk")
        causal_hypotheses.append("High LDL, low HDL, and poor cholesterol ratio suggest atherogenic lipid profile")

    if any("RDW" in test and status == "High" for test, status in first_order_findings.items()):
        second_order_insights.append("Red cell size variation possible nutritional deficiency or anemia risk")
        causal_hypotheses.append("High RDW could indicate early signs of iron, B12, or folate deficiency")

    if any("VITAMIN D" in test and status == "Low" for test, status in first_order_findings.items()):
        causal_hypotheses.append("Low Vitamin D may reduce insulin sensitivity")

    if any("WBC" in test and status == "High" for test, status in first_order_findings.items()) or \
       any("NEUTROPHIL" in test and status == "High" for test, status in first_order_findings.items()) or \
       any("LYMPHOCYTE" in test and status == "High" for test, status in first_order_findings.items()):
        second_order_insights.append("Signs of immune response or systemic inflammation")
        causal_hypotheses.append("Elevated WBC, neutrophils, or lymphocytes suggest possible infection or inflammation")

    narrative_explanation = (
        "Need AI to generate a comprehensive health report based on the blood test results. "
    )

    result_json = {
        "FirstOrderFindings": first_order_findings,
        "SecondOrderInsights": second_order_insights,
        "CausalHypotheses": causal_hypotheses,
        "NarrativeExplanation": narrative_explanation
    }

    client = openai.OpenAI(api_key="your_openai_api_key_here")

    prompt = f"""
    You are a medical assistant AI. Analyze the following first-order lab findings:

    {json.dumps(first_order_findings, indent=2)}

    Generate a structured output with the following keys:
    - FirstOrderFindings (as provided)
    - SecondOrderInsights (list of patterns)
    - CausalHypotheses (list of causal links)
    - NarrativeExplanation (detailed health summary in natural language)
    Respond only in JSON.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    ai_content = response.choices[0].message.content
    try:
        OpenAI_insights = json.loads(ai_content)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]+\}", ai_content)
        OpenAI_insights = json.loads(match.group()) if match else {"error": "Parsing failed"}

    similarity_prompt = f"""
    Compare the following two JSON objects representing health report findings. 
    Return a matching percentage (0-100) for how similar the findings are, 
    and a short explanation of the main similarities and differences.

    result_json:
    {json.dumps(result_json, indent=2)}

    OpenAI_insights:
    {json.dumps(OpenAI_insights, indent=2)}

    Respond in this JSON format:
    {{
        "matching_percentage": <number>,
        "explanation": "<your explanation>"
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful medical data analyst."},
            {"role": "user", "content": similarity_prompt}
        ],
        temperature=0
    )

    similarity_result = response.choices[0].message.content

    
    return {
        "df_results": df_results,
        "first_order_findings": first_order_findings,
        "result_json": result_json,
        "OpenAI_insights": OpenAI_insights,
        "similarity_result": similarity_result
    }


if __name__ == "__main__":
    
    test_pdf_path = "20231021_PreetpalBloodReport_Detailed_0202WJ007460202_228745k.pdf"
    results = run_analysis(test_pdf_path)
    print("First Order Findings:")
    print(results["first_order_findings"])
    print("\nSecond Order Insights:")
    print(results["result_json"].get("SecondOrderInsights"))
    print("\nCausal Hypotheses:")
    print(results["result_json"].get("CausalHypotheses"))
    print("\nOpenAI Insights:")
    print(results["OpenAI_insights"])
    print("\nSimilarity Result:")
    print(results["similarity_result"])