import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage
from tqdm import tqdm

API_KEY = "key"
API_BASE = "https://openrouter.ai/api/v1"
MODEL_NAME = "deepseek/deepseek-r1:free"

CLAUSE_CSV_PATH = "../data/PythonExport.csv"
ISSUES_CSV_PATH = "../data/Clause_Review_Database_Compliant.csv"
OUTPUT_CSV_PATH = "../data/validated_results_with_context.csv"


chat = ChatOpenAI(
    temperature=0.0,
    openai_api_key=API_KEY,
    openai_api_base=API_BASE
)


def validate_clause(clause_text, analysis_result, short_recommendation):
    """Validates whether the analysis is correct according to the short recommendation."""
    prompt = f"""
You are a legal validation agent.

Given the following:
- A clause from a rental agreement
- The analysis result provided for it
- The official recommendation about how this issue should be handled

Rate your confidence (from 0.0 to 1.0) that the analysis result is appropriate based on the recommendation.

Clause:
\"\"\"{clause_text}\"\"\"

Analysis result:
\"\"\"{analysis_result}\"\"\"

Official recommendation for this issue:
\"\"\"{short_recommendation}\"\"\"

Respond ONLY with a float number between 0.0 and 1.0.
"""

    try:
        response = chat(
            [HumanMessage(content=prompt)],
            model=MODEL_NAME
        )
        confidence = float(response.content.strip())
        confidence = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
        return confidence

    except Exception as e:
        print("Validation error:", e)
        return None

def main():
    clauses_df = pd.read_csv(CLAUSE_CSV_PATH)

    issues_df = pd.read_csv(ISSUES_CSV_PATH)

    issue_mapping = {}
    for _, row in issues_df.iterrows():
        issue_key = row['Issue Name'].lower().replace(" ", "").replace("-", "").replace("_", "")
        issue_mapping[issue_key] = {
            "ShortRecommendation": row['Short Recommendation'],
            "CourtDecisionReference": row['Court Decision Reference'],
            "DecisionSentence": row['Exact Sentence from Decision']
        }

    result_rows = []

    print("Validating clauses with issue recommendations...")

    for _, row in tqdm(clauses_df.iterrows(), total=len(clauses_df)):
        for col in clauses_df.columns:
            if col.endswith("_Ana"):
                issue_key_raw = col.replace("_Ana", "")
                issue_key = issue_key_raw.lower().replace(" ", "").replace("-", "").replace("_", "")

                issue_info = issue_mapping.get(issue_key, {
                    "ShortRecommendation": "No recommendation available.",
                    "CourtDecisionReference": "Not available.",
                    "DecisionSentence": "Not available."
                })

                clause_text = str(row[col]).strip()
                result_col = f"{issue_key_raw}_Result"
                analysis_result = str(row[result_col]).strip() if result_col in row else "Not provided."

                # Skip empty clauses
                if clause_text and clause_text.lower() != "nan":
                    confidence = validate_clause(clause_text, analysis_result, issue_info["ShortRecommendation"])

                    result_rows.append({
                        "Issue": issue_key_raw,
                        "Bryter Analysis": clause_text,
                        "AnalysisResult": analysis_result,
                        "ShortRecommendation": issue_info["ShortRecommendation"],
                        "CourtDecisionReference": issue_info["CourtDecisionReference"],
                        "DecisionSentence": issue_info["DecisionSentence"],
                        "LLM_Confidence": confidence
                    })

    results_df = pd.DataFrame(result_rows)

    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nValidation completed. Results saved to {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()
