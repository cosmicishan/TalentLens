import os
import json
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq
from datetime import datetime
import re

# --------------------------
# PDF Extraction
# --------------------------
def extract_text_from_pdf(file_path):
    """Extract text from a PDF file using pdfplumber."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

# --------------------------
# Helper: extract JSON substring from model output robustly
# --------------------------
def extract_json_from_text(text):
    """Find the first {...} or [...] JSON substring and parse it."""
    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Use regex to find JSON object/array
    # Find first { and matching } (approx) — safer to find longest {...}
    obj_matches = re.findall(r"\{.*\}", text, flags=re.S)
    arr_matches = re.findall(r"\[.*\]", text, flags=re.S)
    candidates = arr_matches + obj_matches
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue
    # As a fallback return None
    return None

# --------------------------
# Configurations
# --------------------------
JOB_DESCRIPTION_PATH = "ml_job_description.pdf"  # Path to job description PDF
RESUMES_FOLDER = "Random_Resume"  # Folder containing resume PDFs
OUTPUT_CSV = "resume_match_results.csv"
OUTPUT_PLOTS_DIR = "candidate_plots"
USE_REACT_LOOP = True   # Set False to do single-pass evaluation

os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

# --------------------------
# Step 1 – Extract Job Description
# --------------------------
job_description = extract_text_from_pdf(JOB_DESCRIPTION_PATH)

# --------------------------
# Step 2 – Extract All Resumes from Folder
# --------------------------
resumes = []
for file_name in os.listdir(RESUMES_FOLDER):
    if file_name.lower().endswith(".pdf"):
        candidate_name = os.path.splitext(file_name)[0]
        file_path = os.path.join(RESUMES_FOLDER, file_name)
        resume_text = extract_text_from_pdf(file_path)
        resumes.append({"name": candidate_name, "content": resume_text, "file": file_path})

# --------------------------
# Step 3 – Enhanced System Prompt (rich metadata + score breakdown)
# --------------------------
system_prompt = r"""
You are an expert technical recruiter and AI resume-matching engine. You will be given
one JOB DESCRIPTION and one RESUME. Your task is to evaluate the resume for the job
and return a strict JSON object (no extra commentary).

REQUIREMENTS:
1) Produce the following top-level JSON object (fields and types MUST match):
{
  "name": "<Candidate Name (string)>",
  "score": <int 0-100>,
  "score_breakdown": {
     "skills_score": <int 0-100>,
     "experience_score": <int 0-100>,
     "education_score": <int 0-100>,
     "certifications_score": <int 0-100>,
     "infra_deployment_score": <int 0-100>,
     "culture_fit_score": <int 0-100>,
     "weights": {
         "skills_weight": <float 0-1>,
         "experience_weight": <float 0-1>,
         "education_weight": <float 0-1>,
         "certifications_weight": <float 0-1>,
         "infra_deployment_weight": <float 0-1>,
         "culture_fit_weight": <float 0-1>
     }
  },
  "matching_analysis": "<Detailed strengths and gaps (string)>",
  "description": "<2-3 sentence summary of fit (string)>",
  "recommendation": "<Actionable improvements (string)>",
  "metadata": {
      "years_experience": <int or null>,
      "seniority": "<junior/mid/senior/lead/unknown>",
      "education_level": "<PhD/MS/BS/Other/Unknown>",
      "certifications": ["list", "or", "empty"],
      "matched_skills": ["skill1","skill2",...],
      "missing_skills": ["skillA","skillB",...],
      "domain_experience": ["finance","healthcare",...],
      "cloud_experience": ["AWS","GCP","Azure",...],
      "ml_infra_experience": ["docker","k8s","tf-serving","sagemaker",...],
      "vector_db_experience": ["ChromaDB","Weaviate","Milvus", "None"],
      "deployment_pipeline_experience": ["CI/CD","airflow","dataflow",...],
      "leadership_years": <int or 0>,
      "publications_or_patents": ["list or empty"],
      "languages": ["Python","Go",...],
      "location_or_remote": "<City/Region or Remote/Hybrid/Unknown>",
      "interview_readiness": "<Ready/NeedsPrep/SignificantGaps>"
  },
  "analysis_steps": ["short", "bullet", "summary", "of", "how", "you", "arrived"],
  "timestamp": "<ISO8601 timestamp string>"
}

2) The "score" must equal the weighted sum of the component scores according to the
   "weights" specified. Weights should sum approximately to 1.0 (you can use two decimals).

3) Keep "analysis_steps" short (3–6 items), factual, not chain-of-thought style; do NOT reveal private hidden chain-of-thought. These are concise observable steps.

4) If a field is unknown, use null (for numbers) or "Unknown"/empty-list for strings/lists.

5) Be conservative — do not inflate. Scores:
   - >85 excellent, 70–85 good, 50–70 average, <50 weak.

6) Output ONLY the JSON object (no backticks, no surrounding text). Ensure valid JSON.

7) IMPORTANT: If you cannot find the information (e.g., years of experience), estimate conservatively and explain in "matching_analysis".
"""

# --------------------------
# Helper: single-pass evaluate
# --------------------------
client = Groq()

def evaluate_resume_single(resume, job_text):
    user_prompt = f"""
JOB DESCRIPTION:
{job_text}

RESUME for {resume['name']}:
{resume['content']}

Return ONLY the JSON object as specified in system instructions, no extra commentary.
"""
    resp = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.15,
        max_tokens=1600
    )
    out = resp.choices[0].message.content
    parsed = extract_json_from_text(out)
    return parsed, out

# --------------------------
# Helper: ReAct-style 2-step evaluate (observe -> act)
# --------------------------
def evaluate_resume_react(resume, job_text):
    # Step 1: observation (short structured)
    obs_system = "You are an assistant that OBSERVES the resume relative to the job description. Output a short JSON with 'observations' list and 'critical_missing' list."
    obs_prompt = f"""
JOB DESCRIPTION:
{job_text}

RESUME for {resume['name']}:
{resume['content']}

Return JSON: {{ "observations": ["obs1","obs2",...], "critical_missing": ["skillA", ...] }}
Keep it short (max 6 observations). No commentary.
"""
    obs_resp = client.chat.completions.create(
        messages=[
            {"role": "system", "content": obs_system},
            {"role": "user", "content": obs_prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        max_tokens=400
    )
    obs_text = obs_resp.choices[0].message.content
    obs_json = extract_json_from_text(obs_text) or {"observations": [], "critical_missing": []}

    # Step 2: final act — give full evaluation, but include the observations as extra context
    user_prompt = f"""
JOB DESCRIPTION:
{job_text}

RESUME for {resume['name']}:
{resume['content']}

PREVIOUS_OBSERVATIONS:
{json.dumps(obs_json)}

Now produce the FINAL JSON object exactly in the format required by the system prompt (score, score_breakdown, metadata, matching_analysis, etc).
Return ONLY the JSON object.
"""
    final_resp = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.15,
        max_tokens=1600
    )
    final_text = final_resp.choices[0].message.content
    parsed = extract_json_from_text(final_text)
    return obs_json, parsed, final_text

# --------------------------
# Step 4 – Evaluate All Resumes (choose REACT or SINGLE)
# --------------------------
results = []
for resume in resumes:
    try:
        if USE_REACT_LOOP:
            obs, parsed, raw = evaluate_resume_react(resume, job_description)
        else:
            parsed, raw = evaluate_resume_single(resume, job_description)
            obs = None

        if parsed is None:
            print(f"⚠️ Could not parse JSON for {resume['name']}. Raw output saved to logs.")
            results.append({
                "name": resume["name"],
                "file": resume["file"],
                "raw_output": raw,
                "parsed": None
            })
            continue

        # Attach file path for reference
        parsed["file"] = resume["file"]
        results.append(parsed)

    except Exception as e:
        print(f"Error evaluating {resume['name']}: {e}")
        continue

# --------------------------
# Step 5 – Save & Sort Results
# --------------------------
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

# If JSON fields are nested, normalize some into columns for sorting
def safe_get(d, *keys, default=None):
    try:
        for k in keys:
            d = d[k]
        return d
    except Exception:
        return default

df['score_int'] = df.apply(lambda r: safe_get(r.to_dict(), 'score', default=0) if isinstance(r.to_dict().get('score'), int) else (int(r.to_dict().get('score')) if r.to_dict().get('score') else 0), axis=1)
top5 = df.sort_values("score_int", ascending=False).head(5)

# --------------------------
# Step 6 – Top 5 combined bar (saved)
# --------------------------
plt.figure(figsize=(10, 6))
plt.bar(top5["name"], top5["score_int"], color='skyblue')
plt.title("Top 5 Resume Matches")
plt.ylabel("Match Score")
plt.ylim(0, 100)
for i, score in enumerate(top5["score_int"]):
    plt.text(i, score + 1, str(score), ha='center', fontsize=10)
top5_plot_path = os.path.join(OUTPUT_PLOTS_DIR, f"top5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
plt.savefig(top5_plot_path, bbox_inches='tight')
plt.close()
print(f"Saved top5 plot to {top5_plot_path}")

# --------------------------
# Step 7 – Individual Candidate Plots for Top 5 (saved)
# --------------------------
for idx, row in top5.iterrows():
    try:
        name = row['name']
        data = row.to_dict()
        metadata = data.get('metadata', {}) or {}
        sb = data.get('score_breakdown', {}) or {}
        if isinstance(sb, dict):
            comp_scores = [
                sb.get('skills_score', 0),
                sb.get('experience_score', 0),
                sb.get('education_score', 0),
                sb.get('certifications_score', 0),
                sb.get('infra_deployment_score', 0),
                sb.get('culture_fit_score', 0)
            ]
            comp_labels = ["Skills", "Experience", "Education", "Certs", "Infra/Deploy", "Culture"]
        else:
            comp_scores = [0]*6
            comp_labels = ["Skills", "Experience", "Education", "Certs", "Infra/Deploy", "Culture"]

        matched_skills = metadata.get('matched_skills', []) or []
        missing_skills = metadata.get('missing_skills', []) or []

        # Create a 2x2 grid of plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"{name} — Score: {data.get('score', 'N/A')}", fontsize=16)

        # Top-left: Pie chart matched vs missing
        axes[0,0].pie([len(matched_skills), len(missing_skills)],
                      labels=["Matched", "Missing"], autopct='%1.1f%%')
        axes[0,0].set_title("Skill Match")

        # Top-right: Component score bar
        axes[0,1].bar(comp_labels, comp_scores)
        axes[0,1].set_ylim(0, 100)
        axes[0,1].set_title("Component Scores")

        # Bottom-left: Small metadata table (text)
        meta_text = ""
        meta_keys = [
            ("Years Exp", metadata.get("years_experience")),
            ("Seniority", metadata.get("seniority")),
            ("Education", metadata.get("education_level")),
            ("Certifications", ", ".join(metadata.get("certifications", [])) if metadata.get("certifications") else "None"),
            ("Cloud", ", ".join(metadata.get("cloud_experience", [])) if metadata.get("cloud_experience") else "None"),
            ("Vector DBs", ", ".join(metadata.get("vector_db_experience", [])) if metadata.get("vector_db_experience") else "None"),
            ("ML Infra", ", ".join(metadata.get("ml_infra_experience", [])) if metadata.get("ml_infra_experience") else "None"),
            ("Location", metadata.get("location_or_remote"))
        ]
        for k, v in meta_keys:
            meta_text += f"{k}: {v}\n"
        axes[1,0].text(0, 0.5, meta_text, va='center', fontsize=10)
        axes[1,0].axis('off')
        axes[1,0].set_title("Metadata Summary")

        # Bottom-right: Top matched skills (bars)
        top_matched = matched_skills[:8]
        axes[1,1].barh(top_matched[::-1], [1]*len(top_matched[::-1]))
        axes[1,1].set_title("Top Matched Skills (presence)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = f"{name.replace(' ','_')}_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        outpath = os.path.join(OUTPUT_PLOTS_DIR, fname)
        plt.savefig(outpath, bbox_inches='tight')
        plt.close()
        print(f"Saved candidate plot for {name} -> {outpath}")
    except Exception as e:
        print(f"Error plotting for {row.get('name')}: {e}")
        continue

print("Done — results saved to CSV and plots folder.")