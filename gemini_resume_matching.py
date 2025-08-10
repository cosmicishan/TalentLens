import os
import google.generativeai as genai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import Dict, List, Tuple

# Configure the Gemini API
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Define the models to be used
GENERATION_MODEL = "gemini-1.5-pro-latest"
EMBEDDING_MODEL = "models/text-embedding-004"


def get_embedding(text: str) -> List[float]:
    """Get embedding for a given text using Gemini's embedding model."""
    try:
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="SEMANTIC_SIMILARITY",
        )
        return response["embedding"]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts using embeddings."""
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    if not embedding1 or not embedding2:
        return 0.0
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity


def analyze_resume_details(text: str, job_desc: str) -> Dict:
    """Analyze resume text and provide actionable feedback using Gemini's generation model."""
    try:
        model = genai.GenerativeModel(GENERATION_MODEL)
        prompt = f"""
        Please analyze the following resume text and provide insights in the following categories:
        - Skills
        - Experience
        - Education
        - Domain expertise
        - Certifications
        
        Additionally, provide actionable feedback on how the candidate can improve their resume to better match the following job description:
        
        Job Description: {job_desc}
        
        Resume Text: {text}
        
        Provide the analysis in a valid JSON format with these exact keys: skills, experience, education, domain, certifications, feedback.
        """
        response = model.generate_content(prompt)
        content = response.text.strip()

        # Extract JSON from the response
        try:
            # Handle potential markdown formatting
            if content.startswith("```json"):
                content = content[7:-3].strip()
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            print(f"Content that failed to decode: {content}")
            return {
                "skills": "",
                "experience": "",
                "education": "",
                "domain": "",
                "certifications": "",
                "feedback": "Unable to generate feedback due to an invalid JSON response.",
            }
    except Exception as e:
        print(f"Error in resume analysis: {e}")
        return {
            "skills": "",
            "experience": "",
            "education": "",
            "domain": "",
            "certifications": "",
            "feedback": "Unable to generate feedback due to an error.",
        }


def calculate_match_score(
    resume_text: str, job_desc: str
) -> Tuple[Dict[str, float], float, Dict]:
    """Calculate detailed match scores between resume and job description."""
    weights = {
        "skills": 0.35,
        "experience": 0.25,
        "education": 0.15,
        "domain": 0.15,
        "certifications": 0.10,
    }

    try:
        resume_analysis = analyze_resume_details(resume_text, job_desc)
        job_analysis = analyze_resume_details(job_desc, job_desc)

        scores = {}
        for category, weight in weights.items():
            similarity = calculate_semantic_similarity(
                str(resume_analysis.get(category, "")),
                str(job_analysis.get(category, "")),
            )
            scores[category] = similarity * weight

        total_score = sum(scores.values())
        feedback = resume_analysis.get("feedback", "No feedback available.")
        return scores, total_score, feedback
    except Exception as e:
        print(f"Error in match calculation: {e}")
        return (
            {category: 0.0 for category in weights.keys()},
            0.0,
            "Unable to calculate match score due to an error.",
        )


def main():
    """Main function to run the resume matcher."""
    # This example does not include the UI components from the original code.
    # It focuses on the core logic of using Gemini for matching.
    
    # Example usage:
    job_description = """
    We are looking for a Python Developer with 5+ years of experience. The ideal candidate should have strong skills in Django or Flask, REST APIs, and database management (SQL, NoSQL). Experience with cloud platforms like AWS and CI/CD pipelines is a plus. A degree in Computer Science is required.
    """

    resumes = [
        ("Resume_A.txt", """
        John Doe
        Summary: A software engineer with 6 years of experience primarily in Python.
        Experience: Developed web applications using Flask, integrated RESTful APIs, and managed MySQL databases. Worked with Docker and Jenkins for CI/CD.
        Education: B.S. in Computer Science from a top university.
        Skills: Python, Flask, REST APIs, MySQL, Docker, Jenkins, JavaScript.
        Certifications: None.
        """),
        ("Resume_B.txt", """
        Jane Smith
        Summary: An experienced web developer specializing in frontend technologies.
        Experience: Built responsive user interfaces with React and Vue.js. Familiar with Node.js for backend services.
        Education: M.S. in Information Technology.
        Skills: JavaScript, React, Vue.js, Node.js, HTML, CSS.
        Certifications: AWS Certified Developer.
        """),
    ]

    results = []
    
    print("Starting resume analysis...")
    for file_name, resume_text in resumes:
        print(f"\nAnalyzing {file_name}...")
        scores, overall_score, feedback = calculate_match_score(
            resume_text, job_description
        )
        
        result = {
            "Resume Name": file_name,
            "Overall Match": f"{overall_score * 100:.2f}%",
            **{f"{k.title()} Match": f"{v * 100:.2f}%" for k, v in scores.items()},
            "Feedback": feedback,
        }
        results.append(result)
    
    if results:
        results_df = pd.DataFrame(results).sort_values(
            "Overall Match", ascending=False
        )
        print("\n--- Match Results ---")
        print(results_df)

        print("\n--- Detailed Feedback ---")
        for result in results:
            print(f"\nFeedback for {result['Resume Name']}:")
            print(result['Feedback'])
    else:
        print("No valid results were generated.")


if __name__ == "__main__":
    main()