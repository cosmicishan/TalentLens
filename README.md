# AI Resume Analyzer – Hack Nation 2025 (MIT Sloan AI Club)

<img width="1536" height="1024" alt="ChatGPT Image Aug 10, 2025, 03_33_43 PM" src="https://github.com/user-attachments/assets/90d14479-7304-4187-9b8b-13bcf102b0c8" />


## 📌 Overview
This project was built for **Hack Nation Hackathon 2025**, organized by the **MIT Sloan AI Club**.  
It is an **AI-powered resume analysis and insights platform** that enables recruiters to post jobs, candidates to upload resumes, and the system to automatically evaluate candidate-job fit using advanced AI techniques.

The platform assigns **matching scores**, generates **data-driven insights**, and visualizes results through **interactive graphs**, helping recruiters identify top talent quickly.

---

## 🚀 Features
- **Job Posting** – Recruiters can post new job applications.
- **Resume Upload** – Candidates can apply by uploading their resume (PDF).
- **AI Matching** – Uses embeddings to calculate similarity scores between resumes and job descriptions.
- **Visual Insights** – Generates graphs showing candidate ranking, skill coverage, and matching percentage.
- **Data-Driven Decisions** – Helps recruiters shortlist top candidates efficiently.

---

## 📂 Project Structure
```
.
├── flask_app/                  # Entire Flask web application
├── uploads/resumes/            # Uploaded candidate resumes
├── Random_Resume/              # LLM-generated random resumes for testing
├── candidate_plots/            # Plots generated from random resume test data
├── gemini_resume_matching.py   # Testing similarity score calculation using embeddings
├── resume_match_results.csv    # CSV containing top-matched resumes for sample jobs
├── resume_report.py            # Script testing job description against all random resumes
├── test.ipynb                   # Jupyter Notebook for REPL-based experiments
├── ml_job_description.pdf      # Sample job description used for testing
├── README.md                   # Project documentation (this file)
```

---

## 🧪 Testing & Experiments
- **Random_Resume/** → Collection of fake resumes generated using an LLM for experimentation.
- **candidate_plots/** → Visualization outputs (bar graphs, scatter plots, etc.) from similarity score analysis.
- **gemini_resume_matching.py** → Script that computes AI-based matching scores using text embeddings.
- **resume_match_results.csv** → Stores the top N candidate matches for a given job.
- **resume_report.py** → Runs similarity scoring against all test resumes for a sample job description.
- **test.ipynb** → Experimental notebook for running quick REPL-based prototype tests.

---

## 🛠 Tech Stack
- **Backend:** Flask (Python)
- **AI/ML:** Google Gemini API for embeddings
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Frontend:** HTML, CSS, JavaScript (via Flask templates)
- **Storage:** Local file system (uploads)

---

## 📊 How It Works
1. **Recruiter posts a job description** via the Flask web UI.
2. **Candidates upload resumes** (PDF format).
3. The system **extracts text** from resumes and the job description.
4. AI **generates embeddings** for both using Gemini.
5. A **similarity score** is calculated for each resume.
6. **Top candidates** are displayed with matching scores.
7. **Graphs & insights** are shown to visualize the talent pool.

---

## 🏆 Hackathon Context
This project was designed and built for **Hack Nation 2025** under **MIT Sloan AI Club**.  
The goal was to create an AI-driven recruitment tool that **improves hiring efficiency** by automating resume screening and providing actionable insights.

---

## 📸  Demo


---

## 📌 Future Improvements
- Integrate real-time **ATS (Applicant Tracking System)** integration.
- Support **multi-language resumes**.
- Add **cover letter analysis**.
- Deploy to cloud for **public access**.

---

## 👤 Author
**Ishan Purohit** – AI Developer & Hackathon Participant  
