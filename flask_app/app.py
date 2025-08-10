from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import json
import sqlite3
from datetime import datetime
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq
import re
from functools import wraps

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PLOTS_FOLDER'] = 'static/plots'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['PLOTS_FOLDER'], 
               os.path.join(app.config['UPLOAD_FOLDER'], 'resumes'),
               os.path.join(app.config['UPLOAD_FOLDER'], 'job_descriptions')]:
    os.makedirs(folder, exist_ok=True)

# Database setup
def init_db():
    conn = sqlite3.connect('recruitment.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,  -- 'recruiter' or 'candidate'
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Jobs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            recruiter_id INTEGER,
            pdf_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'active',
            FOREIGN KEY (recruiter_id) REFERENCES users (id)
        )
    ''')
    
    # Applications table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER,
            candidate_id INTEGER,
            resume_path TEXT,
            status TEXT DEFAULT 'pending',
            match_score INTEGER,
            analysis_data TEXT,  -- JSON string
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES jobs (id),
            FOREIGN KEY (candidate_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Resume analysis functions (from your original code)
def extract_text_from_pdf(file_path):
    """Extract text from a PDF file using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return ""

def extract_json_from_text(text):
    """Find the first {...} or [...] JSON substring and parse it."""
    try:
        return json.loads(text)
    except Exception:
        pass
    
    obj_matches = re.findall(r"\{.*\}", text, flags=re.S)
    arr_matches = re.findall(r"\[.*\]", text, flags=re.S)
    candidates = arr_matches + obj_matches
    
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None

# System prompt for resume analysis
SYSTEM_PROMPT = """
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
   "weights" specified. Weights should sum approximately to 1.0.

3) Keep "analysis_steps" short (3–6 items), factual.

4) If a field is unknown, use null (for numbers) or "Unknown"/empty-list for strings/lists.

5) Be conservative — do not inflate. Scores: >85 excellent, 70–85 good, 50–70 average, <50 weak.

6) Output ONLY the JSON object (no backticks, no surrounding text). Ensure valid JSON.
"""

def analyze_resume(job_text, resume_text, candidate_name):
    """Analyze resume against job description using Groq API."""
    try:
        client = Groq()  # Make sure GROQ_API_KEY is set in environment
        
        user_prompt = f"""
JOB DESCRIPTION:
{job_text}

RESUME for {candidate_name}:
{resume_text}

Return ONLY the JSON object as specified in system instructions, no extra commentary.
"""
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.15,
            max_tokens=1600
        )
        
        output = response.choices[0].message.content
        parsed = extract_json_from_text(output)
        return parsed
        
    except Exception as e:
        print(f"Error analyzing resume: {e}")
        return None

# Authentication decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def recruiter_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'recruiter':
            flash('Access denied. Recruiter access required.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def candidate_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'candidate':
            flash('Access denied. Candidate access required.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']
        
        if role not in ['recruiter', 'candidate']:
            flash('Invalid role selected', 'error')
            return redirect(url_for('register'))
        
        password_hash = generate_password_hash(password)
        
        conn = sqlite3.connect('recruitment.db')
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (name, email, password_hash, role)
                VALUES (?, ?, ?, ?)
            ''', (name, email, password_hash, role))
            conn.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already exists', 'error')
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = sqlite3.connect('recruitment.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, password_hash, role FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['name'] = user[1]
            session['role'] = user[3]
            
            if user[3] == 'recruiter':
                return redirect(url_for('recruiter_dashboard'))
            else:
                return redirect(url_for('candidate_dashboard'))
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# Recruiter routes
@app.route('/recruiter/dashboard')
@recruiter_required
def recruiter_dashboard():
    conn = sqlite3.connect('recruitment.db')
    cursor = conn.cursor()
    
    # Get recruiter's jobs
    cursor.execute('''
        SELECT j.*, COUNT(a.id) as application_count
        FROM jobs j
        LEFT JOIN applications a ON j.id = a.job_id
        WHERE j.recruiter_id = ?
        GROUP BY j.id
        ORDER BY j.created_at DESC
    ''', (session['user_id'],))
    jobs = cursor.fetchall()
    
    conn.close()
    return render_template('recruiter_dashboard.html', jobs=jobs)

@app.route('/recruiter/post-job', methods=['GET', 'POST'])
@recruiter_required
def post_job():
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        job_file = request.files.get('job_description_pdf')
        
        pdf_path = None
        if job_file and job_file.filename.endswith('.pdf'):
            filename = secure_filename(job_file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filename = timestamp + filename
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'job_descriptions', filename)
            job_file.save(pdf_path)
        
        conn = sqlite3.connect('recruitment.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO jobs (title, description, recruiter_id, pdf_path)
            VALUES (?, ?, ?, ?)
        ''', (title, description, session['user_id'], pdf_path))
        conn.commit()
        conn.close()
        
        flash('Job posted successfully!', 'success')
        return redirect(url_for('recruiter_dashboard'))
    
    return render_template('post_job.html')

@app.route('/recruiter/job/<int:job_id>/applications')
@recruiter_required
def job_applications(job_id):
    conn = sqlite3.connect('recruitment.db')
    cursor = conn.cursor()
    
    # Verify job belongs to current recruiter
    cursor.execute('SELECT * FROM jobs WHERE id = ? AND recruiter_id = ?', 
                  (job_id, session['user_id']))
    job = cursor.fetchone()
    
    if not job:
        flash('Job not found', 'error')
        return redirect(url_for('recruiter_dashboard'))
    
    # Get applications with candidate details
    cursor.execute('''
        SELECT a.*, u.name as candidate_name, u.email as candidate_email
        FROM applications a
        JOIN users u ON a.candidate_id = u.id
        WHERE a.job_id = ?
        ORDER BY a.match_score DESC, a.applied_at DESC
    ''', (job_id,))
    applications = cursor.fetchall()
    
    conn.close()
    return render_template('job_applications.html', job=job, applications=applications)

@app.route('/recruiter/analyze-applications/<int:job_id>')
@recruiter_required
def analyze_applications(job_id):
    """Analyze all applications for a job using the resume matching algorithm."""
    conn = sqlite3.connect('recruitment.db')
    cursor = conn.cursor()
    
    # Get job details
    cursor.execute('SELECT * FROM jobs WHERE id = ? AND recruiter_id = ?', 
                  (job_id, session['user_id']))
    job = cursor.fetchone()
    
    if not job:
        flash('Job not found', 'error')
        return redirect(url_for('recruiter_dashboard'))
    
    # Extract job description text
    job_text = job[2]  # description field
    if job[4]:  # pdf_path field
        pdf_text = extract_text_from_pdf(job[4])
        if pdf_text:
            job_text = pdf_text
    
    # Get applications that haven't been analyzed
    cursor.execute('''
        SELECT a.*, u.name as candidate_name
        FROM applications a
        JOIN users u ON a.candidate_id = u.id
        WHERE a.job_id = ? AND (a.analysis_data IS NULL OR a.match_score IS NULL)
    ''', (job_id,))
    applications = cursor.fetchall()
    
    analyzed_count = 0
    for app in applications:
        try:
            # Extract resume text
            resume_text = extract_text_from_pdf(app[3])  # resume_path
            if not resume_text:
                continue
            
            # Analyze resume
            analysis = analyze_resume(job_text, resume_text, app[8])  # candidate_name
            if analysis and 'score' in analysis:
                # Update application with analysis results
                cursor.execute('''
                    UPDATE applications 
                    SET match_score = ?, analysis_data = ?
                    WHERE id = ?
                ''', (analysis['score'], json.dumps(analysis), app[0]))
                analyzed_count += 1
        
        except Exception as e:
            print(f"Error analyzing application {app[0]}: {e}")
            continue
    
    conn.commit()
    conn.close()
    
    flash(f'Analyzed {analyzed_count} applications successfully!', 'success')
    return redirect(url_for('job_applications', job_id=job_id))

# Candidate routes
@app.route('/candidate/dashboard')
@candidate_required
def candidate_dashboard():
    conn = sqlite3.connect('recruitment.db')
    cursor = conn.cursor()
    
    # Get available jobs with application status
    cursor.execute('''
        SELECT j.id, j.title, j.description, j.recruiter_id, j.pdf_path, j.created_at, j.status,
               u.name as recruiter_name,
               CASE WHEN a.id IS NOT NULL THEN 1 ELSE 0 END as has_applied
        FROM jobs j
        JOIN users u ON j.recruiter_id = u.id
        LEFT JOIN applications a ON j.id = a.job_id AND a.candidate_id = ?
        WHERE j.status = 'active' OR j.status IS NULL
        ORDER BY j.created_at DESC
    ''', (session['user_id'],))
    jobs = cursor.fetchall()
    
    # Get candidate's applications
    cursor.execute('''
        SELECT a.*, j.title as job_title
        FROM applications a
        JOIN jobs j ON a.job_id = j.id
        WHERE a.candidate_id = ?
        ORDER BY a.applied_at DESC
    ''', (session['user_id'],))
    applications = cursor.fetchall()
    
    conn.close()
    return render_template('candidate_dashboard.html', jobs=jobs, applications=applications)

@app.route('/candidate/apply/<int:job_id>', methods=['GET', 'POST'])
@candidate_required
def apply_job(job_id):
    conn = sqlite3.connect('recruitment.db')
    cursor = conn.cursor()
    
    # Check if already applied
    cursor.execute('SELECT id FROM applications WHERE job_id = ? AND candidate_id = ?', 
                  (job_id, session['user_id']))
    if cursor.fetchone():
        flash('You have already applied for this job', 'error')
        return redirect(url_for('candidate_dashboard'))
    
    # Get job details
    cursor.execute('SELECT * FROM jobs WHERE id = ?', (job_id,))
    job = cursor.fetchone()
    
    if not job:
        flash('Job not found', 'error')
        return redirect(url_for('candidate_dashboard'))
    
    if request.method == 'POST':
        resume_file = request.files.get('resume')
        
        if not resume_file or not resume_file.filename.endswith('.pdf'):
            flash('Please upload a PDF resume', 'error')
            return render_template('apply_job.html', job=job)
        
        # Save resume
        filename = secure_filename(resume_file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = f"{timestamp}{session['user_id']}_{filename}"
        resume_path = os.path.join(app.config['UPLOAD_FOLDER'], 'resumes', filename)
        resume_file.save(resume_path)
        
        # Create application
        cursor.execute('''
            INSERT INTO applications (job_id, candidate_id, resume_path)
            VALUES (?, ?, ?)
        ''', (job_id, session['user_id'], resume_path))
        conn.commit()
        conn.close()
        
        flash('Application submitted successfully!', 'success')
        return redirect(url_for('candidate_dashboard'))
    
    conn.close()
    return render_template('apply_job.html', job=job)

@app.route('/application/<int:app_id>/details')
@login_required
def application_details(app_id):
    conn = sqlite3.connect('recruitment.db')
    cursor = conn.cursor()
    
    # Get application with job and candidate details
    cursor.execute('''
        SELECT a.*, j.title as job_title, u.name as candidate_name, u.email as candidate_email
        FROM applications a
        JOIN jobs j ON a.job_id = j.id
        JOIN users u ON a.candidate_id = u.id
        WHERE a.id = ?
    ''', (app_id,))
    application = cursor.fetchone()
    
    if not application:
        flash('Application not found', 'error')
        return redirect(url_for('candidate_dashboard' if session['role'] == 'candidate' else 'recruiter_dashboard'))
    
    # Check access permissions
    if session['role'] == 'candidate' and application[2] != session['user_id']:
        flash('Access denied', 'error')
        return redirect(url_for('candidate_dashboard'))
    elif session['role'] == 'recruiter':
        cursor.execute('SELECT recruiter_id FROM jobs WHERE id = ?', (application[1],))
        job_recruiter = cursor.fetchone()
        if not job_recruiter or job_recruiter[0] != session['user_id']:
            flash('Access denied', 'error')
            return redirect(url_for('recruiter_dashboard'))
    
    # Parse analysis data
    analysis_data = None
    if application[6]:  # analysis_data field
        try:
            analysis_data = json.loads(application[6])
        except:
            pass
    
    conn.close()
    return render_template('application_details.html', 
                         application=application, 
                         analysis_data=analysis_data)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)