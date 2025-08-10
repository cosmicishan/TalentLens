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
import seaborn as sns
from groq import Groq
import re
from functools import wraps
import numpy as np
from io import BytesIO
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'SYGDYE%7656TYFYT76566fygftq43652rt^6$^%4'  
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

# Resume analysis functions
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
        # Check if GROQ_API_KEY is set
        if not os.getenv('GROQ_API_KEY'):
            print("Warning: GROQ_API_KEY environment variable not set")
            return create_dummy_analysis(candidate_name)
        
        client = Groq()
        
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
        return create_dummy_analysis(candidate_name)

def create_dummy_analysis(candidate_name):
    """Create a dummy analysis for testing when Groq API is not available."""
    return {
        "name": candidate_name,
        "score": 75,
        "score_breakdown": {
            "skills_score": 80,
            "experience_score": 70,
            "education_score": 75,
            "certifications_score": 60,
            "infra_deployment_score": 70,
            "culture_fit_score": 85,
            "weights": {
                "skills_weight": 0.25,
                "experience_weight": 0.25,
                "education_weight": 0.15,
                "certifications_weight": 0.10,
                "infra_deployment_weight": 0.15,
                "culture_fit_weight": 0.10
            }
        },
        "matching_analysis": "Good technical background with relevant skills. Experience aligns well with requirements.",
        "description": "Strong candidate with solid technical foundation. Good fit for the role.",
        "recommendation": "Consider for interview. May benefit from additional training in specific technologies.",
        "metadata": {
            "years_experience": 5,
            "seniority": "mid",
            "education_level": "BS",
            "certifications": ["AWS Certified"],
            "matched_skills": ["Python", "JavaScript", "React"],
            "missing_skills": ["Docker", "Kubernetes"],
            "domain_experience": ["web development"],
            "cloud_experience": ["AWS"],
            "ml_infra_experience": ["docker"],
            "vector_db_experience": ["None"],
            "deployment_pipeline_experience": ["CI/CD"],
            "leadership_years": 2,
            "publications_or_patents": [],
            "languages": ["Python", "JavaScript"],
            "location_or_remote": "Remote",
            "interview_readiness": "Ready"
        },
        "analysis_steps": ["Analyzed skills match", "Evaluated experience level", "Assessed cultural fit"],
        "timestamp": datetime.now().isoformat()
    }

def generate_candidate_charts(analysis_data, candidate_name):
    """Generate comprehensive charts for candidate analysis."""
    if not analysis_data:
        return {}
    
    # Set style for professional plots
    plt.style.use('default')
    try:
        sns.set_palette("husl")
    except:
        pass
    
    charts = {}
    
    try:
        # Chart 1: Overall Score Gauge
        fig, ax = plt.subplots(figsize=(8, 6))
        score = analysis_data.get('score', 0)
        
        # Create gauge chart
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Background
        ax.plot(theta, r, 'lightgray', linewidth=20)
        
        # Score arc
        if score > 0:
            score_theta = np.linspace(0, np.pi * (score/100), max(1, int(score)))
            score_r = np.ones_like(score_theta)
            
            if score >= 85:
                color = '#28a745'
            elif score >= 70:
                color = '#ffc107'
            elif score >= 50:
                color = '#fd7e14'
            else:
                color = '#dc3545'
                
            ax.plot(score_theta, score_r, color, linewidth=20)
        else:
            color = '#dc3545'
        
        # Add score text
        ax.text(np.pi/2, 0.5, f'{score}%', ha='center', va='center', 
                fontsize=32, fontweight='bold', color=color)
        ax.text(np.pi/2, 0.2, 'Match Score', ha='center', va='center', 
                fontsize=14, color='gray')
        
        ax.set_ylim(0, 1.2)
        ax.set_xlim(0, np.pi)
        ax.axis('off')
        ax.set_title(f'{candidate_name} - Overall Compatibility', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        charts['overall_score'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # Chart 2: Component Scores Radar Chart
        if 'score_breakdown' in analysis_data:
            scores = analysis_data['score_breakdown']
            categories = ['Skills', 'Experience', 'Education', 'Certifications', 'Infra/Deploy', 'Culture Fit']
            values = [
                scores.get('skills_score', 0),
                scores.get('experience_score', 0),
                scores.get('education_score', 0),
                scores.get('certifications_score', 0),
                scores.get('infra_deployment_score', 0),
                scores.get('culture_fit_score', 0)
            ]
            
            # Add first value at end to close the radar chart
            values += values[:1]
            
            # Calculate angles for radar chart
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            # Plot the radar chart
            ax.plot(angles, values, 'o-', linewidth=2, color='#667eea')
            ax.fill(angles, values, alpha=0.25, color='#667eea')
            
            # Add category labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 100)
            
            # Add grid lines
            ax.grid(True)
            ax.set_title('Component Score Breakdown', size=16, fontweight='bold', pad=20)
            
            plt.tight_layout()
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            charts['radar_chart'] = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
        
        # Chart 3: Skills Analysis
        if 'metadata' in analysis_data:
            metadata = analysis_data['metadata']
            matched_skills = metadata.get('matched_skills', [])
            missing_skills = metadata.get('missing_skills', [])
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Skills pie chart
            sizes = [len(matched_skills), len(missing_skills)]
            labels = ['Matched Skills', 'Missing Skills']
            colors = ['#28a745', '#dc3545']
            
            if sum(sizes) > 0:
                ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax1.set_title('Skill Match Overview', fontweight='bold')
            else:
                ax1.text(0.5, 0.5, 'No skill data available', ha='center', va='center',
                        transform=ax1.transAxes, fontsize=12, color='gray')
                ax1.set_title('Skill Match Overview', fontweight='bold')
            
            # Top skills bar chart
            if matched_skills:
                top_skills = matched_skills[:8]
                y_pos = np.arange(len(top_skills))
                
                ax2.barh(y_pos, [1]*len(top_skills), color='#28a745', alpha=0.7)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(top_skills)
                ax2.set_xlabel('Skill Presence')
                ax2.set_title('Top Matched Skills', fontweight='bold')
                ax2.grid(axis='x', alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No matched skills\nreported', ha='center', va='center',
                        transform=ax2.transAxes, fontsize=12, color='gray')
                ax2.set_title('Top Matched Skills', fontweight='bold')
            
            plt.tight_layout()
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            charts['skills_analysis'] = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
    except Exception as e:
        print(f"Error generating charts: {e}")
    
    return charts

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
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        role = request.form.get('role', '')
        
        # Validation
        if not all([name, email, password, role]):
            flash('All fields are required', 'error')
            return render_template('register.html')
        
        if role not in ['recruiter', 'candidate']:
            flash('Invalid role selected', 'error')
            return render_template('register.html')
        
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
        except Exception as e:
            flash('Registration failed. Please try again.', 'error')
            print(f"Registration error: {e}")
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        
        if not email or not password:
            flash('Email and password are required', 'error')
            return render_template('login.html')
        
        conn = sqlite3.connect('recruitment.db')
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT id, name, password_hash, role FROM users WHERE email = ?', (email,))
            user = cursor.fetchone()
            
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
        except Exception as e:
            flash('Login failed. Please try again.', 'error')
            print(f"Login error: {e}")
        finally:
            conn.close()
    
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
    
    try:
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
    except Exception as e:
        flash('Error loading dashboard', 'error')
        print(f"Dashboard error: {e}")
        jobs = []
    finally:
        conn.close()
    
    return render_template('recruiter_dashboard.html', jobs=jobs)

@app.route('/recruiter/post-job', methods=['GET', 'POST'])
@recruiter_required
def post_job():
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        description = request.form.get('description', '').strip()
        job_file = request.files.get('job_description_pdf')
        
        if not title:
            flash('Job title is required', 'error')
            return render_template('post_job.html')
        
        pdf_path = None
        if job_file and job_file.filename and job_file.filename.endswith('.pdf'):
            filename = secure_filename(job_file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filename = timestamp + filename
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'job_descriptions', filename)
            try:
                job_file.save(pdf_path)
            except Exception as e:
                flash('Error saving job description file', 'error')
                print(f"File save error: {e}")
                return render_template('post_job.html')
        
        conn = sqlite3.connect('recruitment.db')
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO jobs (title, description, recruiter_id, pdf_path)
                VALUES (?, ?, ?, ?)
            ''', (title, description, session['user_id'], pdf_path))
            conn.commit()
            flash('Job posted successfully!', 'success')
            return redirect(url_for('recruiter_dashboard'))
        except Exception as e:
            flash('Error posting job. Please try again.', 'error')
            print(f"Job posting error: {e}")
        finally:
            conn.close()
    
    return render_template('post_job.html')

@app.route('/recruiter/job/<int:job_id>/applications')
@recruiter_required
def job_applications(job_id):
    conn = sqlite3.connect('recruitment.db')
    cursor = conn.cursor()
    
    try:
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
            ORDER BY CASE WHEN a.match_score IS NULL THEN 1 ELSE 0 END, 
                     a.match_score DESC, a.applied_at DESC
        ''', (job_id,))
        applications = cursor.fetchall()
        
    except Exception as e:
        flash('Error loading applications', 'error')
        print(f"Applications loading error: {e}")
        return redirect(url_for('recruiter_dashboard'))
    finally:
        conn.close()
    
    return render_template('job_applications.html', job=job, applications=applications)

@app.route('/recruiter/analyze-applications/<int:job_id>')
@recruiter_required
def analyze_applications(job_id):
    """Analyze all applications for a job using the resume matching algorithm."""
    conn = sqlite3.connect('recruitment.db')
    cursor = conn.cursor()
    
    try:
        # Get job details
        cursor.execute('SELECT * FROM jobs WHERE id = ? AND recruiter_id = ?', 
                      (job_id, session['user_id']))
        job = cursor.fetchone()
        
        if not job:
            flash('Job not found', 'error')
            return redirect(url_for('recruiter_dashboard'))
        
        # Extract job description text
        job_text = job[2] or ""
        if job[4]:  # pdf_path field
            try:
                pdf_text = extract_text_from_pdf(job[4])
                if pdf_text:
                    job_text = pdf_text
            except Exception as e:
                print(f"Error extracting job PDF: {e}")
        
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
                if not app[3]:  # resume_path
                    continue
                    
                resume_text = extract_text_from_pdf(app[3])
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
        
        if analyzed_count > 0:
            flash(f'Analyzed {analyzed_count} applications successfully!', 'success')
        else:
            flash('No new applications to analyze', 'info')
            
    except Exception as e:
        flash('Error during analysis. Please try again.', 'error')
        print(f"Analysis error: {e}")
    finally:
        conn.close()
    
    return redirect(url_for('job_applications', job_id=job_id))

# Candidate routes
@app.route('/candidate/dashboard')
@candidate_required
def candidate_dashboard():
    conn = sqlite3.connect('recruitment.db')
    cursor = conn.cursor()
    
    try:
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
        
    except Exception as e:
        flash('Error loading dashboard', 'error')
        print(f"Candidate dashboard error: {e}")
        jobs = []
        applications = []
    finally:
        conn.close()
    
    return render_template('candidate_dashboard.html', jobs=jobs, applications=applications)

@app.route('/candidate/apply/<int:job_id>', methods=['GET', 'POST'])
@candidate_required
def apply_job(job_id):
    conn = sqlite3.connect('recruitment.db')
    cursor = conn.cursor()
    
    try:
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
            
            if not resume_file or not resume_file.filename:
                flash('Please upload a resume', 'error')
                return render_template('apply_job.html', job=job)
                
            if not resume_file.filename.lower().endswith('.pdf'):
                flash('Please upload a PDF resume', 'error')
                return render_template('apply_job.html', job=job)
            
            # Save resume
            filename = secure_filename(resume_file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filename = f"{timestamp}{session['user_id']}_{filename}"
            resume_path = os.path.join(app.config['UPLOAD_FOLDER'], 'resumes', filename)
            
            try:
                resume_file.save(resume_path)
            except Exception as e:
                flash('Error saving resume file', 'error')
                print(f"Resume save error: {e}")
                return render_template('apply_job.html', job=job)
            
            # Create application
            cursor.execute('''
                INSERT INTO applications (job_id, candidate_id, resume_path)
                VALUES (?, ?, ?)
            ''', (job_id, session['user_id'], resume_path))
            conn.commit()
            
            flash('Application submitted successfully!', 'success')
            return redirect(url_for('candidate_dashboard'))
        
    except Exception as e:
        flash('Error processing application', 'error')
        print(f"Apply job error: {e}")
        return redirect(url_for('candidate_dashboard'))
    finally:
        conn.close()
    
    return render_template('apply_job.html', job=job)

@app.route('/application/<int:app_id>/details')
@login_required
def application_details(app_id):
    conn = sqlite3.connect('recruitment.db')
    cursor = conn.cursor()
    
    try:
        # Get application with job and candidate details
        cursor.execute('''
            SELECT a.*, j.title as job_title, j.recruiter_id, u.name as candidate_name, u.email as candidate_email
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
        elif session['role'] == 'recruiter' and application[9] != session['user_id']:  # recruiter_id
            flash('Access denied', 'error')
            return redirect(url_for('recruiter_dashboard'))
        
        # Parse analysis data
        analysis_data = None
        charts = {}
        if application[6]:  # analysis_data field
            try:
                analysis_data = json.loads(application[6])
                # Generate charts for this candidate
                charts = generate_candidate_charts(analysis_data, application[10])  # candidate_name
            except Exception as e:
                print(f"Error parsing analysis data: {e}")
        
    except Exception as e:
        flash('Error loading application details', 'error')
        print(f"Application details error: {e}")
        return redirect(url_for('candidate_dashboard' if session['role'] == 'candidate' else 'recruiter_dashboard'))
    finally:
        conn.close()
    
    return render_template('application_details.html', 
                         application=application, 
                         analysis_data=analysis_data,
                         charts=charts)

@app.route('/download-resume/<int:app_id>')
@login_required
def download_resume(app_id):
    conn = sqlite3.connect('recruitment.db')
    cursor = conn.cursor()
    
    try:
        # Get application details
        cursor.execute('''
            SELECT a.resume_path, a.candidate_id, j.recruiter_id, u.name as candidate_name
            FROM applications a
            JOIN jobs j ON a.job_id = j.id
            JOIN users u ON a.candidate_id = u.id
            WHERE a.id = ?
        ''', (app_id,))
        result = cursor.fetchone()
        
        if not result:
            flash('Application not found', 'error')
            return redirect(url_for('recruiter_dashboard'))
        
        resume_path, candidate_id, recruiter_id, candidate_name = result
        
        # Check permissions
        if session['role'] == 'recruiter' and recruiter_id != session['user_id']:
            flash('Access denied', 'error')
            return redirect(url_for('recruiter_dashboard'))
        elif session['role'] == 'candidate' and candidate_id != session['user_id']:
            flash('Access denied', 'error')
            return redirect(url_for('candidate_dashboard'))
        
        # Check if file exists
        if not resume_path or not os.path.exists(resume_path):
            flash('Resume file not found', 'error')
            return redirect(url_for('application_details', app_id=app_id))
        
        return send_file(resume_path, as_attachment=True, 
                        download_name=f"{candidate_name}_resume.pdf")
        
    except Exception as e:
        flash('Error downloading resume', 'error')
        print(f"Download resume error: {e}")
        return redirect(url_for('recruiter_dashboard'))
    finally:
        conn.close()

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True)