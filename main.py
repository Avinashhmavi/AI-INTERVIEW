import os
import json
import time
import sqlite3
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response
from openai import OpenAI
import pdfplumber
import docx2txt
from dotenv import load_dotenv
from collections import defaultdict
import logging
import re
import threading
import cv2
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

app = Flask(__name__, template_folder='.', static_folder='.')
app.secret_key = os.urandom(24)
os.makedirs('uploads', exist_ok=True)
os.makedirs('uploads/snapshots', exist_ok=True)

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not os.getenv("OPENAI_API_KEY"):
        logging.warning("OPENAI_API_KEY not found. OpenAI dependent features will not work.")
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}")
    client = None

qna_evaluations = []
current_use_voice_mode = False
listening_active = False
interview_context = {}
visual_analysis_thread = None
visual_analyses = []

interview_context_template = {
    'questions_list': [], 'current_q_idx': 0, 'previous_answers_list': [], 'scores_list': [],
    'question_depth_counter': 0, 'max_followup_depth': 2, 'current_interview_track': None,
    'current_sub_track': None, 'questions_already_asked': set(), 'current_job_description': None,
    'use_camera_feature': False,
    'visual_analysis_data_session_key': 'visual_analysis_data_for_session',
    'visual_score_final_session_key': 'visual_score_final_for_session',
    'generated_resume_questions_cache': []
}

# PDF Question Loading
structure = {
    'mba': {'resume_flow': [], 'school_based': defaultdict(list), 'interest_areas': defaultdict(list)},
    'bank': {'resume_flow': [], 'bank_type': defaultdict(list), 'technical_analytical': defaultdict(list)}
}

mba_pdf_path = "MBA_Question.pdf"
bank_pdf_path = "Bank_Question.pdf"

def normalize_text(text): 
    return " ".join(text.strip().split()).lower()

def strip_numbering(text): 
    return re.sub(r'^\d+\.\s*', '', text).strip()

def load_questions_into_memory(pdf_path, section_type):
    if not os.path.exists(pdf_path):
        logging.error(f"PDF '{pdf_path}' not found.")
        return False
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ''.join(page.extract_text() or '' for page in pdf.pages)
        lines = full_text.split('\n')
        current_section = None
        current_subsection = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if section_type == 'mba':
                if "1. Resume Flow" in line:
                    current_section = 'resume_flow'
                    current_subsection = None
                    continue
                elif "2. Pre-Defined Question Selection" in line:
                    current_section = 'school_based'
                    current_subsection = None
                    continue
                elif "3. Interface to Select Question Areas" in line:
                    current_section = 'interest_areas'
                    current_subsection = None
                    continue
                if current_section == 'school_based':
                    if "For IIMs" in line:
                        current_subsection = 'IIM'
                        continue
                    elif "For ISB" in line:
                        current_subsection = 'ISB'
                        continue
                    elif "For Other B-Schools" in line:
                        current_subsection = 'Other'
                        continue
                if current_section == 'interest_areas':
                    if "General Business & Leadership" in line:
                        current_subsection = 'General Business'
                        continue
                    elif "Finance & Economics" in line:
                        current_subsection = 'Finance'
                        continue
                    elif "Marketing & Strategy" in line:
                        current_subsection = 'Marketing'
                        continue
                    elif "Operations & Supply Chain" in line:
                        current_subsection = 'Operations'
                        continue
            elif section_type == 'bank':
                if "Resume-Based Questions" in line:
                    current_section = 'resume_flow'
                    current_subsection = None
                    continue
                elif "Bank-Type Specific Questions" in line:
                    current_section = 'bank_type'
                    current_subsection = None
                    continue
                elif "Technical & Analytical Questions" in line:
                    current_section = 'technical_analytical'
                    current_subsection = None
                    continue
                elif "Current Affairs" in line:
                    current_section = 'technical_analytical'
                    current_subsection = 'Current Affairs'
                    continue
                if current_section == 'bank_type':
                    if "Public Sector Banks" in line:
                        current_subsection = 'Public Sector Banks'
                        continue
                    elif "Private Banks" in line:
                        current_subsection = 'Private Banks'
                        continue
                    elif "Regulatory Roles" in line:
                        current_subsection = 'Regulatory Roles'
                        continue
                if current_section == 'technical_analytical' and current_subsection != 'Current Affairs':
                    if "Banking Knowledge" in line:
                        current_subsection = 'Banking Knowledge'
                        continue
                    elif "Logical Reasoning" in line:
                        current_subsection = 'Logical Reasoning'
                        continue
                    elif "Situational Judgement" in line:
                        current_subsection = 'Situational Judgement'
                        continue
            if line and line[0].isdigit() and '.' in line.split()[0]:
                question = strip_numbering(line)
                is_sequence = bool(re.search(r'\d+,\s*\d+,\s*\d+.*,_', question))
                question_data = {'text': question, 'type': 'sequence' if is_sequence else 'standard'}
                if not question_data['text'].endswith('?'):
                    question_data['text'] += '?'
                if current_section == 'resume_flow':
                    structure[section_type]['resume_flow'].append(question_data)
                elif current_section and current_subsection:
                    structure[section_type][current_section][current_subsection].append(question_data)
        logging.info(f"Loaded questions for {section_type}")
        return True
    except Exception as e:
        logging.error(f"Error loading {pdf_path}: {e}", exc_info=True)
        return False

if not load_questions_into_memory(mba_pdf_path, 'mba'):
    logging.error("MBA questions fallback used.")
    structure['mba']['school_based']['IIM'] = [{'text': "Why IIM?", 'type': 'standard'}]
if not load_questions_into_memory(bank_pdf_path, 'bank'):
    logging.error("Bank questions fallback used.")
    structure['bank']['resume_flow'] = [{'text': "Your resume?", 'type': 'standard'}]

def get_openai_response_generic(prompt_messages, temperature=0.7, max_tokens=500):
    if not client:
        logging.error("OpenAI client not available")
        return "OpenAI client not available."
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI API call error: {e}")
        return f"Error: {e}"

def generate_resume_questions(resume_text, job_type, asked_qs_set):
    if not resume_text:
        return ["Tell me about yourself?"]
    prompt = f"""Based on the following resume, generate 15 unique and relevant {'MBA' if job_type == 'mba' else 'banking'} interview questions tailored to the candidate's experience and background. Each question should be a complete sentence, concise, and end with a question mark. Avoid truncating questions mid-sentence. Resume: {resume_text}"""
    response_text = get_openai_response_generic([{"role": "user", "content": prompt}], max_tokens=1024)
    fallback_qs = ["What’s your biggest career achievement?", "What skills do you bring to this role?", "Describe a challenge in your last role?"]
    if "Error" in response_text or "OpenAI client not available" in response_text:
        return fallback_qs
    generated_qs = [strip_numbering(q.strip()) for q in response_text.split('\n') if q.strip() and q not in asked_qs_set]
    generated_qs = [q if q.endswith('?') else q + '?' for q in generated_qs]
    generated_qs = [q for q in generated_qs if 3 <= len(q.split()) <= 25 and q.endswith('?')]
    return generated_qs[:10] if len(generated_qs) >= 7 else fallback_qs + generated_qs[:3]

def generate_answer_feedback(question, answer, job_description):
    prompt = f"""
As an expert interviewer for {job_description}, provide concise, constructive feedback to help the candidate improve their interview performance. Focus on clarity, detail, relevance to the question, and communication skills. Provide 2-3 sentences of specific, actionable advice tailored to the answer's content and weaknesses. Avoid repeating the question or answer verbatim, and do not include scores or numerical ratings. Ensure the feedback is encouraging, professional, and unique for each response.

Question: {question}
Answer: {answer}
"""
    feedback = get_openai_response_generic([{"role": "user", "content": prompt}], temperature=0.7, max_tokens=120)
    if "Error" in feedback or "OpenAI client not available" in feedback:
        return "Provide specific examples and structure your response clearly to fully address the question."
    return feedback.strip()

CATEGORY_ALIASES = {
    "ideas": "Ideas",
    "organization": "Organization",
    "accuracy": "Accuracy",
    "voice": "Voice",
    "grammar usage and sentence fluency": "Grammar Usage and Sentence Fluency",
    "stop words": "Stop words"
}

WEIGHTS = {
    "Ideas": 0.2,
    "Organization": 0.25,
    "Accuracy": 0.2,
    "Voice": 0.2,
    "Grammar Usage and Sentence Fluency": 0.05,
    "Stop words": 0.1
}

def parse_response(raw_response):
    parsed = {}
    lines = [line.strip() for line in raw_response.split('\n') if line.strip()]
    current_category = None
    for line in lines:
        match = re.match(r'^Category:\s*(.+?)\s*\((\d+)(?:/10)?\)$', line, re.IGNORECASE)
        if match:
            category_alias = match.group(1).strip().lower()
            category = CATEGORY_ALIASES.get(category_alias, match.group(1).strip())
            score = int(match.group(2).strip())
            parsed[category] = {"score": score}
            current_category = category
            continue
        if current_category and line.lower().startswith("justification:"):
            justification = line.split(":", 1)[1].strip()
            parsed[current_category]["justification"] = justification
            current_category = None
    return parsed

def calculate_weighted_score(scores_dict):
    total = 0.0
    for category, values in scores_dict.items():
        score = values.get("score", 0)
        weight = WEIGHTS.get(category, 0)
        total += score * weight
    return round(total, 2)

def evaluate_sequence_response(question, answer):
    if "2,5,10,17,26" in question.replace(" ", ""):
        correct_answer = "37"
        try:
            user_answer = str(answer).strip()
            if user_answer == correct_answer:
                return "[Correct sequence completion] Score: 10/10", 10
            else:
                return f"[Incorrect sequence. Expected {correct_answer}] Score: 0/10", 0
        except Exception as e:
            logging.error(f"Error evaluating sequence answer: {e}")
            return "[Invalid answer format] Score: 0/10", 0
    return "[Sequence evaluation not implemented for this pattern] Score: 5/10", 5

def fallback_evaluation(question, answer):
    answer = answer.lower().strip()
    if len(answer) < 5 or not any(c.isalpha() for c in answer):
        return "[Answer is irrelevant or gibberish] Score: 0/10", 0
    question_keywords = set(normalize_text(question).split())
    answer_keywords = set(normalize_text(answer).split())
    common_keywords = question_keywords.intersection(answer_keywords)
    if not common_keywords:
        return "[Answer is irrelevant to the question] Score: 0/10", 0
    score = min(10, max(3, len(answer.split()) // 5))
    feedback = "[Answer is relevant but could use more detail]" if score < 7 else "[Answer is relevant and detailed]"
    return f"{feedback} Score: {score}/10", score

def evaluate_response(question, answer, job_description):
    is_sequence = bool(re.search(r'\d+,\s*\d+,\s*\d+.*,_', question))
    if is_sequence:
        return evaluate_sequence_response(question, answer)

    prompt = f"""
You are an interviewer for {job_description}, evaluating a candidate's answer based on:

1. Ideas:
The answer should focus on one clear idea, maintained throughout without tangents.

2. Organization:
Ideas should flow logically and cohesively.

3. Accuracy:
The answer should fully address all parts of the question.

4. Voice:
The answer should be unique and not generic.

5. Grammar Usage and Sentence Fluency:
The answer should use correct grammar and sentence structure.

6. Stop words:
Minimize filler words (e.g., uhh, ahh, ummm).

Provide a score (1-10, 1 lowest, 10 highest) for each category with a one-line justification.

Format the response exactly as:
Category: <category> (<score>/10)
Justification: <explanation>

List all six categories.

Question: {question}
Answer: {answer}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )
        evaluation_text = response.choices[0].message.content.strip()
        parsed_scores = parse_response(evaluation_text)
        if not parsed_scores:
            logging.warning("No scores parsed, using fallback evaluation")
            return fallback_evaluation(question, answer)
        weighted_score = calculate_weighted_score(parsed_scores)
        feedback = "[AI Evaluation Complete] "
        for cat, data in parsed_scores.items():
            feedback += f"{cat}: {data['score']}/10 | "
        feedback += f"Weighted Score: {weighted_score}/10"
        return feedback, weighted_score
    except Exception as e:
        logging.error(f"Error in OpenAI evaluation: {e}")
        return fallback_evaluation(question, answer)

def generate_next_question(question, answer, score, interview_track, job_type, asked_qs_set, attempt=1):
    if attempt > 2:
        return None
    focus = ('experience, skills, career goals' if interview_track == 'resume' else
             'academic motivations, school fit' if interview_track == 'school_based' else
             'passion, knowledge, application' if interview_track == 'interest_areas' else
             'banking operations, customer service' if interview_track == 'bank_type' else
             'technical banking, logical reasoning' if interview_track == 'technical_analytical' else 'relevance')
    prompt = f"""Given the Q&A for a {job_type} candidate (score: {score}/10), generate a related question.The question should be a complete sentence, concise, and end with a question mark. Focus on {focus}.
Q: {question}
A: {answer}
Score: {score}/10"""
    resp_text = get_openai_response_generic([{"role": "user", "content": prompt}], max_tokens=100)
    fallback_q = "Can you elaborate on that?"
    if "Error" in resp_text or "OpenAI client not available" in resp_text:
        return fallback_q if attempt == 1 else None
    next_q = strip_numbering(resp_text.strip())
    if not next_q.endswith('?'):
        next_q += '?'
    if len(next_q.split()) > 20:
        next_q = ' '.join(next_q.split()[:20]) + '?'
    if next_q in asked_qs_set or not next_q or len(next_q.split()) < 3:
        return fallback_q if attempt == 1 else None
    return next_q

def generate_conversational_reply(answer, job_type):
    sys_prompt = f"As a friendly {'HR' if job_type == 'mba' else 'banking HR'} interviewer, generate a short, complete sentence as a reply to the candidate’s answer. Keep it engaging and human-like, and ensure it's a full thought. The reply must be a statement (ending with a period or exclamation mark) and must not contain any questions (do not end with a question mark). Provide only feedback or encouragement without asking for further information."
    resp_text = get_openai_response_generic([{"role": "system", "content": sys_prompt}, {"role": "user", "content": answer}], temperature=0.8, max_tokens=60)
    if "Error" in resp_text or "OpenAI client not available" in resp_text:
        return "Thanks for your response."
    reply = resp_text.strip()
    if reply.endswith('?'):
        reply = reply[:-1] + '.'
    elif not reply.endswith(('.', '!')):
        reply += '.'
    if '?' in reply:
        reply = reply.replace('?', '.')
    return reply

def authenticate_user_db_old(username, password):
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT Allowed FROM users WHERE Username = ? AND Password = ?', (username, password))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    except Exception as e:
        logging.error(f"Auth DB error: {e}")
        return None

def analyze_frame(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            logging.error("Failed to load Haar cascade classifier")
            return {'eye_contact': False, 'confidence': 3.0, 'emotion': 'neutral', 'timestamp': time.time()}
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(150, 150))
        eye_contact = len(faces) > 0
        variance = np.var(gray)
        confidence = max(4, min(8, 8.5 - (variance / 8000)))  # Calibrated to 4-8 range
        brightness = np.mean(gray)
        emotion = 'positive' if brightness > 130 and variance < 20000 else 'neutral' if brightness > 80 else 'negative'
        return {
            'eye_contact': eye_contact,
            'confidence': round(confidence, 1),
            'emotion': emotion,
            'timestamp': time.time()
        }
    except Exception as e:
        logging.error(f"Error in analyze_frame: {e}")
        return {'eye_contact': False, 'confidence': 3.0, 'emotion': 'neutral', 'timestamp': time.time()}

def capture_and_analyze_visuals():
    global visual_analyses
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Failed to open webcam")
        return
    last_snapshot = 0
    try:
        while interview_context.get('use_camera_feature', False) and visual_analysis_thread and visual_analysis_thread.is_alive():
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to capture frame")
                time.sleep(0.1)
                continue
            analysis = analyze_frame(frame)
            visual_analyses.append(analysis)
            current_time = time.time()
            if current_time - last_snapshot >= 30:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join('uploads', 'snapshots', f"snapshot_{timestamp}.jpg")
                cv2.imwrite(path, frame)
                logging.info(f"Saved snapshot: {path}")
                last_snapshot = current_time
            time.sleep(0.1)
    except Exception as e:
        logging.error(f"Visual analysis error: {e}")
    finally:
        cap.release()

def calculate_visual_score():
    if not visual_analyses:
        return 0, "No visual data captured."
    try:
        eye_contact_ratio = sum(1 for a in visual_analyses if a['eye_contact']) / len(visual_analyses)
        avg_confidence = sum(a['confidence'] for a in visual_analyses) / len(visual_analyses)
        positive_emotions = sum(1 for a in visual_analyses if a['emotion'] == 'positive') / len(visual_analyses)
        eye_contact_score = (eye_contact_ratio * 10) * 0.4  # 40% weight
        confidence_score = (avg_confidence / 10) * 10 * 0.4  # 40% weight
        emotion_score = (positive_emotions * 10) * 0.2  # 20% weight
        total_score = eye_contact_score + confidence_score + emotion_score
        feedback = (f"Eye contact: {round(eye_contact_ratio*100)}% | "
                    f"Confidence: {round(avg_confidence,1)}/10 | "
                    f"Positive emotion: {round(positive_emotions*100)}%")
        return round(total_score, 1), feedback
    except Exception as e:
        logging.error(f"Error in calculate_visual_score: {e}")
        return 0, "Error calculating visual score."

@app.route('/')
def index_route():
    if 'allowed_user_type' not in session:
        return redirect(url_for('login_html_route'))
    return render_template('index.html')

@app.route('/login.html')
def login_html_route():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post_route():
    try:
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            logging.error("Missing username or password")
            return jsonify({'success': False, 'error': 'Username and password required'}), 400
        allowed_type = authenticate_user_db_old(username, password)
        if allowed_type:
            session['allowed_user_type'] = allowed_type
            session['username'] = username
            logging.info(f"User {username} logged in successfully")
            return jsonify({'success': True, 'allowed': allowed_type})
        logging.warning(f"Failed login attempt for username: {username}")
        return jsonify({'success': False, 'error': 'Invalid username or password'}), 401
    except Exception as e:
        logging.error(f"Error in login_post_route: {e}")
        return jsonify({'success': False, 'error': 'Server error'}), 500

@app.route('/logout')
def logout_route():
    global visual_analysis_thread, visual_analyses
    try:
        session.pop('allowed_user_type', None)
        session.pop('username', None)
        session.pop(interview_context_template['visual_analysis_data_session_key'], None)
        session.pop(interview_context_template['visual_score_final_session_key'], None)
        if visual_analysis_thread:
            visual_analysis_thread = None
        visual_analyses = []
        global interview_context, qna_evaluations
        interview_context = {}
        qna_evaluations = []
        logging.info("User logged out successfully")
        return redirect(url_for('login_html_route'))
    except Exception as e:
        logging.error(f"Error in logout_route: {e}")
        return jsonify({'error': 'Server error during logout'}), 500

@app.route('/analyze_visuals', methods=['POST'])
def analyze_visuals_route():
    try:
        if 'allowed_user_type' not in session:
            logging.error("Unauthorized access to analyze_visuals")
            return jsonify({"error": "Unauthorized"}), 401
        if not interview_context.get('use_camera_feature', False):
            return jsonify({"message": "Camera not enabled."}), 200
        return jsonify({"message": "Visual analysis running in background."})
    except Exception as e:
        logging.error(f"Error in analyze_visuals_route: {e}")
        return jsonify({"error": "Server error"}), 500

@app.route('/start_interview', methods=['POST'])
def start_interview_route():
    global qna_evaluations, current_use_voice_mode, interview_context, listening_active, visual_analysis_thread, visual_analyses
    try:
        if 'allowed_user_type' not in session:
            logging.error("Unauthorized access to start_interview")
            return jsonify({"error": "Unauthorized"}), 401
        qna_evaluations = []
        visual_analyses = []
        interview_context = interview_context_template.copy()
        interview_context['questions_already_asked'] = set()
        interview_context['generated_resume_questions_cache'] = []
        session[interview_context['visual_analysis_data_session_key']] = []
        session[interview_context['visual_score_final_session_key']] = None
        allowed_type = session['allowed_user_type']
        current_use_voice_mode = request.form.get('mode') == 'voice'
        track = request.form.get('interview_track')
        sub_track_val = request.form.get('sub_track', '')
        if not track:
            logging.error("Missing interview track")
            return jsonify({"error": "Interview track required"}), 400
        interview_context.update({
            'current_interview_track': track,
            'current_sub_track': sub_track_val,
            'use_camera_feature': request.form.get('use_camera') == 'true'
        })
        if (allowed_type == 'MBA' and track in ['bank_type', 'technical_analytical']) or \
           (allowed_type == 'Bank' and track in ['school_based', 'interest_areas']):
            logging.error(f"Access denied for track {track} by user type {allowed_type}")
            return jsonify({"error": "Access denied for this interview track."}), 403
        resume_file = request.files.get('resume')
        if not resume_file:
            logging.error("Resume file missing")
            return jsonify({"error": "Resume required"}), 400
        resume_text = ""
        temp_path = os.path.join('uploads', resume_file.filename)
        try:
            resume_file.save(temp_path)
            if temp_path.lower().endswith('.pdf'):
                with pdfplumber.open(temp_path) as pdf:
                    resume_text = ''.join(p.extract_text() or '' for p in pdf.pages)
            elif temp_path.lower().endswith('.docx'):
                resume_text = docx2txt.process(temp_path)
            else:
                logging.error(f"Unsupported resume format: {resume_file.filename}")
                return jsonify({"error": "Unsupported resume format"}), 400
        except Exception as e:
            logging.error(f"Resume processing error: {e}")
            return jsonify({"error": "Resume processing error"}), 500
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logging.warning(f"Failed to remove temp file {temp_path}: {e}")
        if not resume_text.strip():
            resume_text = "Resume empty."
        job_key = 'mba' if allowed_type == 'MBA' else 'bank'
        interview_context['current_job_description'] = f"{allowed_type} Candidate"
        current_q_list = []
        if not interview_context['generated_resume_questions_cache']:
            interview_context['generated_resume_questions_cache'] = generate_resume_questions(
                resume_text, job_key, interview_context['questions_already_asked'])
        interview_context['questions_already_asked'].update(interview_context['generated_resume_questions_cache'])
        if job_key == 'mba':
            if track == "resume":
                predef = [q['text'] for q in structure['mba']['resume_flow'][:3]]
                current_q_list = interview_context['generated_resume_questions_cache'] + \
                                 [q for q in predef if q not in interview_context['questions_already_asked']]
            elif track == "school_based":
                school_qs = [q['text'] for q in structure['mba']['school_based'].get(sub_track_val, [])]
                if not school_qs:
                    school_qs = [q['text'] for sl in structure['mba']['school_based'].values() for q in sl]
                current_q_list = interview_context['generated_resume_questions_cache'][:5] + \
                                 [q for q in school_qs if q not in interview_context['questions_already_asked']]
            elif track == "interest_areas":
                interest_qs = [q['text'] for q in structure['mba']['interest_areas'].get(sub_track_val, [])]
                if not interest_qs:
                    interest_qs = [q['text'] for sl in structure['mba']['interest_areas'].values() for q in sl]
                current_q_list = interview_context['generated_resume_questions_cache'][:5] + \
                                 [q for q in interest_qs if q not in interview_context['questions_already_asked']]
        else:
            if track == "resume":
                predef = [q['text'] for q in structure['bank']['resume_flow'][:3]]
                current_q_list = interview_context['generated_resume_questions_cache'] + \
                                 [q for q in predef if q not in interview_context['questions_already_asked']]
            elif track == "bank_type":
                bank_qs = [q['text'] for q in structure['bank']['bank_type'].get(sub_track_val, [])]
                if not bank_qs:
                    bank_qs = [q['text'] for sl in structure['bank']['bank_type'].values() for q in sl]
                current_q_list = interview_context['generated_resume_questions_cache'][:5] + \
                                 [q for q in bank_qs if q not in interview_context['questions_already_asked']]
            elif track == "technical_analytical":
                tech_qs = [q['text'] for q in structure['bank']['technical_analytical'].get(sub_track_val, [])]
                if not tech_qs:
                    tech_qs = [q['text'] for sl in structure['bank']['technical_analytical'].values() for q in sl]
                current_q_list = interview_context['generated_resume_questions_cache'][:5] + \
                                 [q for q in tech_qs if q not in interview_context['questions_already_asked']]
        final_interview_questions = []
        temp_asked_this_round = set()
        for q_text in current_q_list:
            stripped_q = strip_numbering(q_text)
            if stripped_q not in interview_context['questions_already_asked'] and stripped_q not in temp_asked_this_round:
                final_interview_questions.append(stripped_q)
                temp_asked_this_round.add(stripped_q)
        interview_context['questions_list'] = final_interview_questions
        interview_context['questions_already_asked'].update(final_interview_questions)
        if not interview_context['questions_list']:
            logging.error(f"No unique questions for {track}/{sub_track_val}")
            return jsonify({"error": f"No unique questions for {track}/{sub_track_val}"}), 400
        interview_context['current_q_idx'] = 0
        listening_active = True
        if interview_context['use_camera_feature']:
            visual_analysis_thread = threading.Thread(target=capture_and_analyze_visuals, daemon=True)
            visual_analysis_thread.start()
        logging.info(f"Started {job_key} interview with {len(interview_context['questions_list'])} questions")
        return jsonify({
            "message": f"Starting {job_key} interview",
            "total_questions": len(interview_context['questions_list']),
            "current_question": interview_context['questions_list'][0],
            "question_number": 1,
            "use_voice": current_use_voice_mode,
            "use_camera": interview_context['use_camera_feature']
        })
    except Exception as e:
        logging.error(f"Error in start_interview_route: {e}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def calculate_final_overall_score(qna_evals, visual_score_0_to_10=None):
    try:
        qna_contrib = 0
        if qna_evals:
            total_pts = sum(e["score"] for e in qna_evals)
            max_pts = len(qna_evals) * 10
            if max_pts > 0:
                qna_contrib = (total_pts / max_pts) * 90.0
        visual_contrib = float(visual_score_0_to_10) if isinstance(visual_score_0_to_10, (int, float)) else 0.0
        overall = round(qna_contrib + visual_contrib, 2)
        return max(0, min(100, overall))
    except Exception as e:
        logging.error(f"Error in calculate_final_overall_score: {e}")
        return 0

@app.route('/submit_answer', methods=['POST'])
def submit_answer_route():
    global qna_evaluations, current_use_voice_mode, interview_context, listening_active
    try:
        if 'allowed_user_type' not in session:
            logging.error("Unauthorized access to submit_answer")
            return jsonify({"error": "Unauthorized"}), 401
        if not interview_context or not interview_context.get('questions_list'):
            logging.error("Interview context missing or not started")
            return jsonify({"error": "Interview not started/context lost."}), 400

        if not request.is_json:
            logging.error("Invalid request: JSON expected")
            return jsonify({"error": "Invalid request: JSON expected"}), 400

        data = request.get_json()
        answer_text = data.get('answer', "").strip()
        if not answer_text:
            answer_text = "No answer provided by candidate."
        logging.info(f"Received answer for question {interview_context['current_q_idx'] + 1}: {answer_text[:50]}...")

        job_key = 'mba' if session['allowed_user_type'] == 'MBA' else 'bank'
        if not (0 <= interview_context['current_q_idx'] < len(interview_context['questions_list'])):
            logging.warning("Invalid question index, ending interview")
            final_visual_score, visual_feedback = calculate_visual_score()
            session[interview_context_template['visual_score_final_session_key']] = final_visual_score
            overall_score = calculate_final_overall_score(qna_evaluations, final_visual_score)
            for eval_item in qna_evaluations:
                eval_item['feedback'] = generate_answer_feedback(
                    eval_item['question'],
                    eval_item['answer'],
                    interview_context['current_job_description']
                )
            listening_active = False
            logging.info("Interview concluded due to invalid question index")
            return jsonify({
                "reply": "An issue occurred with the question sequence. The interview will now conclude.",
                "finished": True,
                "evaluations": qna_evaluations,
                "overall_score": overall_score,
                "visual_score_details": {
                    "score": final_visual_score if final_visual_score is not None else "N/A",
                    "feedback": visual_feedback
                }
            })

        main_q_text = interview_context['questions_list'][interview_context['current_q_idx']]
        current_track = interview_context["current_interview_track"]
        job_desc = interview_context["current_job_description"]
        reply = generate_conversational_reply(answer_text, job_key)
        evaluation, score = evaluate_response(main_q_text, answer_text, job_desc)
        feedback = generate_answer_feedback(main_q_text, answer_text, job_desc)
        qna_evaluations.append({
            "question": main_q_text,
            "answer": answer_text,
            "evaluation": evaluation,
            "score": score,
            "feedback": feedback
        })
        interview_context["previous_answers_list"].append(answer_text)
        interview_context["scores_list"].append(score)

        if interview_context["question_depth_counter"] < interview_context["max_followup_depth"]:
            next_q = generate_next_question(
                main_q_text, answer_text, score, current_track, job_key,
                interview_context['questions_already_asked'], attempt=1
            )
            if not next_q and interview_context["question_depth_counter"] == 0:
                next_q = "Can you elaborate on that?"
            if next_q and next_q not in interview_context['questions_already_asked']:
                interview_context['questions_list'].insert(interview_context['current_q_idx'] + 1, next_q)
                interview_context['questions_already_asked'].add(next_q)
                interview_context["question_depth_counter"] += 1
                interview_context['current_q_idx'] += 1
                listening_active = True
                logging.info(f"Generated follow-up question: {next_q}")
                return jsonify({
                    "reply": reply,
                    "current_question": next_q,
                    "question_number": interview_context['current_q_idx'] + 1,
                    "total_questions": len(interview_context['questions_list']),
                    "next_question": True
                })

        interview_context["question_depth_counter"] = 0
        interview_context['current_q_idx'] += 1
        if interview_context['current_q_idx'] < len(interview_context['questions_list']):
            listening_active = True
            next_question = interview_context['questions_list'][interview_context['current_q_idx']]
            logging.info(f"Proceeding to next question: {next_question}")
            return jsonify({
                "reply": reply,
                "current_question": next_question,
                "question_number": interview_context['current_q_idx'] + 1,
                "total_questions": len(interview_context['questions_list']),
                "next_question": True
            })
        else:
            logging.info("Interview finished. No more questions to ask.")
            final_visual_score, visual_feedback = calculate_visual_score()
            session[interview_context_template['visual_score_final_session_key']] = final_visual_score
            overall_score = calculate_final_overall_score(qna_evaluations, final_visual_score)
            for eval_item in qna_evaluations:
                if 'feedback' not in eval_item:
                    eval_item['feedback'] = generate_answer_feedback(
                        eval_item['question'],
                        eval_item['answer'],
                        interview_context['current_job_description']
                    )
            listening_active = False
            logging.info(f"Interview completed. Overall score: {overall_score}")
            return jsonify({
                "reply": "Thanks for the chat! That’s all for today.",
                "finished": True,
                "evaluations": qna_evaluations,
                "overall_score": overall_score,
                "visual_score_details": {
                    "score": final_visual_score if final_visual_score is not None else "N/A",
                    "feedback": visual_feedback
                }
            })

    except Exception as e:
        logging.error(f"Error in submit_answer_route: {e}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/generate_speech', methods=['POST'])
def generate_speech_route():
    try:
        if 'allowed_user_type' not in session:
            logging.error("Unauthorized access to generate_speech")
            return jsonify({"error": "Unauthorized"}), 401
        if not client:
            logging.error("OpenAI client not available for TTS")
            return jsonify({"error": "OpenAI client not available."}), 500
        if not request.is_json:
            logging.error("Invalid request: JSON expected for TTS")
            return jsonify({"error": "Invalid request: JSON expected"}), 400
        data = request.get_json()
        text = data.get('text', '')
        voice = data.get('voice', 'alloy')
        if not text:
            logging.error("Text required for TTS")
            return jsonify({"error": "Text required"}), 400
        supported_voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer', 'sage']
        if voice not in supported_voices:
            voice = 'alloy'
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="mp3"
        )
        logging.info("Generated speech successfully")
        return Response(response.content, mimetype='audio/mp3')
    except Exception as e:
        logging.error(f"TTS Error: {e}", exc_info=True)
        return jsonify({"error": f"TTS failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
