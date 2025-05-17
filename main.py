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
# import queue 
import random

# Setup logging
logging.basicConfig(level=logging.DEBUG)
load_dotenv()

app = Flask(__name__, template_folder='.', static_folder='.')
app.secret_key = os.urandom(24)
os.makedirs('uploads', exist_ok=True)

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not os.getenv("OPENAI_API_KEY"):
        logging.warning("OPENAI_API_KEY not found. OpenAI dependent features will not work.")
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}")
    client = None

qna_evaluations = []
current_use_voice_mode = False # Tracks if current interview uses voice I/O

interview_context_template = {
    'questions_list': [], 'current_q_idx': 0, 'previous_answers_list': [], 'scores_list': [],
    'question_depth_counter': 0, 'max_followup_depth': 2, 'current_interview_track': None,
    'current_sub_track': None, 'questions_already_asked': set(), 'current_job_description': None,
    'use_camera_feature': False,
    'visual_analysis_data_session_key': 'visual_analysis_data_for_session',
    'visual_score_final_session_key': 'visual_score_final_for_session',
    'generated_resume_questions_cache': []
}
interview_context = {}

# PDF Question Loading 
structure = {
    'mba': { 'resume_flow': [], 'school_based': defaultdict(list), 'interest_areas': defaultdict(list) },
    'bank': { 'resume_flow': [], 'bank_type': defaultdict(list), 'technical_analytical': defaultdict(list) }
}
mba_pdf_path = "MBA_Question.pdf"
bank_pdf_path = "Bank_Question.pdf"

def normalize_text(text): return " ".join(text.strip().split()).lower()
def strip_numbering(text): return re.sub(r'^\d+\.\s*', '', text).strip()

def load_questions_into_memory(pdf_path, section_type): 
    if not os.path.exists(pdf_path): logging.error(f"PDF '{pdf_path}' not found."); return False
    try:
        with pdfplumber.open(pdf_path) as pdf: full_text = ''.join(page.extract_text() or '' for page in pdf.pages)
        lines = full_text.split('\n'); current_section = None; current_subsection = None
        for line in lines:
            line = line.strip();
            if not line: continue
            if section_type == 'mba':
                if "1. Resume Flow" in line: current_section = 'resume_flow'; current_subsection = None; continue
                elif "2. Pre-Defined Question Selection" in line: current_section = 'school_based'; current_subsection = None; continue
                elif "3. Interface to Select Question Areas" in line: current_section = 'interest_areas'; current_subsection = None; continue
                if current_section == 'school_based':
                    if "For IIMs" in line: current_subsection = 'IIM'; continue
                    elif "For ISB" in line: current_subsection = 'ISB'; continue
                    elif "For Other B-Schools" in line: current_subsection = 'Other'; continue
                if current_section == 'interest_areas':
                    if "General Business & Leadership" in line: current_subsection = 'General Business'; continue
                    elif "Finance & Economics" in line: current_subsection = 'Finance'; continue
                    elif "Marketing & Strategy" in line: current_subsection = 'Marketing'; continue
                    elif "Operations & Supply Chain" in line: current_subsection = 'Operations'; continue
            elif section_type == 'bank':
                if "Resume-Based Questions" in line: current_section = 'resume_flow'; current_subsection = None; continue
                elif "Bank-Type Specific Questions" in line: current_section = 'bank_type'; current_subsection = None; continue
                elif "Technical & Analytical Questions" in line: current_section = 'technical_analytical'; current_subsection = None; continue
                elif "Current Affairs" in line: current_section = 'technical_analytical'; current_subsection = 'Current Affairs'; continue
                if current_section == 'bank_type':
                    if "Public Sector Banks" in line: current_subsection = 'Public Sector Banks'; continue
                    elif "Private Banks" in line: current_subsection = 'Private Banks'; continue
                    elif "Regulatory Roles" in line: current_subsection = 'Regulatory Roles'; continue
                if current_section == 'technical_analytical' and current_subsection != 'Current Affairs':
                    if "Banking Knowledge" in line: current_subsection = 'Banking Knowledge'; continue
                    elif "Logical Reasoning" in line: current_subsection = 'Logical Reasoning'; continue
                    elif "Situational Judgement" in line: current_subsection = 'Situational Judgement'; continue
            if line and line[0].isdigit() and '.' in line.split()[0]:
                question = strip_numbering(line); is_sequence = bool(re.search(r'\d+,\s*\d+,\s*\d+.*,_', question))
                question_data = {'text': question, 'type': 'sequence' if is_sequence else 'standard'}
                if not question_data['text'].endswith('?'): question_data['text'] += '?'
                if current_section == 'resume_flow': structure[section_type]['resume_flow'].append(question_data)
                elif current_section and current_subsection: structure[section_type][current_section][current_subsection].append(question_data)
        logging.info(f"Loaded questions for {section_type}"); return True
    except Exception as e: logging.error(f"Err loading {pdf_path}: {e}", exc_info=True); return False

if not load_questions_into_memory(mba_pdf_path, 'mba'):
    logging.error("MBA questions fallback used."); structure['mba']['school_based']['IIM'] = [{'text': "Why IIM?", 'type': 'standard'}]
if not load_questions_into_memory(bank_pdf_path, 'bank'):
    logging.error("Bank questions fallback used."); structure['bank']['resume_flow'] = [{'text': "Your resume?", 'type': 'standard'}]

def get_openai_response_generic(prompt_messages, temperature=0.7, max_tokens=500):
    if not client: return "OpenAI client not available."
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=prompt_messages, temperature=temperature, max_tokens=max_tokens)
        return response.choices[0].message.content.strip()
    except Exception as e: logging.error(f"OpenAI API call err: {e}"); return f"Error: {e}"

def generate_resume_questions(resume_text, job_type, asked_qs_set): 
    if not resume_text: return ["Tell me about yourself?"]
    prompt = f"""Based on the following resume, generate 15 unique and relevant {'MBA' if job_type == 'mba' else 'banking'} interview questions tailored to the candidate's experience and background. Each question should be a complete sentence, concise, and end with a question mark. Avoid truncating questions mid-sentence. Resume: {resume_text}"""
    response_text = get_openai_response_generic([{"role": "user", "content": prompt}], max_tokens=1024) # Increased max_tokens slightly
    fallback_qs = ["What’s your biggest career achievement?", "What skills do you bring to this role?", "Describe a challenge in your last role?"]
    if "Error" in response_text or "OpenAI client not available" in response_text: return fallback_qs
    generated_qs = [strip_numbering(q.strip()) for q in response_text.split('\n') if q.strip() and q not in asked_qs_set]
    generated_qs = [q if q.endswith('?') else q + '?' for q in generated_qs]
    generated_qs = [q for q in generated_qs if 3 <= len(q.split()) <= 25 and q.endswith('?')]
    return generated_qs[:10] if len(generated_qs) >= 7 else fallback_qs + generated_qs[:3] 

def evaluate_response(question, answer, job_description):
    def evaluate_sequence_response_old(q_text, ans_text):
        if "2,5,10,17,26" in q_text.replace(" ",""): # Normalize spaces
            correct_answer = "37" # n^2+1 -> 6^2+1 = 37. 
            if str(ans_text).strip() == correct_answer: return "[Correct sequence completion] Score: 10/10", 10
            return f"[Incorrect sequence. Expected {correct_answer}] Score: 0/10", 0
        return "[Sequence evaluation not implemented for this pattern] Score: 3/10", 3
    is_sequence = bool(re.search(r'\d+,\s*\d+,\s*\d+.*,_', question))
    if is_sequence: return evaluate_sequence_response_old(question, answer)
    def fallback_evaluation_old(q,a):
        a_l=a.lower().strip(); q_k=set(normalize_text(q).split()); a_k=set(normalize_text(a).split())
        if len(a_l)<5 or not any(c.isalpha() for c in a_l) or not q_k.intersection(a_k): return "[Answer irrelevant/gibberish] Score: 0/10",0
        s=min(10,max(3,len(a.split())//5)); f="[Relevant, could use more detail]" if s<7 else "[Relevant and detailed]"
        return f"{f} Score: {s}/10",s
    eval_prompt = f"""Evaluate the answer for the question in a {job_description} context. Assess relevance, depth, insight. Question: {question}\nAnswer: {answer}\nProvide feedback and a score out of 10: - 0: Completely irrelevant, gibberish, or no answer. - 0-3: Barely relevant, lacks substance. - 3-8: Somewhat relevant, basic understanding, limited detail. - 8-9: Relevant, good understanding, decent detail. - 9-10: Highly relevant, detailed, insightful. Ensure the score reflects the answer's quality relative to the question. Format: '[Feedback] Score: X/10'"""
    resp_text = get_openai_response_generic([{"role": "user", "content": eval_prompt}], temperature=0.5, max_tokens=150)
    if "Error" in resp_text or "OpenAI client not available" in resp_text: return fallback_evaluation_old(question,answer)
    s_m=re.search(r'Score:\s*(\d+)/10',resp_text); score=int(s_m.group(1)) if s_m and s_m.group(1).isdigit() else 5
    return resp_text,score

def generate_next_question(question, answer, score, interview_track, job_type, asked_qs_set, attempt=1): 
    if attempt > 2: return None
    focus = 'experience, skills, career goals' if interview_track == 'resume' else \
            'academic motivations, school fit' if interview_track == 'school_based' else \
            'passion, knowledge, application' if interview_track == 'interest_areas' else \
            'banking operations, customer service' if interview_track == 'bank_type' else \
            'technical banking, logical reasoning' if interview_track == 'technical_analytical' else 'relevance'
    prompt = f"""Given Q&A for a {job_type} candidate (score: {score}/10), generate a related question.The question should be a complete sentence, concise, and end with a question mark.Focus on {focus}. Q: {question}\nA: {answer}\nScore: {score}/10"""
    resp_text = get_openai_response_generic([{"role": "user", "content": prompt}], max_tokens=100) # Shorter for follow-up
    fallback_q = "Can you elaborate on that?" # Default fallback
    if "Error" in resp_text or "OpenAI client not available" in resp_text: return fallback_q if attempt == 1 else None
    next_q = strip_numbering(resp_text.strip())
    if not next_q.endswith('?'): next_q += '?'
    if len(next_q.split()) > 20: next_q = ' '.join(next_q.split()[:20]) + '?' # Shorter follow-ups
    if next_q in asked_qs_set or not next_q or len(next_q.split()) < 3: return fallback_q if attempt == 1 else None
    return next_q

def generate_conversational_reply(answer, job_type): 
    sys_prompt = f"As a friendly {'HR' if job_type=='mba' else 'banking HR'} interviewer, generate a short, complete sentence as a reply to the candidate’s answer. Keep it engaging and human-like, and ensure it's a full thought. The reply must be a statement (ending with a period or exclamation mark) and must not contain any questions (do not end with a question mark). Provide only feedback or encouragement without asking for further information."
    resp_text = get_openai_response_generic([{"role":"system", "content":sys_prompt}, {"role":"user","content":answer}], temperature=0.8, max_tokens=60)
    if "Error" in resp_text or "OpenAI client not available" in resp_text: return "Thanks for your response."
    reply = resp_text.strip()
    if reply.endswith('?'): reply = reply[:-1] + '.'
    elif not reply.endswith(('.', '!')): reply += '.'
    if '?' in reply: reply = reply.replace('?', '.')
    return reply

def authenticate_user_db_old(username, password):
    try:
        conn = sqlite3.connect('users.db'); cursor = conn.cursor()
        cursor.execute('SELECT Allowed FROM users WHERE Username = ? AND Password = ?', (username, password)) # Case sensitive for password usually
        result = cursor.fetchone(); conn.close()
        return result[0] if result else None
        # Placeholder auth user code:
        if username == "mba_user" and password == "pass": return "MBA"
        if username == "bank_user" and password == "pass": return "Bank"
        return None
    except Exception as e: logging.error(f"Auth DB err: {e}"); return None

@app.route('/')
def index_route():
    if 'allowed_user_type' not in session: return redirect(url_for('login_html_route'))
    return render_template('index.html')

@app.route('/login.html')
def login_html_route(): return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post_route():
    username = request.form.get('username'); password = request.form.get('password')
    allowed_type = authenticate_user_db_old(username, password)
    if allowed_type:
        session['allowed_user_type'] = allowed_type; session['username'] = username
        return jsonify({'success': True, 'allowed': allowed_type})
    return jsonify({'success': False, 'error': 'Invalid username or password'}), 401

@app.route('/logout')
def logout_route():
    session.pop('allowed_user_type', None); session.pop('username', None)
    session.pop(interview_context_template['visual_analysis_data_session_key'], None)
    session.pop(interview_context_template['visual_score_final_session_key'], None)
    global interview_context, qna_evaluations; interview_context = {}; qna_evaluations = []
    return redirect(url_for('login_html_route'))

@app.route('/analyze_visuals', methods=['POST'])
def analyze_visuals_route():
    if 'allowed_user_type' not in session: return jsonify({"error": "Unauthorized"}), 401
    if not interview_context.get('use_camera_feature', False): return jsonify({"message":"Camera not enabled."}), 200
    sim_analysis = {"confidence": round(random.uniform(4,9.0),1), "ts":time.time()}
    key = interview_context_template['visual_analysis_data_session_key']
    if key not in session: session[key] = []
    session[key].append(sim_analysis)
    if len(session[key]) > 20: session[key].pop(0)
    session.modified = True
    return jsonify({"message":"Visuals received (simulated).", "analysis":sim_analysis})

@app.route('/start_interview', methods=['POST'])
def start_interview_route():
    global qna_evaluations, current_use_voice_mode, interview_context
    if 'allowed_user_type' not in session: return jsonify({"error": "Unauthorized"}), 401
    
    qna_evaluations = []
    interview_context = interview_context_template.copy()
    interview_context['questions_already_asked'] = set()
    interview_context['generated_resume_questions_cache'] = []
    session[interview_context['visual_analysis_data_session_key']] = []
    session[interview_context['visual_score_final_session_key']] = None

    allowed_type = session['allowed_user_type']
    current_use_voice_mode = request.form['mode'] == 'voice'
    track = request.form['interview_track']
    sub_track_val = request.form.get('sub_track', '') # Ensure this key matches HTML
    interview_context.update({
        'current_interview_track': track, 'current_sub_track': sub_track_val,
        'use_camera_feature': request.form.get('use_camera') == 'true'
    })
    
    if (allowed_type == 'MBA' and track in ['bank_type', 'technical_analytical']) or \
       (allowed_type == 'Bank' and track in ['school_based', 'interest_areas']):
        return jsonify({"error": "Access denied for this interview track."}), 403

    resume_file = request.files.get('resume')
    if not resume_file: return jsonify({"error": "Resume required"}), 400
    resume_text = ""; temp_path = os.path.join('uploads', resume_file.filename)
    try:
        resume_file.save(temp_path)
        if temp_path.lower().endswith('.pdf'):
            with pdfplumber.open(temp_path) as pdf: resume_text = ''.join(p.extract_text() or '' for p in pdf.pages)
        elif temp_path.lower().endswith('.docx'): resume_text = docx2txt.process(temp_path)
        else: return jsonify({"error": "Unsupported resume format"}), 400
    except Exception as e: logging.error(f"Resume process err: {e}"); return jsonify({"error":"Resume error"}),500
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)
    if not resume_text.strip(): resume_text = "Resume empty."

    job_key = 'mba' if allowed_type == 'MBA' else 'bank'
    interview_context['current_job_description'] = f"{allowed_type} Candidate"
    
    # Question selection logic 
    current_q_list = []
    if not interview_context['generated_resume_questions_cache']: # Generate only if not cached for this session
        interview_context['generated_resume_questions_cache'] = generate_resume_questions(resume_text, job_key, interview_context['questions_already_asked'])
    
    # Update already_asked with all generated resume questions at the start
    interview_context['questions_already_asked'].update(interview_context['generated_resume_questions_cache'])

    # Logic for question assembly
    if job_key == 'mba':
        if track == "resume":
            predef = [q['text'] for q in structure['mba']['resume_flow'][:3]]
            current_q_list = interview_context['generated_resume_questions_cache'] + [q for q in predef if q not in interview_context['questions_already_asked']]
        elif track == "school_based":
            school_qs = [q['text'] for q in structure['mba']['school_based'].get(sub_track_val, [])]
            if not school_qs: school_qs = [q['text'] for sl in structure['mba']['school_based'].values() for q in sl]
            current_q_list = interview_context['generated_resume_questions_cache'][:5] + [q for q in school_qs if q not in interview_context['questions_already_asked']]
        elif track == "interest_areas":
            interest_qs = [q['text'] for q in structure['mba']['interest_areas'].get(sub_track_val, [])]
            if not interest_qs: interest_qs = [q['text'] for sl in structure['mba']['interest_areas'].values() for q in sl]
            current_q_list = interview_context['generated_resume_questions_cache'][:5] + [q for q in interest_qs if q not in interview_context['questions_already_asked']]
    else: # bank
        if track == "resume":
            predef = [q['text'] for q in structure['bank']['resume_flow'][:3]]
            current_q_list = interview_context['generated_resume_questions_cache'] + [q for q in predef if q not in interview_context['questions_already_asked']]
        elif track == "bank_type":
            bank_qs = [q['text'] for q in structure['bank']['bank_type'].get(sub_track_val, [])]
            if not bank_qs: bank_qs = [q['text'] for sl in structure['bank']['bank_type'].values() for q in sl]
            current_q_list = interview_context['generated_resume_questions_cache'][:5] + [q for q in bank_qs if q not in interview_context['questions_already_asked']]
        elif track == "technical_analytical":
            tech_qs = [q['text'] for q in structure['bank']['technical_analytical'].get(sub_track_val, [])]
            if not tech_qs: tech_qs = [q['text'] for sl in structure['bank']['technical_analytical'].values() for q in sl]
            current_q_list = interview_context['generated_resume_questions_cache'][:5] + [q for q in tech_qs if q not in interview_context['questions_already_asked']]
    
    # Final unique question list for the interview from potentially mixed sources
    final_interview_questions = []
    temp_asked_this_round = set() # To ensure questions within this list are unique
    for q_text in current_q_list:
        stripped_q = strip_numbering(q_text)
        if stripped_q not in interview_context['questions_already_asked'] and stripped_q not in temp_asked_this_round:
            final_interview_questions.append(stripped_q)
            temp_asked_this_round.add(stripped_q) 
    
    interview_context['questions_list'] = final_interview_questions
    interview_context['questions_already_asked'].update(final_interview_questions) # Add chosen questions to master set

    if not interview_context['questions_list']:
        return jsonify({"error": f"No unique questions for {track}/{sub_track_val}"}), 400
    interview_context['current_q_idx'] = 0
    return jsonify({
        "message": f"Starting {job_key} interview", "total_questions": len(interview_context['questions_list']),
        "current_question": interview_context['questions_list'][0], "question_number": 1,
        "use_voice": current_use_voice_mode, "use_camera": interview_context['use_camera_feature']
    })

def calculate_final_overall_score(qna_evals, visual_score_0_to_10=None):
    qna_contrib = 0
    if qna_evals:
        total_pts = sum(e["score"] for e in qna_evals); max_pts = len(qna_evals) * 10
        if max_pts > 0: qna_contrib = (total_pts / max_pts) * 90.0
    visual_contrib = float(visual_score_0_to_10) if isinstance(visual_score_0_to_10, (int,float)) else 0.0
    overall = round(qna_contrib + visual_contrib, 2)
    return max(0, min(100, overall))

@app.route('/submit_answer', methods=['POST'])
def submit_answer_route():
    global qna_evaluations, current_use_voice_mode, interview_context
    if 'allowed_user_type' not in session: return jsonify({"error": "Unauthorized"}), 401
    if not interview_context or not interview_context.get('questions_list'):
        return jsonify({"error": "Interview not started/context lost."}), 400
    
    answer_text = request.json.get('answer', "").strip()
    if not answer_text: # If frontend sends empty after trim
        answer_text = "No answer provided by candidate."
        
    job_key = 'mba' if session['allowed_user_type'] == 'MBA' else 'bank'
    
    # Check if current_q_idx is valid before accessing questions_list
    if not (0 <= interview_context['current_q_idx'] < len(interview_context['questions_list'])):
        logging.error(f"Error: current_q_idx {interview_context.get('current_q_idx')} out of bounds for questions list of length {len(interview_context.get('questions_list', []))}. Ending interview.")
        # Gracefully end if something went wrong with question indexing
        final_visual_score = session.get(interview_context_template['visual_score_final_session_key'])
        overall_score = calculate_final_overall_score(qna_evaluations, final_visual_score)
        return jsonify({
            "reply": "An issue occurred with the question sequence. The interview will now conclude.", "finished": True,
            "evaluations": qna_evaluations, "overall_score": overall_score,
            "visual_score_details": {"score": final_visual_score if final_visual_score is not None else "N/A", "feedback": "Visual analysis (if enabled)."}
        })

    main_q_text = interview_context['questions_list'][interview_context['current_q_idx']]
    current_track = interview_context["current_interview_track"]
    job_desc = interview_context["current_job_description"]
    
    reply = generate_conversational_reply(answer_text, job_key)
    evaluation, score = evaluate_response(main_q_text, answer_text, job_desc)
    
    qna_evaluations.append({"question":main_q_text, "answer":answer_text, "evaluation":evaluation, "score":score})
    interview_context["previous_answers_list"].append(answer_text)
    interview_context["scores_list"].append(score)

    # Follow-up logic 
    if interview_context["question_depth_counter"] < interview_context["max_followup_depth"]:
        next_q = generate_next_question(main_q_text, answer_text, score, current_track, job_key, 
                                        interview_context['questions_already_asked'],
                                        attempt=1) # attempt is for generate_next_question internal use
        if not next_q and interview_context["question_depth_counter"] == 0: 
            next_q = "Can you elaborate on that?"
        
        if next_q and next_q not in interview_context['questions_already_asked']:
            interview_context['questions_list'].insert(interview_context['current_q_idx'] + 1, next_q)
            interview_context['questions_already_asked'].add(next_q)
            interview_context["question_depth_counter"] += 1
            interview_context['current_q_idx'] += 1
            return jsonify({
                "reply":reply, "current_question":next_q, 
                "question_number":interview_context['current_q_idx']+1, 
                "total_questions":len(interview_context['questions_list']), "next_question":True})
    
    interview_context["question_depth_counter"] = 0
    interview_context['current_q_idx'] += 1

    if interview_context['current_q_idx'] < len(interview_context['questions_list']):
        return jsonify({
            "reply":reply, "current_question":interview_context['questions_list'][interview_context['current_q_idx']],
            "question_number":interview_context['current_q_idx']+1, 
            "total_questions":len(interview_context['questions_list']), "next_question":True})
    else: 
        logging.info("Interview finished. No more questions to ask.")
        final_visual_score = None; visual_feedback = "Visual analysis not enabled."
        if interview_context.get('use_camera_feature'):
            analyses = session.get(interview_context_template['visual_analysis_data_session_key'], [])
            if analyses:
                confidences = [item.get('confidence',5) for item in analyses]
                avg_conf = sum(confidences)/len(confidences) if confidences else 5
                final_visual_score = round(max(0,min(10,avg_conf)),1)
                visual_feedback = f"Simulated avg confidence: {final_visual_score}/10."
            else: final_visual_score = 0; visual_feedback = "Camera on, no visual data."
            session[interview_context_template['visual_score_final_session_key']] = final_visual_score
        
        overall_score = calculate_final_overall_score(qna_evaluations, final_visual_score)
        return jsonify({
            "reply":"Thanks for the chat! That’s all for today.", "finished":True, 
            "evaluations":qna_evaluations, "overall_score":overall_score,
            "visual_score_details":{"score":final_visual_score if final_visual_score is not None else "N/A", "feedback":visual_feedback}})


@app.route('/generate_speech', methods=['POST'])
def generate_speech_route():
    if 'allowed_user_type' not in session: return jsonify({"error": "Unauthorized"}), 401
    if not client: return jsonify({"error": "OpenAI client not available."}), 500
    data = request.json; text = data.get('text',''); voice = data.get('voice','alloy')
    if not text: return jsonify({"error":"Text required"}),400
    
    # Standard OpenAI TTS voices
    supported_voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer', 'sage'] 
    if voice not in supported_voices: voice = 'alloy' # Fallback
    
    try:
        response = client.audio.speech.create(model="tts-1", voice=voice, input=text, response_format="mp3")
        return Response(response.content, mimetype='audio/mp3')
    except Exception as e:
        logging.error(f"TTS Error: {e}", exc_info=True)
        return jsonify({"error": f"TTS failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)