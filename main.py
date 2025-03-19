import os
import json
import time
from flask import Flask, render_template, request, jsonify
from groq import Groq
import pdfplumber
import docx2txt
from dotenv import load_dotenv
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, template_folder='.', static_folder='.')
os.makedirs('uploads', exist_ok=True)

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Global state management
questions = []
current_question = 0
evaluations = []
job_description = "MBA Candidate"
use_voice = False
asked_questions = set()
resume_questions = []

# Interview context
interview_context = {
    'questions': [],
    'current_question_idx': 0,
    'previous_answers': [],
    'scores': [],
    'follow_up_depth': 0,
    'max_follow_ups': 2,
    'interview_track': None,
    'sub_track': None,
    'asked_questions': set()
}

# Structure for organizing predefined questions
structure = {
    'resume_flow': [],
    'school_based': defaultdict(list),
    'interest_areas': defaultdict(list)
}

# PDF path for MBA questions
pdf_path = "MBA_Question.pdf"

# Weightage constants
DEFAULT_TECHNICAL_WEIGHT = 0.8
DEFAULT_PERSONAL_WEIGHT = 0.2
INTERPERSONAL_TECHNICAL_WEIGHT = 0.4
INTERPERSONAL_WEIGHT = 0.4
INTERPERSONAL_PERSONAL_WEIGHT = 0.2
INTERPERSONAL_KEYWORDS = ["team", "collaboration", "communication", "leadership", "conflict", "group", "colleague"]

def normalize_text(text):
    return " ".join(text.strip().split()).lower()

def load_questions_into_memory():
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file '{pdf_path}' not found.")
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
            
            if "1. Resume Flow" in line:
                current_section = 'resume_flow'
                current_subsection = None
                logging.debug("Switched to Resume Flow")
                continue
            elif "2. Pre-Defined Question Selection" in line:
                current_section = 'school_based'
                current_subsection = None
                logging.debug("Switched to School Based")
                continue
            elif "3. Interface to Select Question Areas" in line:
                current_section = 'interest_areas'
                current_subsection = None
                logging.debug("Switched to Interest Areas")
                continue
            
            elif current_section == 'school_based':
                if "For IIMs" in line:
                    current_subsection = 'IIM'
                    logging.debug("Switched to IIM")
                    continue
                elif "For ISB" in line:
                    current_subsection = 'ISB'
                    logging.debug("Switched to ISB")
                    continue
                elif "For Other B-Schools" in line:
                    current_subsection = 'Other'
                    logging.debug("Switched to Other B-Schools")
                    continue
            
            elif current_section == 'interest_areas':
                if "General Business & Leadership" in line:
                    current_subsection = 'General Business'
                    logging.debug("Switched to General Business")
                    continue
                elif "Finance & Economics" in line:
                    current_subsection = 'Finance'
                    logging.debug("Switched to Finance")
                    continue
                elif "Marketing & Strategy" in line:
                    current_subsection = 'Marketing'
                    logging.debug("Switched to Marketing")
                    continue
                elif "Operations & Supply Chain" in line:
                    current_subsection = 'Operations'
                    logging.debug("Switched to Operations")
                    continue
            
            if line and line[0].isdigit() and '.' in line.split()[0]:
                question = line.split('.', 1)[1].strip()
                if current_section == 'resume_flow':
                    structure['resume_flow'].append(question)
                    logging.debug(f"Added to resume_flow: {question}")
                elif current_section == 'school_based' and current_subsection:
                    structure['school_based'][current_subsection].append(question)
                    logging.debug(f"Added to school_based[{current_subsection}]: {question}")
                elif current_section == 'interest_areas' and current_subsection:
                    structure['interest_areas'][current_subsection].append(question)
                    logging.debug(f"Added to interest_areas[{current_subsection}]: {question}")
        
        logging.info(f"Loaded predefined questions: resume_flow={len(structure['resume_flow'])}, "
                     f"school_based={dict(structure['school_based'])}, "
                     f"interest_areas={dict(structure['interest_areas'])}")
        return True
    except Exception as e:
        logging.error(f"Error loading questions from PDF: {e}")
        return False

if not load_questions_into_memory():
    logging.error("Failed to load questions at startup. Using fallback questions.")
    structure['school_based']['IIM'] = [
        "Why do you want to pursue an MBA from IIM specifically?",
        "What are your short-term and long-term career goals post-MBA?"
    ]

def generate_resume_questions(resume_text):
    if not resume_text:
        logging.warning("Empty resume text provided.")
        return ["Tell me about yourself."]
    
    prompt = f"Based on the following resume, generate 10 unique and relevant interview questions tailored to the candidate's experience and background:\n\n{resume_text}"
    try:
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        questions_text = response.choices[0].message.content
        questions = [q.strip() for q in questions_text.split('\n') if q.strip() and q not in asked_questions]
        logging.debug(f"Generated resume questions: {questions}")
        if not questions or len(questions) < 5:
            logging.warning("Insufficient or no valid questions generated from resume.")
            questions = [
                "Tell me about your most significant achievement in your career so far.",
                "What skills from your experience do you bring to an MBA program?",
                "Can you describe a challenge you faced in your last role?",
                "Why did you choose your current career path?",
                "How has your experience prepared you for an MBA?"
            ]
        return questions[:10]
    except Exception as e:
        logging.error(f"Error generating resume questions: {e}")
        return [
            "What motivated you to apply for this MBA?",
            "Can you walk me through your career journey?",
            "What’s one key lesson from your professional experience?"
        ]

def generate_follow_up_question(question, answer, attempt=1):
    if attempt > 3:
        return None
    prompt = f"Based on the following question and answer, generate a unique follow-up question if appropriate, or say 'No follow-up needed.'\n\nQuestion: {question}\nAnswer: {answer}"
    try:
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=50,
        )
        follow_up = response.choices[0].message.content.strip()
        if "no follow-up needed" in follow_up.lower() or follow_up in asked_questions:
            return generate_follow_up_question(question, answer, attempt + 1)
        logging.debug(f"Generated follow-up: {follow_up}")
        return follow_up
    except Exception as e:
        logging.error(f"Error generating follow-up question: {e}")
        return None

def generate_conversational_reply(answer):
    system_prompt = "As a friendly HR interviewer, generate a short, complete sentence as a reply to the candidate’s answer. Keep it engaging, human-like, and ensure it’s a full thought."
    try:
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": answer}],
            temperature=0.8,
            max_tokens=50,  # Increased to ensure complete sentences
        )
        reply = response.choices[0].message.content.strip()
        # Ensure the reply ends with proper punctuation if truncated
        if not reply.endswith(('.', '!', '?')):
            reply += '.'
        return reply
    except Exception:
        return "That’s a great response, thanks for sharing!"

def evaluate_response(question, answer, job_description):
    evaluation_prompt = f"Evaluate this for a {job_description} role: Q: {question} A: {answer}. Brief feedback, score out of 10 (e.g., 'Score: 8/10')."
    try:
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=0.5,
            max_tokens=50,
        )
        evaluation_text = response.choices[0].message.content
        score_line = [line for line in evaluation_text.split('\n') if "Score:" in line]
        score = int(score_line[0].split("Score: ")[1].split("/")[0]) if score_line else 5
        return evaluation_text, score
    except Exception:
        return "Good effort! Score: 5/10", 5

def is_interpersonal_question(question):
    return any(keyword in question.lower() for keyword in INTERPERSONAL_KEYWORDS)

def calculate_overall_score(evaluations, personal_count, technical_count):
    if not evaluations or (personal_count + technical_count == 0):
        return 0
    
    total_personal_score = 0
    total_technical_score = 0
    total_interpersonal_score = 0
    interpersonal_count = 0

    for eval in evaluations:
        category = eval["category"]
        score = eval["score"]
        if category == "personal":
            total_personal_score += score
        elif is_interpersonal_question(eval["question"]):
            total_interpersonal_score += score
            interpersonal_count += 1
        else:
            total_technical_score += score

    avg_personal = total_personal_score / personal_count if personal_count > 0 else 0
    avg_technical = total_technical_score / (technical_count - interpersonal_count) if technical_count > interpersonal_count else 0
    avg_interpersonal = total_interpersonal_score / interpersonal_count if interpersonal_count > 0 else 0

    if interpersonal_count > 0:
        overall_score = (INTERPERSONAL_TECHNICAL_WEIGHT * avg_technical +
                         INTERPERSONAL_WEIGHT * avg_interpersonal +
                         INTERPERSONAL_PERSONAL_WEIGHT * avg_personal) * 10
    else:
        overall_score = (DEFAULT_TECHNICAL_WEIGHT * avg_technical +
                         DEFAULT_PERSONAL_WEIGHT * avg_personal) * 10

    return round(overall_score, 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_interview', methods=['POST'])
def start_interview():
    global questions, current_question, evaluations, use_voice, asked_questions, resume_questions, interview_context
    
    language = request.form['language']
    mode = request.form['mode']
    interview_track = request.form['interview_track']
    sub_track = request.form.get('sub_track', '')
    use_voice = mode == 'voice'
    resume_file = request.files.get('resume')

    if not resume_file:
        return jsonify({"error": "Resume file is required"}), 400

    resume_path = os.path.join('uploads', resume_file.filename)
    resume_file.save(resume_path)
    if resume_path.lower().endswith('.pdf'):
        with pdfplumber.open(resume_path) as pdf:
            resume_text = ''.join(page.extract_text() or '' for page in pdf.pages)
    elif resume_path.lower().endswith('.docx'):
        resume_text = docx2txt.process(resume_path)
    else:
        os.remove(resume_path)
        return jsonify({"error": "Unsupported file format"}), 400
    os.remove(resume_path)
    logging.debug(f"Resume text extracted: {resume_text[:100]}...")

    questions = []
    current_question = 0
    evaluations = []
    asked_questions = set()
    resume_questions = generate_resume_questions(resume_text)
    logging.debug(f"Resume questions generated: {resume_questions}")
    asked_questions.update(resume_questions)

    if interview_track == "resume":
        predefined_questions = structure['resume_flow'][:3]
        questions = resume_questions + [q for q in predefined_questions if q not in resume_questions]
        logging.debug(f"Resume track questions: resume={resume_questions}, predefined={predefined_questions}, total={questions}")
    elif interview_track == "school_based":
        if sub_track in structure['school_based'] and structure['school_based'][sub_track]:
            questions = structure['school_based'][sub_track].copy()
            logging.debug(f"Selected school_based[{sub_track}]: {questions}")
        else:
            questions = [q for sublist in structure['school_based'].values() for q in sublist]
            logging.debug(f"Fallback to all school_based questions: {questions}")
    elif interview_track == "interest_areas":
        if sub_track in structure['interest_areas'] and structure['interest_areas'][sub_track]:
            questions = structure['interest_areas'][sub_track].copy()
            logging.debug(f"Selected interest_areas[{sub_track}]: {questions}")
        else:
            questions = [q for sublist in structure['interest_areas'].values() for q in sublist]
            logging.debug(f"Fallback to all interest_areas questions: {questions}")

    questions = [q for q in questions if q not in asked_questions]
    logging.debug(f"Questions after filtering: {questions}")
    asked_questions.update(questions)

    if not questions:
        logging.error(f"No questions available for track={interview_track}, sub_track={sub_track}")
        return jsonify({"error": f"No questions available for the selected track: {interview_track} - {sub_track}"}), 400
    
    interview_context.update({
        'questions': questions,
        'current_question_idx': 0,
        'previous_answers': [],
        'scores': [],
        'follow_up_depth': 0,
        'max_follow_ups': 2,
        'interview_track': interview_track,
        'sub_track': sub_track,
        'asked_questions': asked_questions
    })
    
    logging.info(f"Starting interview with {len(questions)} questions")
    return jsonify({
        "message": "Starting interview",
        "total_questions": len(questions),
        "current_question": questions[0],
        "question_number": 1,
        "use_voice": use_voice
    })

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    global current_question, evaluations, questions, asked_questions, interview_context
    
    answer = request.json.get('answer', "No response provided")
    main_question = questions[current_question]
    
    if main_question in resume_questions:
        category = "resume"
    elif main_question in structure['resume_flow']:
        category = "personal"
    else:
        category = "predefined"
    
    reply = generate_conversational_reply(answer)
    evaluation, score = evaluate_response(main_question, answer, job_description)
    
    evaluations.append({
        "question": main_question,
        "answer": answer,
        "evaluation": evaluation,
        "score": score,
        "category": category
    })
    interview_context["previous_answers"].append(answer)
    interview_context["scores"].append(score)
    interview_context["current_question_idx"] = current_question

    if interview_context["follow_up_depth"] < interview_context["max_follow_ups"]:
        follow_up = generate_follow_up_question(main_question, answer)
        if follow_up and follow_up not in asked_questions:
            questions.insert(current_question + 1, follow_up)
            asked_questions.add(follow_up)
            interview_context["follow_up_depth"] += 1
            current_question += 1
            logging.debug(f"Added follow-up question: {follow_up}")
            return jsonify({
                "reply": reply,
                "current_question": follow_up,
                "question_number": current_question + 1,
                "total_questions": len(questions),
                "next_question": True
            })
    
    current_question += 1
    if current_question < len(questions):
        logging.debug(f"Moving to next question: {questions[current_question]}")
        return jsonify({
            "reply": reply,
            "current_question": questions[current_question],
            "question_number": current_question + 1,
            "total_questions": len(questions),
            "next_question": True
        })
    else:
        personal_count = len([q for q in questions if q in structure['resume_flow']])
        technical_count = len(questions) - personal_count
        overall_score = calculate_overall_score(evaluations, personal_count, technical_count)
        logging.info(f"Interview finished. Overall score: {overall_score}")
        return jsonify({
            "reply": "Thanks for the chat! That’s all for today.",
            "finished": True,
            "evaluations": evaluations,
            "overall_score": overall_score
        })

if __name__ == "__main__":
    app.run(debug=True, port=5000)