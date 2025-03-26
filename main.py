import os
import json
import time
from flask import Flask, render_template, request, jsonify
from llamaapi import LlamaAPI
import pdfplumber
import docx2txt
from dotenv import load_dotenv
from collections import defaultdict
import logging
import re

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, template_folder='.', static_folder='.')
os.makedirs('uploads', exist_ok=True)

# Initialize LlamaAPI client
try:
    client = LlamaAPI("c5f4d16a-62a5-407e-b854-3cea14e3891a")
    logging.info("LlamaAPI client initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize LlamaAPI client: {e}")
    client = None

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

def normalize_text(text):
    """Normalize text by removing extra whitespace and converting to lowercase."""
    return " ".join(text.strip().split()).lower()

def strip_numbering(text):
    """Remove leading numbers (e.g., '1. ') from text."""
    return re.sub(r'^\d+\.\s*', '', text).strip()

def load_questions_into_memory():
    """Load predefined questions from a PDF file into memory."""
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
                question = strip_numbering(line)
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

# Load questions at startup and provide fallback if loading fails
if not load_questions_into_memory():
    logging.error("Failed to load questions at startup. Using fallback questions.")
    structure['school_based']['IIM'] = [
        "Why do you want to pursue an MBA from IIM specifically?",
        "What are your short-term and long-term career goals post-MBA?",
        "How does IIM’s curriculum align with your career aspirations?",
        "How do you plan to contribute to the peer-learning culture at IIM?",
        "Which specialization are you interested in, and why?"
    ]

def generate_resume_questions(resume_text):
    """Generate interview questions based on resume text."""
    if not resume_text:
        logging.warning("Empty resume text provided.")
        return ["Tell me about yourself."]
    
    prompt = f"Based on the following resume, generate 10 unique and relevant interview questions tailored to the candidate's experience and background (do not include numbers in questions):\n\n{resume_text}"
    try:
        api_request_json = {
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 500,
            "stream": False
        }
        response = client.run(api_request_json)
        response_json = response.json()
        questions_text = response_json['choices'][0]['message']['content']
        questions = [strip_numbering(q.strip()) for q in questions_text.split('\n') if q.strip() and q not in asked_questions]
        logging.debug(f"Generated resume questions: {questions}")
        if not questions or len(questions) < 5:
            logging.warning("Insufficient or no valid questions generated from resume.")
            questions = [
                "Tell me about your most significant achievement in your career so far",
                "What skills from your experience do you bring to an MBA program",
                "Can you describe a challenge you faced in your last role",
                "Why did you choose your current career path",
                "How has your experience prepared you for an MBA"
            ]
        return questions[:10]
    except Exception as e:
        logging.error(f"Error generating resume questions: {e}")
        return [
            "What motivated you to apply for this MBA",
            "Can you walk me through your career journey",
            "What’s one key lesson from your professional experience"
        ]

def evaluate_response(question, answer, job_description):
    """Evaluate the candidate's answer using the LlamaAPI model with a fallback."""
    # Fallback evaluation if LlamaAPI fails
    def fallback_evaluation(question, answer):
        answer = answer.lower().strip()
        if len(answer) < 5 or not any(c.isalpha() for c in answer):
            return "[Answer is irrelevant or gibberish] Score: 0/10", 0
        
        question_keywords = set(normalize_text(question).split())
        answer_keywords = set(normalize_text(answer).split())
        common_keywords = question_keywords.intersection(answer_keywords)
        
        if not common_keywords:
            return "[Answer is irrelevant to the question] Score: 0/10", 0
        
        score = min(10, max(3, len(answer.split()) // 5))  # Scale score based on length, min 3 for relevance
        feedback = "[Answer is relevant but could use more detail]" if score < 7 else "[Answer is relevant and detailed]"
        return f"{feedback} Score: {score}/10", score

    if client:
        evaluation_prompt = f"""Evaluate the following answer for the question in the context of a {job_description} role. Assess relevance, depth, and insightfulness.

Question: {question}
Answer: {answer}

Provide feedback and a score out of 10:
- 0: Completely irrelevant, gibberish, or no answer.
- 0-1: Barely relevant, lacks substance.
- 2-6: Somewhat relevant, basic understanding, limited detail.
- 7-8: Relevant, good understanding, decent detail.
- 9-10: Highly relevant, detailed, insightful.

Ensure the score reflects the answer's quality relative to the question. Format: '[Feedback] Score: X/10'"""
        try:
            api_request_json = {
                "model": "deepseek-r1",
                "messages": [{"role": "user", "content": evaluation_prompt}],
                "temperature": 0.5,
                "max_tokens": 150,
                "stream": False
            }
            response = client.run(api_request_json)
            evaluation_text = response.json()['choices'][0]['message']['content'].strip()
            logging.debug(f"Evaluation text: {evaluation_text}")
            
            score_match = re.search(r'Score:\s*(\d+)/10', evaluation_text)
            score = int(score_match.group(1)) if score_match else 5  # Default to 5 if parsing fails
            return evaluation_text, score
        except Exception as e:
            logging.error(f"Error in LlamaAPI evaluation: {e}")
            return fallback_evaluation(question, answer)
    else:
        logging.warning("LlamaAPI client not initialized. Using fallback evaluation.")
        return fallback_evaluation(question, answer)

def generate_follow_up_question(question, answer, score, attempt=1):
    """Generate a follow-up question based on the answer and its score, ensuring at least one follow-up."""
    if attempt > 2:  # Limit to 2 attempts max
        return None
    
    # Force a follow-up on the first attempt, regardless of score
    if attempt == 2:
        prompt = f"""Given the question and answer below for an MBA candidate interview, with a score of {score}/10, generate a relevant follow-up question to probe further or encourage elaboration. Even if the answer is poor or irrelevant, craft a question that steers the candidate back to the interview context (e.g., MBA goals, experience, or skills).

Question: {question}
Answer: {answer}
Score: {score}/10"""
    else:
        # For subsequent attempts, only generate if score < 7
        if score >= 7:
            return None
        prompt = f"""Given the question and answer below for an MBA candidate interview, with a score of {score}/10, generate a follow-up question to encourage elaboration or clarification if the answer is relevant but lacks detail or depth.

Question: {question}
Answer: {answer}
Score: {score}/10"""

    try:
        api_request_json = {
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 50,
            "stream": False
        }
        response = client.run(api_request_json)
        follow_up = response.json()['choices'][0]['message']['content'].strip()
        if "no follow-up needed" in follow_up.lower() or follow_up in asked_questions:
            # Fallback for first attempt if LLM fails to generate a valid question
            if attempt == 1:
                return "Can you tell me more about how this relates to your MBA goals?"
            return None
        follow_up = strip_numbering(follow_up)
        logging.debug(f"Generated follow-up: {follow_up}")
        return follow_up
    except Exception as e:
        logging.error(f"Error generating follow-up question: {e}")
        # Fallback for first attempt if API fails
        if attempt == 1:
            return "Can you explain how this connects to your career aspirations?"
        return None
def generate_conversational_reply(answer):
    """Generate a friendly, human-like reply to the candidate's answer."""
    system_prompt = "As a friendly HR interviewer, generate a short, complete sentence as a reply to the candidate’s answer. Keep it engaging and human-like,and ensure it's a full thought."
    try:
        api_request_json = {
            "model": "deepseek-r1",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": answer}
            ],
            "temperature": 0.8,
            "max_tokens": 50,
            "stream": False
        }
        response = client.run(api_request_json)
        reply = response.json()['choices'][0]['message']['content'].strip()
        if not reply.endswith(('.', '!', '?')):
            reply += '.'
        return reply
    except Exception as e:
        logging.error(f"Error generating reply: {e}")
        return "Thanks for your response."

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    """Handle submission of an answer and proceed with the interview."""
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

    # Follow-up logic: ask follow-ups for relevant answers (score > 1) up to max_follow_ups
    if score > 1 and interview_context["follow_up_depth"] < interview_context["max_follow_ups"]:
        follow_up = generate_follow_up_question(main_question, answer, score)
        if not follow_up and score < 7:  # Fallback for relevant but low-scoring answers
            follow_up = "Can you provide more details on that?"
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
    
    # Move to next question if no follow-up or max follow-ups reached
    interview_context["follow_up_depth"] = 0
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

def calculate_overall_score(evaluations, personal_count, technical_count):
    """Calculate the overall score based on evaluations."""
    if not evaluations or (personal_count + technical_count == 0):
        return 0
    total_score = sum(e["score"] for e in evaluations)
    return round((total_score / (len(evaluations) * 10)) * 100, 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_interview', methods=['POST'])
def start_interview():
    global questions, current_question, evaluations, use_voice, asked_questions, resume_questions, interview_context
    
    # Extract request data
    language = request.form['language']
    mode = request.form['mode']
    interview_track = request.form['interview_track']
    sub_track = request.form.get('sub_track', '')
    use_voice = mode == 'voice'
    resume_file = request.files.get('resume') if interview_track == 'resume' else None

    # Handle resume file for resume track
    resume_text = ""
    if interview_track == 'resume':
        if not resume_file:
            return jsonify({"error": "Resume file is required for resume-based track"}), 400
        
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

    # Initialize variables
    questions = []
    current_question = 0
    evaluations = []
    asked_questions = set()
    
    # Populate questions based on interview track
    if interview_track == "resume":
        resume_questions = generate_resume_questions(resume_text)
        predefined_questions = structure['resume_flow'][:3]
        questions = resume_questions + [q for q in predefined_questions if q not in resume_questions]
        asked_questions.update(resume_questions)
        logging.debug(f"Resume track questions: resume={resume_questions}, predefined={predefined_questions}, total={questions}")
    elif interview_track == "school_based":
        if sub_track in structure['school_based'] and structure['school_based'][sub_track]:
            questions = structure['school_based'][sub_track].copy()
        else:
            questions = [q for sublist in structure['school_based'].values() for q in sublist]
        logging.debug(f"School-based questions: {questions}")
    elif interview_track == "interest_areas":
        if sub_track in structure['interest_areas'] and structure['interest_areas'][sub_track]:
            questions = structure['interest_areas'][sub_track].copy()
        else:
            questions = [q for sublist in structure['interest_areas'].values() for q in sublist]
        logging.debug(f"Interest areas questions: {questions}")

    # Clean up questions and update asked set
    questions = [strip_numbering(q) for q in questions if q not in asked_questions]
    asked_questions.update(questions)
    logging.debug(f"Questions after filtering: {questions}")

    # Check if questions list is empty
    if not questions:
        logging.error(f"No questions available for track={interview_track}, sub_track={sub_track}")
        return jsonify({"error": f"No questions available for the selected track: {interview_track} - {sub_track}"}), 400
    
    # Update interview context
    interview_context.update({
        'questions': questions,
        'current_question_idx': 0,
        'previous_answers': [],
        'scores': [],
        'follow_up_depth': 1,
        'max_follow_ups': 2,
        'interview_track': interview_track,
        'sub_track': sub_track,
        'asked_questions': asked_questions
    })
    
    # Return successful response
    logging.info(f"Starting interview with {len(questions)} questions")
    return jsonify({
        "message": "Starting interview",
        "total_questions": len(questions),
        "current_question": questions[0],
        "question_number": 1,
        "use_voice": use_voice
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's PORT or default to 5000 locally
    app.run(host="0.0.0.0", port=port, debug=False)  # Bind to 0.0.0.0, disable debug in production
