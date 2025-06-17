import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
import pdfplumber
import docx2txt
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def is_valid_resume(text):
    """Validate if the text content is actually a resume."""
    # Common resume sections and keywords
    resume_sections = [
        r'(?i)(education|academic|qualification)',
        r'(?i)(experience|work|employment|professional)',
        r'(?i)(skills|technical|competencies)',
        r'(?i)(projects|achievements|accomplishments)',
        r'(?i)(certifications|certificates)',
        r'(?i)(summary|profile|objective)'
    ]
    
    # Required minimum sections
    required_sections = 3
    
    # Check for personal information
    personal_info = [
        r'(?i)(email|e-mail)',
        r'(?i)(phone|mobile|contact)',
        r'(?i)(address|location)',
        r'(?i)(linkedin|github|portfolio)'
    ]
    
    # Count matching sections
    section_matches = sum(1 for section in resume_sections if re.search(section, text))
    personal_info_matches = sum(1 for info in personal_info if re.search(info, text))
    
    # Additional checks
    has_dates = bool(re.search(r'\b(19|20)\d{2}\b', text))  # Check for years
    has_bullet_points = bool(re.search(r'[•\-\*]', text))  # Check for bullet points
    
    # Calculate a score based on various factors
    score = 0
    score += section_matches * 2  # Each section is worth 2 points
    score += personal_info_matches  # Each personal info match is worth 1 point
    score += 2 if has_dates else 0  # Dates are worth 2 points
    score += 2 if has_bullet_points else 0  # Bullet points are worth 2 points
    
    # Minimum score required to be considered a resume
    min_score = 8
    
    # Get detailed validation results
    validation_details = {
        "sections_found": section_matches,
        "personal_info_found": personal_info_matches,
        "has_dates": has_dates,
        "has_bullet_points": has_bullet_points,
        "total_score": score,
        "is_valid": score >= min_score and section_matches >= required_sections
    }
    
    return validation_details

def extract_text_from_resume(file):
    """Extract text from uploaded resume file."""
    try:
        if file.name.endswith('.pdf'):
            with pdfplumber.open(file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                return text
        elif file.name.endswith('.docx'):
            return docx2txt.process(file)
        else:
            return None
    except Exception as e:
        logging.error(f"Error extracting text from resume: {e}")
        return None

def generate_resume_questions(resume_text, job_type):
    """Generate questions based on resume content."""
    try:
        # Create prompt for question generation
        prompt = f"""Based on the following resume, generate 10 relevant interview questions for a {job_type} position. 
        Focus on the candidate's experience, skills, and achievements. Make questions specific to their background.
        
        Resume:
        {resume_text}
        
        Generate 10 questions in the following format:
        1. [Question 1]
        2. [Question 2]
        3. [Question 3]
        4. [Question 4]
        5. [Question 5]
        .....
        """

        # Get response from OpenAI
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an expert interviewer specializing in resume-based questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content

    except Exception as e:
        logging.error(f"Error generating questions: {e}")
        return f"Error generating questions: {str(e)}"

def main():
    st.title("Resume-Based Question Generator")
    st.write("Upload a resume to generate interview questions for MBA or Banking positions.")

    # File upload
    uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=['pdf', 'docx'])

    # Job type selection
    job_type = st.radio(
        "Select Interview Track:",
        ["MBA", "Banking"],
        horizontal=True
    )

    if uploaded_file is not None:
        # Extract text from resume
        resume_text = extract_text_from_resume(uploaded_file)
        
        if resume_text:
            # Validate if it's actually a resume
            validation_results = is_valid_resume(resume_text)
            
            if validation_results["is_valid"]:
                st.success("✅ Valid resume detected!")
                
                # Show validation details in expander
                with st.expander("Resume Validation Details"):
                    st.write(f"• Sections found: {validation_results['sections_found']}")
                    st.write(f"• Personal information found: {validation_results['personal_info_found']}")
                    st.write(f"• Contains dates: {'Yes' if validation_results['has_dates'] else 'No'}")
                    st.write(f"• Contains bullet points: {'Yes' if validation_results['has_bullet_points'] else 'No'}")
                    st.write(f"• Validation score: {validation_results['total_score']}/20")
                
                # Show extracted text in expander
                with st.expander("View Extracted Resume Text"):
                    st.text(resume_text)
                
                # Generate questions button
                if st.button("Generate Questions"):
                    with st.spinner("Generating questions..."):
                        questions = generate_resume_questions(resume_text, job_type)
                        
                        # Display questions
                        st.subheader("Generated Questions")
                        st.write(questions)
                        
                        # Add download button for questions
                        st.download_button(
                            label="Download Questions",
                            data=questions,
                            file_name=f"{job_type}_interview_questions.txt",
                            mime="text/plain"
                        )
            else:
                st.error("❌ This doesn't appear to be a valid resume. Please upload a proper resume with sections like Education, Experience, Skills, etc.")
                with st.expander("Validation Details"):
                    st.write(f"• Sections found: {validation_results['sections_found']}")
                    st.write(f"• Personal information found: {validation_results['personal_info_found']}")
                    st.write(f"• Contains dates: {'Yes' if validation_results['has_dates'] else 'No'}")
                    st.write(f"• Contains bullet points: {'Yes' if validation_results['has_bullet_points'] else 'No'}")
                    st.write(f"• Validation score: {validation_results['total_score']}/20")
                    st.write("\nA valid resume should have:")
                    st.write("• At least 3 main sections (Education, Experience, Skills, etc.)")
                    st.write("• Personal contact information")
                    st.write("• Dates of education/experience")
                    st.write("• Proper formatting with bullet points")
        else:
            st.error("Failed to process resume. Please try again with a different file.")

if __name__ == "__main__":
    main()
