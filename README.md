---

# 🎉 HR Interviewer: Your AI-Powered MBA Prep Buddy! 🚀

Welcome to **HR Interviewer**—a dazzling, AI-driven web app that turns interview prep into an exciting adventure! Whether you're aiming for an MBA at IIM, ISB, or beyond, this tool crafts personalized questions, chats with you in text or voice, and dishes out scores and feedback faster than you can say "hire me!" Powered by Deepseek r1, it’s your ticket to acing that interview with flair! 🌟

---

## ✨ Killer Features That’ll Wow You

- 🚀 **Resume Magic:** Drop your PDF/DOCX resume, and watch it spin out questions tailored just for you!
- 🎓 **School Vibes:** Pick your B-School track—IIM, ISB, or others—and dive into custom question sets.
- 🔥 **Interest Zones:** Choose your passion—Leadership, Finance, Marketing, or Operations—and get questions that match your groove.
- 🎙️ **Voice Party:** Talk your answers out loud with slick speech-to-text action.
- ✍️ **Text Jam:** Prefer typing? Nail your responses the classic way.
- 🤖 **AI Chat Star:** Grok brings the convo to life with witty replies, follow-ups, and spot-on evaluations.
- ⭐ **Scoreboard Glory:** Get instant feedback and scores (out of 10) per answer, plus a dazzling overall score (out of 100)!
- 🌍 **Language Flex:** Chat in English (India, US, UK)—your call!
- 🎨 **Sleek Looks:** A vibrant, smooth UI that’s as fun to use as it is to look at.

---

## 🛠️ Pre-Install Goodies You’ll Need

- **Python 3.11+**: The engine under the hood—get it revving!
- **Git**: Clone this baby like a pro.
- **MBA_Question.pdf**: Our question bank—comes with the repo or whip up your own!

---

## 🌐 Live Demo—Check It Out!

Hop over to: [https://ai-interview-hce4.onrender.com](https://ai-interview-hce4.onrender.com)  
*Psst:* It’s on Render’s free tier, so it might snooze after 15 mins—give it a sec to wake up! 😴➡️🚀

---

## ⚡ Installation: Get It Poppin’ in Minutes!

1. **Grab the Code:**
   ```bash
   git clone https://github.com/avinashhmavi/hr-interviewer.git
   cd hr-interviewer
   ```

2. **Set Up Your Space (Optional but Cool):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Load the Party Supplies:**
   Save this to `requirements.txt`:
   ```
   Flask==2.3.3
   llamaapi
   pdfplumber==0.11.4
   python-docx==1.1.2
   python-dotenv==1.0.1
   docx2txt
   ```
   Then run:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set the Secret Sauce:**
   Create a `.env` file:
   ```bash
   echo "Llama_API_KEY=your-llama-api-key" > .env
   ```
   Swap in your llama-API key—keep it hush-hush! 🤫

5. **Drop the Question Bank:**
   - Make sure `MBA_Question.pdf` is in the root folder. It’s included, or craft your own with sections like:
     ```
     1. Resume Flow
     1. Can you walk us through your resume?
     2. Pre-Defined Question Selection
     For IIMs
     1. Why IIM?
     ```

---

## 🏃‍♂️ Running Locally: Let’s Roll!

1. **Fire It Up:**
   ```bash
   python main.py
   ```

2. **Jump In:**
   Hit `http://localhost:5000` in your browser—boom, you’re live!

3. **Play Around:**
   - Upload your resume.
   - Pick a track (Resume, School, Interests).
   - Go text or voice—your stage, your rules!

---
