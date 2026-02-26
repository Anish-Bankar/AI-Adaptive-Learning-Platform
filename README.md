# 🧠 AI Adaptive Learning Platform
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white)
![Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)

**Developed for Edunet Foundation & IBM SkillsBuild Internship**

An adaptive, AI-driven Learning Management System (LMS) built with Python, Streamlit, and MongoDB. It uses Google's Gemini 2.5 Flash and FAISS vector databases to convert any Syllabus, PYQ, or Research Paper into a highly personalized, hierarchical learning workspace.

Project Overview

Traditional study platforms rely on static quizzes and rigid pathways. The AI Adaptive Learning Platform is a dynamic, multi-compartment Learning Management System designed to bridge the gap between official curriculum and actual exam reality.

Powered by Google's Gemini API and FAISS vector search, this engine ingests raw PDF materials (Syllabi, PYQs, and Notes) to automatically construct a hierarchical knowledge map. It features an intelligent Gap Analysis Engine that audits past exam papers to inject "hidden" high-ROI topics into the student's timeline. By integrating MongoDB, the platform persistently tracks micro-level accuracy across specific subjects, utilizing active active recall and predictive burn-rate analytics to optimize a student's study trajectory.

### 🔥 Core Features
* **Dynamic Compartments:** Switch seamlessly between Exam Prep, Academic Research, or General Learning modes.
* **Auto-Hierarchy Extraction:** AI automatically builds a Subject/Topic tree directly from uploaded syllabus PDFs.
* **Gap Analysis:** AI audits Previous Year Questions (PYQs) to find "hidden" topics missing from the official syllabus.
* **Precision Quizzing:** Chain-of-Thought AI generates strict, context-bound MCQs with embedded solutions.
* **MongoDB Knowledge Tracing:** Persistent tracking of mastery, calculating pacing, burn-rates, and predictive timelines.

### ⚙️ Installation & Setup
1. Clone the repository: `git clone https://github.com/yourusername/your-repo-name.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure you have MongoDB running locally on port `27017`.
4. Create a `.env` file in the root directory and add your Gemini API Key: `GEMINI_API_KEY="your_key_here"`
5. Run the app: `streamlit run app.py`
