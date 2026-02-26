# 🧠 AI Adaptive Learning Platform
**Developed for Edunet Foundation & IBM SkillsBuild Internship**

An adaptive, AI-driven Learning Management System (LMS) built with Python, Streamlit, and MongoDB. It uses Google's Gemini 2.5 Flash and FAISS vector databases to convert any Syllabus, PYQ, or Research Paper into a highly personalized, hierarchical learning workspace.

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
5. Run the app: `streamlit run app1.py`