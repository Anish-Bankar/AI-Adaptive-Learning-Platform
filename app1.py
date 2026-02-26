import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import time
import json
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime, date, timedelta

# --- 1. CONFIG & CREDENTIALS ---
st.set_page_config(page_title="Universal Knowledge Engine", layout="wide", page_icon="🧠")

load_dotenv() # This loads the variables from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# --- 2. MONGODB DATABASE SETUP ---
MONGO_URI = "mongodb://localhost:27017/"


@st.cache_resource
def init_db():
    try:
        client = MongoClient(MONGO_URI)
        return client['adaptive_lms_db']
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None


db = init_db()
CURRENT_USER_ID = "student_001"


# --- 3. HELPER FUNCTIONS ---
def log_quiz_attempt(mode, category, parent_node, child_node, q_type, difficulty, is_correct, time_taken):
    if db is not None:
        db.quiz_history.insert_one({
            "user_id": CURRENT_USER_ID,
            "purpose_mode": mode,
            "category": category,
            "parent_node": parent_node,
            "child_node": child_node,
            "question_type": q_type,
            "difficulty": difficulty,
            "is_correct": is_correct,
            "time_taken_sec": time_taken,
            "timestamp": datetime.now()
        })


def fetch_user_history(mode=None, category=None, parent_node=None):
    if db is None: return pd.DataFrame()
    query = {"user_id": CURRENT_USER_ID}
    if mode: query["purpose_mode"] = mode
    if category: query["category"] = category
    if parent_node and parent_node != "All": query["parent_node"] = parent_node
    return pd.DataFrame(list(db.quiz_history.find(query)))


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


hf_model = load_embedding_model()


def extract_full_text(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    return "".join([page.extract_text() or "" for page in reader.pages])


def process_pdf_for_rag(uploaded_file, doc_type):
    raw_text = extract_full_text(uploaded_file).replace('\n\n', '\n').strip()
    chunks = [raw_text[i:i + 1200] for i in range(0, len(raw_text), 1000)]
    return [{"text": chunk, "type": doc_type} for chunk in chunks if len(chunk.strip()) > 50]


def extract_document_hierarchy(text, mode, category):
    if mode == "Exam Preparation":
        sys_prompt = f"Analyze this {category} syllabus. Extract the main Subjects (Keys) and specific testable Topics (Values in array)."
    elif mode == "Academic Research":
        sys_prompt = f"Analyze these research documents for the field of {category}. Extract the main Research Domains/Papers (Keys) and the specific Methodologies/Themes/Findings (Values in array)."
    else:
        sys_prompt = f"Analyze this learning material for {category}. Extract the Broad Modules (Keys) and specific Sub-topics (Values in array)."

    prompt = f"""You are an advanced data extraction engine.
    {sys_prompt}
    Identify the underlying hierarchical pattern regardless of formatting.
    Return ONLY a JSON object matching this schema:
    {{"Parent Name 1": ["Child 1", "Child 2"], "Parent Name 2": ["Child A", "Child B"]}}

    TEXT: {text[:40000]}"""

    try:
        model = genai.GenerativeModel('gemini-2.5-flash',
                                      generation_config={"temperature": 0.0, "response_mime_type": "application/json"})
        return json.loads(model.generate_content(prompt).text.strip())
    except:
        return {}

    # --- NEW: GAP ANALYSIS AI LAYER (SYLLABUS vs PYQs) ---


def augment_hierarchy_with_pyqs(pyq_text, current_structure, category):
    """Checks PYQs for 'hidden' topics not in the official syllabus and appends them."""
    prompt = f"""You are an expert curriculum auditor for {category}.
    Here is the officially extracted syllabus hierarchy:
    {json.dumps(current_structure)}

    Here is the text extracted from actual Previous Year Questions (PYQs):
    {pyq_text[:60000]}

    TASK: Perform a Gap Analysis. Identify recurring topics, sections, or concepts in the PYQs that are MISSING from the official syllabus hierarchy (e.g., General Aptitude, logical reasoning, or unlisted sub-topics).
    1. If a missing topic belongs to an existing Parent/Subject, append it to that array.
    2. If a missing topic belongs to a completely new section (like 'General Aptitude'), create a new Parent key and add the topics.
    3. Do NOT remove any existing topics. Only ADD to the JSON.

    Return ONLY the COMPLETE, AUGMENTED JSON hierarchy.
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash',
                                      generation_config={"temperature": 0.1, "response_mime_type": "application/json"})
        return json.loads(model.generate_content(prompt).text.strip())
    except Exception as e:
        st.warning(f"Gap Analysis minor error: {e}. Falling back to standard syllabus.")
        return current_structure


def get_active_context(query, compartment_key, k=8, filter_type=None):
    active_profile = st.session_state.compartments.get(compartment_key)
    if not active_profile or active_profile['faiss'] is None or not active_profile['chunks']: return ""
    query_emb = hf_model.encode([query])
    faiss.normalize_L2(query_emb)
    D, I = active_profile['faiss'].search(query_emb, k * 2)
    results = [active_profile['chunks'][idx]['text'] for idx in I[0] if idx < len(active_profile['chunks']) and (
                filter_type is None or active_profile['chunks'][idx]['type'] == filter_type)]
    return "\n\n".join(results[:k])


# --- 4. STATE MANAGEMENT ---
if 'compartments' not in st.session_state: st.session_state.compartments = {}
if 'quiz_active' not in st.session_state: st.session_state.quiz_active = False
if 'chat_history' not in st.session_state: st.session_state.chat_history = []

# --- 5. DYNAMIC UI CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Workspace Setup")
    app_mode = st.radio("Primary Purpose:", ["Exam Preparation", "Academic Research", "General Learning"])

    if app_mode == "Exam Preparation":
        ui = {"cat": "Exam Type", "parent": "Subject", "child": "Topic", "f1": "1. Upload Syllabus",
              "f2": "2. Upload PYQs", "f3": "3. Upload Notes", "tab1": "🗺️ Success Path", "tab2": "🎯 Quiz Drill"}
        cat_options = ["University (IT Engineering)", "GATE (CS & DA)", "NPTEL", "Other Exams"]
    elif app_mode == "Academic Research":
        ui = {"cat": "Research Field", "parent": "Domain / Paper", "child": "Key Theme", "f1": "1. Upload Core Papers",
              "f2": "2. Upload Lit Reviews", "f3": "3. Datasets / Notes", "tab1": "🗺️ Literature Map",
              "tab2": "🎯 Comprehension Check"}
        cat_options = ["Computer Science", "Engineering", "Sciences", "Humanities", "Other Research"]
    else:
        ui = {"cat": "Category", "parent": "Module", "child": "Sub-topic", "f1": "1. Upload Curriculum / Index",
              "f2": "2. Upload Books / Articles", "f3": "3. Upload Notes", "tab1": "🗺️ Learning Roadmap",
              "tab2": "🎯 Knowledge Check"}
        cat_options = ["Programming", "Languages", "Business", "Hobbies", "Other"]

    st.divider()
    st.header(f"🎯 Active {ui['cat']}")
    active_cat = st.selectbox(f"Select {ui['cat']}:", cat_options)

    comp_key = f"{app_mode} - {active_cat}"
    if comp_key not in st.session_state.compartments:
        st.session_state.compartments[comp_key] = {"faiss": None, "chunks": [], "primary_doc": "", "structure": {},
                                                   "start_date": date.today(), "target_date": None}

    if app_mode == "Exam Preparation":
        st.markdown("### ⏳ Timeline")
        t_date = st.date_input("Target/Exam Date:", min_value=date.today(),
                               value=st.session_state.compartments[comp_key].get('target_date',
                                                                                 date.today() + timedelta(days=30)))
        st.session_state.compartments[comp_key]['target_date'] = t_date

    st.divider()
    active_structure = st.session_state.compartments[comp_key].get('structure', {})
    if active_structure:
        current_parent = st.selectbox(f"Active {ui['parent']}:", list(active_structure.keys()))
    else:
        current_parent = st.text_input(f"Active {ui['parent']}:", placeholder="Upload docs to auto-populate")
        if not current_parent: current_parent = "General"

    st.divider()
    st.header("📚 Add Materials")
    file_1 = st.file_uploader(ui['f1'], type=["pdf"])
    file_2 = st.file_uploader(ui['f2'], type=["pdf"], accept_multiple_files=True)
    file_3 = st.file_uploader(ui['f3'], type=["pdf"])

    if st.button("Process & Build Workspace", type="primary", use_container_width=True):
        with st.spinner(f"Building {app_mode} workspace for {active_cat}..."):
            all_chunks = st.session_state.compartments[comp_key]['chunks']

            # STEP 1: Process Primary Syllabus
            if file_1:
                full_text = extract_full_text(file_1)
                st.session_state.compartments[comp_key]['primary_doc'] = full_text
                with st.spinner("Extracting Official Knowledge Hierarchy..."):
                    structure = extract_document_hierarchy(full_text, app_mode, active_cat)
                    if structure:
                        st.session_state.compartments[comp_key]['structure'] = structure

            # STEP 2: Process PYQs & GAP ANALYSIS
            pyq_combined_text = ""
            if file_2:
                for f in file_2:
                    # Extract text for Gap Analysis
                    extracted = extract_full_text(f)
                    pyq_combined_text += extracted + "\n"
                    # Process chunks for RAG
                    all_chunks.extend(process_pdf_for_rag(f, "Secondary"))

                # TRIGGER GAP ANALYSIS if we have both a syllabus and PYQs
                if 'structure' in st.session_state.compartments[comp_key] and st.session_state.compartments[comp_key][
                    'structure']:
                    with st.spinner("Auditing PYQs for hidden topics (Gap Analysis)..."):
                        updated_structure = augment_hierarchy_with_pyqs(pyq_combined_text,
                                                                        st.session_state.compartments[comp_key][
                                                                            'structure'], active_cat)
                        st.session_state.compartments[comp_key]['structure'] = updated_structure

            if file_3:
                all_chunks.extend(process_pdf_for_rag(file_3, "Tertiary"))

            if all_chunks:
                embeddings = hf_model.encode([c["text"] for c in all_chunks])
                faiss.normalize_L2(embeddings)
                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings.astype(np.float32))
                st.session_state.compartments[comp_key]['faiss'] = index
                st.session_state.compartments[comp_key]['chunks'] = all_chunks

        st.success("Workspace Ready! Hierarchy Augmented.")
        time.sleep(0.5)
        st.rerun()

# --- 6. MAIN UI TABS ---
st.title("🧠 Universal Knowledge Engine")
st.caption(f"**Mode:** {app_mode} | **Category:** {active_cat} | **{ui['parent']}:** {current_parent}")

if app_mode == "Exam Preparation":
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [ui['tab1'], ui['tab2'], "🛠️ Deep Eval", "📊 Hierarchy", "⏳ Timeline & Pacing"])
else:
    tab1, tab2, tab3, tab4 = st.tabs([ui['tab1'], ui['tab2'], "🛠️ Deep Eval", "📊 Hierarchy"])

# TAB 1: GRAPHICAL ROADMAP & CHAT
with tab1:
    col1, col2 = st.columns([1.2, 1], gap="large")
    with col1:
        st.subheader(ui['tab1'].replace("🗺️ ", ""))
        if st.button(f"Generate Graphical {ui['tab1'].replace('🗺️ ', '')}", use_container_width=True):
            with st.spinner("Analyzing materials to construct milestones..."):
                sec_context = get_active_context("key findings summary weightage", comp_key, k=10,
                                                 filter_type="Secondary")
                model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"temperature": 0.2,
                                                                                     "response_mime_type": "application/json"})

                prompt = f"""You are an expert project manager for {app_mode}.
                Based on the primary document, build a structured, step-by-step roadmap.
                PRIMARY: {st.session_state.compartments[comp_key]['primary_doc'][:80000]}
                SECONDARY: {sec_context}

                CRITICAL INSTRUCTION: Return a JSON array of 'milestones'. Format exactly like this:
                [
                  {{"phase": "Phase 1: Foundation", "description": "Brief explanation", "tasks": ["Task A", "Task B"]}}
                ]"""

                try:
                    st.session_state.roadmap_data = json.loads(model.generate_content(prompt).text.strip())
                except Exception as e:
                    st.error("Failed to map roadmap format. Try again.")

        if 'roadmap_data' in st.session_state:
            st.write("### Your Target Milestones")
            for idx, ms in enumerate(st.session_state.roadmap_data):
                with st.container(border=True):
                    st.markdown(f"#### 🚩 {ms.get('phase', f'Milestone {idx + 1}')}")
                    st.info(ms.get('description', ''))
                    for task in ms.get('tasks', []):
                        st.checkbox(task, key=f"ms_{idx}_{task}")

    with col2:
        st.subheader("Workspace Chat")
        with st.container(border=True, height=550):
            for msg in st.session_state.chat_history:
                st.chat_message(msg["role"]).write(msg["content"])

        user_query = st.chat_input("Ask a question about your documents...")
        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            st.rerun()

        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            with st.spinner("Searching..."):
                latest_q = st.session_state.chat_history[-1]["content"]
                context = get_active_context(latest_q, comp_key)
                primary = st.session_state.compartments[comp_key]['primary_doc']
                prompt = f"You are an assistant for {app_mode}. PRIMARY DOC: {primary[:15000]}\nCONTEXT: {context}\nQUESTION: {latest_q}"
                resp = genai.GenerativeModel('gemini-2.5-flash').generate_content(prompt).text
                st.session_state.chat_history.append({"role": "assistant", "content": resp})
                st.rerun()

# TAB 2: PRECISION QUIZ / COMPREHENSION
with tab2:
    st.header(ui['tab2'].replace("🎯 ", ""))

    if not st.session_state.quiz_active:
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(4)
            active_structure = st.session_state.compartments[comp_key].get('structure', {})
            if active_structure and current_parent in active_structure:
                child_options = active_structure[current_parent]
                quiz_topic = c1.selectbox(f"{ui['child']}:", child_options)
            else:
                quiz_topic = c1.text_input(f"{ui['child']}:", placeholder="Enter specific topic/theme")

            num_q = c2.selectbox("Questions:", [3, 5, 10])
            q_type = c3.selectbox("Type:", ["Mixed", "Theoretical", "Application / Methodology"])
            c_diff = c4.radio("Difficulty:", ["Easy", "Medium", "Hard Level"], horizontal=True)

            st.write("")
            if st.button("🚀 Generate Drill", type="primary", use_container_width=True) and quiz_topic:
                st.session_state.quiz_topic = quiz_topic
                st.session_state.q_type_selected = q_type
                st.session_state.diff_selected = c_diff

                with st.spinner(f"Generating strict questions exclusively for '{quiz_topic}'..."):
                    context = get_active_context(quiz_topic, comp_key, k=15)
                    model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"temperature": 0.1,
                                                                                         "response_mime_type": "application/json"})
                    prompt = f"""Mode: {app_mode} | Category: {active_cat}
                    {ui['parent']}: {current_parent} | {ui['child']}: {quiz_topic}
                    Context: {context[:4000]}
                    CRITICAL INSTRUCTIONS: ALL questions MUST be exclusively about '{quiz_topic}'.
                    Return exactly {num_q} questions matching this JSON schema:
                    [{{ "q": "Question?", "opts": ["A) 1", "B) 2", "C) 3", "D) 4"], "sol": "Explanation", "ans": "A" }}]"""

                    try:
                        st.session_state.quiz_data = json.loads(model.generate_content(prompt).text.strip())
                        st.session_state.quiz_active = True
                        st.session_state.start_time = time.time()
                        st.rerun()
                    except Exception as e:
                        st.error("Failed to generate valid JSON format. Adjust parameters and retry.")

    if st.session_state.quiz_active:
        st.info(f"Active Drill: **{st.session_state.quiz_topic}**")

        user_answers = {}
        for i, q in enumerate(st.session_state.quiz_data):
            with st.container(border=True):
                st.write(f"**Q{i + 1}: {q['q']}**")
                options = [opt[0] for opt in q['opts']]
                user_answers[i] = st.radio("Select:", options, key=f"q_{i}", index=None, horizontal=True,
                                           format_func=lambda x: q['opts'][options.index(x)])

        colA, colB = st.columns(2)
        if colA.button("✅ Submit & Review", type="primary", use_container_width=True):
            avg_time = round((time.time() - st.session_state.start_time) / len(st.session_state.quiz_data), 1)
            correct_count = sum(
                [1 for i, q in enumerate(st.session_state.quiz_data) if user_answers.get(i) == q['ans']])

            for i, q in enumerate(st.session_state.quiz_data):
                log_quiz_attempt(app_mode, active_cat, current_parent, st.session_state.quiz_topic,
                                 st.session_state.q_type_selected, st.session_state.diff_selected,
                                 user_answers.get(i) == q['ans'], avg_time)

            st.success(f"Score: {(correct_count / len(st.session_state.quiz_data)) * 100}% | Avg Time: {avg_time}s")

            st.markdown("### 📖 Solutions")
            for i, q in enumerate(st.session_state.quiz_data):
                is_wrong = (user_answers.get(i) != q['ans'])
                with st.expander(f"Q{i + 1}: {q['q'][:60]}...", expanded=is_wrong):
                    if is_wrong:
                        st.error(f"Your answer: {user_answers.get(i)} | Correct: {q['ans']}")
                    else:
                        st.success(f"Correct: {q['ans']}")
                    st.info(f"**Explanation:** {q.get('sol', '')}")

        if colB.button("❌ Cancel Drill", use_container_width=True):
            st.session_state.quiz_active = False
            st.rerun()

# TAB 3: DEEP EVALUATION
with tab3:
    st.header("🛠️ Knowledge Gaps & Deep Dive")
    df = fetch_user_history(mode=app_mode, category=active_cat, parent_node=current_parent)
    if not df.empty:
        stats = df.groupby('child_node')['is_correct'].agg(['mean', 'count']).reset_index()
        struggling = stats[stats['mean'] < 0.6]['child_node'].tolist()

        st.progress(len(stats[stats['mean'] >= 0.8]) / max(1, len(stats)),
                    text=f"Mastery Progress across {ui['child']}s.")

        if struggling:
            st.error(f"🚨 Requires Review: {', '.join(struggling)}")
            target = st.selectbox(f"Select a {ui['child']} to deeply understand:", struggling)
            if st.button("Generate Deep Dive Guide", type="primary"):
                context = get_active_context(target, comp_key, k=15)
                prompt = f"Explain '{target}' deeply based on this context: {context[:5000]}. Mode is {app_mode}."
                with st.container(border=True):
                    st.write(genai.GenerativeModel('gemini-2.5-flash').generate_content(prompt).text)
        else:
            st.success("Solid mastery in this section!")
    else:
        st.info("Complete drills to unlock insights.")

# TAB 4: HIERARCHY
with tab4:
    st.header(f"📊 Hierarchical {app_mode} Tracker")
    df_full = fetch_user_history(mode=app_mode, category=active_cat)
    if not df_full.empty:
        h_stats = df_full.groupby(['parent_node', 'child_node'])['is_correct'].agg(['mean', 'count']).reset_index()
        h_stats['Accuracy (%)'] = (h_stats['mean'] * 100).round(1)
        h_stats.rename(columns={'count': 'Attempts', 'parent_node': ui['parent'], 'child_node': ui['child']},
                       inplace=True)
        st.dataframe(h_stats[[ui['parent'], ui['child'], 'Accuracy (%)', 'Attempts']], use_container_width=True)
    else:
        st.info("No data yet. Start a quiz!")

# TAB 5: TIMELINE & PACING
if app_mode == "Exam Preparation":
    with tab5:
        st.header("⏳ Timeline & Pacing Projection")

        active_structure = st.session_state.compartments[comp_key].get('structure', {})
        total_topics = sum([len(v) for v in active_structure.values()]) if active_structure else 0

        if total_topics == 0:
            st.warning(
                f"Upload and process a {ui['f1'].split(' ', 1)[1]} first so the AI can extract the total number of {ui['child']}s.")
        else:
            target_date = st.session_state.compartments[comp_key]['target_date']
            start_date = st.session_state.compartments[comp_key]['start_date']
            days_remaining = (target_date - date.today()).days
            total_days = max(1, (target_date - start_date).days)

            df_full = fetch_user_history(mode=app_mode, category=active_cat)
            mastered_topics_count = 0
            if not df_full.empty:
                topic_acc = df_full.groupby('child_node')['is_correct'].mean()
                mastered_topics_count = len(topic_acc[topic_acc >= 0.75])

            topics_remaining = max(0, total_topics - mastered_topics_count)
            ideal_burn_rate = round(total_topics / max(1, (total_days / 7)), 1)
            required_burn_rate = round(topics_remaining / max(1, (days_remaining / 7)), 1) if days_remaining > 0 else 0

            with st.container(border=True):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Days Remaining", days_remaining, delta="Exam Passed" if days_remaining < 0 else None)
                col2.metric(f"Total Expected Domains", total_topics,
                            help="This includes topics explicitly listed in the syllabus AND unlisted topics dynamically discovered by the AI in your uploaded PYQs.")

                pace_delta = round(ideal_burn_rate - required_burn_rate, 1)
                col3.metric("Required Pace (Topics/Week)", required_burn_rate, delta=f"{pace_delta} vs Ideal",
                            delta_color="normal")
                col4.metric("Current Accuracy",
                            f"{(df_full['is_correct'].mean() * 100):.1f}%" if not df_full.empty else "N/A")

            st.subheader("📈 Projection Burn-Up Chart")

            dates = pd.date_range(start=start_date, end=target_date)
            ideal_line = np.linspace(0, total_topics, len(dates))
            chart_df = pd.DataFrame({"Ideal Mastery Path": ideal_line}, index=dates)

            today_idx = (date.today() - start_date).days
            if today_idx >= 0 and today_idx < len(dates):
                actual_line = [np.nan] * len(dates)
                actual_line[:today_idx + 1] = np.linspace(0, mastered_topics_count, today_idx + 1)
                chart_df["Your Actual Mastery"] = actual_line

            st.line_chart(chart_df, color=["#1f77b4", "#ff7f0e"])