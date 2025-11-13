import os
import base64
import tempfile
import PyPDF2
from pdf2image import convert_from_path
import streamlit as st
from groq import Groq

# -------------------- CONFIG --------------------
import streamlit as st

api_key =os.environ.get("api_key") # Replace with your Groq API key
groq_client = Groq(api_key=api_key)
#POPPLER_PATH = r"C:\Program Files\poppler-24.08.0\Library\bin" 
import os
import platform

# Detect platform and set Poppler path
if platform.system() == "Windows":
    POPPLER_PATH = r"C:\Program Files\poppler-24.08.0\Library\bin"
else:
    # On Streamlit Cloud (Linux), poppler-utils is installed system-wide
    POPPLER_PATH = None


import os
import base64
import logging
import tempfile
import re
import pandas as pd
import PyPDF2
import streamlit as st
from pdf2image import convert_from_path
from sympy import sympify, latex

# groq_client must be initialized globally (as you already have it)
# Example:
# from groq import Groq
# groq_client = Groq(api_key="YOUR_API_KEY")

# -------------------- UTILS --------------------
def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def count_pdf_pages(pdf_path):
    """Count PDF pages."""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return len(pdf_reader.pages)
    except Exception as e:
        print(f"Error counting pages: {e}")
        return 0
    
def structure_diagram_text(raw_text: str) -> str:
    """
    Converts a raw diagram/flowchart transcription into a structured, stepwise format.
    """
    lines = raw_text.splitlines()
    structured_lines = []
    step_count = 0
    diagram_detected = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Detect flowchart/diagram indicators
        if any(x in stripped.lower() for x in ["->", "↓", "layer", "weight", "step"]):
            diagram_detected = True

        if diagram_detected:
            # Convert numbered or arrowed lines into stepwise format
            if stripped[0].isdigit() and "." in stripped:
                step_count += 1
                structured_lines.append(f"Step {step_count}: {stripped.split('.', 1)[1].strip()}")
            else:
                structured_lines.append(stripped)
        else:
            # Keep normal text as-is
            structured_lines.append(stripped)

    # Add a note for missing/unknown info
    

    return "\n".join(structured_lines)



# -------------------- EXTRACTION LOGIC --------------------
def extract_handwritten_answers(pdf_path):
    """
    Extracts handwritten answers using a single, plain-text transcription strategy
    to avoid Markdown headings and structured JSON sub-answers.
    """
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    results = {"roll_no": filename.upper(), "handwritten_answers": []}

    num_pages = count_pdf_pages(pdf_path)
    if num_pages == 0:
        raise ValueError("No pages found")

    combined_answer_data = {}

    # Map pages to questions
    question_page_map = {}
    qnum = 1
    for i in range(1, num_pages + 1, 2):
        question_page_map[qnum] = [i]
        if i + 1 <= num_pages:
            question_page_map[qnum].append(i + 1)
        qnum += 1

    try:
        pages = convert_from_path(
            pdf_path, dpi=400,
            poppler_path=POPPLER_PATH,
            output_folder=None, fmt="png"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to convert PDF to images. Check Poppler installation: {e}")

    for page_num, page in enumerate(pages, start=1):
        current_qnum = None
        for qnum, plist in question_page_map.items():
            if page_num in plist:
                current_qnum = qnum
                break
        if not current_qnum:
            print(f"[WARN] Page {page_num} not mapped to a question; skipping.")
            continue

        qnum_str = str(current_qnum)
        if qnum_str not in combined_answer_data:
            combined_answer_data[qnum_str] = []

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_file.close()
            page.save(tmp_file.name, "PNG")
            b64 = encode_image(tmp_file.name)
            temp_file_path = tmp_file.name
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        plain_text_prompt = f"""
This page belongs to MAIN QUESTION {current_qnum}.

Transcribe everything visible on this handwritten page, including text, math, flowcharts, or network/web diagrams.

Rules:

1. **Text/Math:** transcribe line by line exactly.

2. **Diagrams/Flowcharts:** use a structured format:
   - Layers: list each layer and its neurons
   - Connections: list neuron-to-neuron connections using -->; mark missing weights as 'unknown'
   - Flow: describe data or computation flow (forward pass, loss, backprop)
   - Stepwise numbering for process diagrams (Step 1, Step 2, …)
   - Include formulas or annotations
   
3. **Text + Diagrams:** transcribe text in reading order; diagrams immediately after corresponding text.
Return **raw transcription only**, in this structured, evaluator-friendly format, compact to reduce token usage.
"""




        try:
            resp = groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": plain_text_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ]
                }],
                max_tokens=6000,
                temperature=0.0,
            )
            raw_text_output = resp.choices[0].message.content.strip()

            if raw_text_output:
                structured_output = structure_diagram_text(raw_text_output)
                combined_answer_data[qnum_str].append(structured_output)

        except Exception as e:
            print(f"[WARN] Plain text extraction failed for Q{current_qnum} on Page {page_num}: {e}")

    for qnum_str, answer_list in combined_answer_data.items():
        qnum_int = int(qnum_str)
        final_answer_text = "\n\n".join(answer_list)
        results["handwritten_answers"].append({
            "page": question_page_map[qnum_int][0],
            "question_id": f"Q{qnum_str}",
            "answer_text": final_answer_text,
            "position": f"Question {qnum_str}"
        })

    return results

def evaluate_answer(question, answer, max_score=5):
    if not groq_client:
        
        return {"score": None, "feedback": "client call unavailable"}

    try:
        prompt = (
            f"Evaluate the following answer for the question: '{question}'. "
            f"Answer: {answer}. "
            f"Provide a score out of {max_score} based on how well the answer addresses the given question. "
            f"If the answer clearly and accurately addresses all key components of the question, full marks should be awarded. "
            f"In the feedback, explain in natural and human-like language why this score was awarded. "
            f"Keep the feedback short (1–2 sentences), highlighting strengths and improvements. "
            f"Return in exact format:\nScore: <number>\nFeedback: <detailed feedback>"
        )

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an expert evaluator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3,
            timeout=15
        )
        response_text = response.choices[0].message.content
        score_match = re.search(r'Score:\s*(\d+\.?\d*)', response_text, re.IGNORECASE)
        score = float(score_match.group(1)) if score_match else None
        if score is not None:
            score = min(max(score, 0), max_score)
        feedback_match = re.search(r'Feedback:\s*(.*?)$', response_text, re.DOTALL | re.IGNORECASE)
        feedback = feedback_match.group(1).strip() if feedback_match else "Evaluation completed based on content relevance and accuracy."
        return {"score": score, "feedback": feedback}
    except Exception as e:
       
        return {"score": None, "feedback": f"API evaluation failed: {str(e)}"}

# -------------------- SAMPLE QUESTIONS PER DOMAIN --------------------
domain_questions = {
    "AI": [
        "Neural network forward pass calculation?",
        "Web diagram of neural network architecture?",
        "Flow charts of neural network architecture?"
    ],
    "Finance": [
        "Explain NPV calculation.",
        "Describe DCF analysis."
    ],
    "English Literature": [
        "Interpret Shakespeare's Sonnet 18.",
        "Explain themes in 'Pride and Prejudice'."
    ]
}

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="Oritm-sample demo", layout="wide")


# -------------------- Sidebar: domain selection --------------------
previous_domain = st.session_state.get("selected_domain", None)
domain = st.sidebar.selectbox("Select Domain", list(domain_questions.keys()))

# Reset question index and extracted results if domain changed
if previous_domain != domain:
    st.session_state.current_q = 0
    st.session_state.extracted_results = None
    st.session_state.selected_domain = domain  # store current domain

questions_for_domain = domain_questions.get(domain, [])

# -------------------- Session state to track current question --------------------
if "current_q" not in st.session_state:
    st.session_state.current_q = 0
if "extracted_results" not in st.session_state:
    st.session_state.extracted_results = None

# Current question index
current_index = st.session_state.current_q

# -------------------- Upload PDF --------------------
uploaded_file = st.file_uploader(
    f"Upload PDF for current question ({questions_for_domain[current_index]})",
    type=["pdf"],
    key=f"upload_{current_index}"
)

# Trigger extraction only if a new PDF is uploaded for the current question
if uploaded_file is not None and st.session_state.extracted_results is None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(uploaded_file.getvalue())
        tmp_pdf_path = tmp_pdf.name

    with st.spinner("Extracting handwritten answers... ⏳"):
        st.session_state.extracted_results = extract_handwritten_answers(tmp_pdf_path)

    st.success("✅ Extraction Complete!")
    os.remove(tmp_pdf_path)

# -------------------- Display current question and answer --------------------
if st.session_state.extracted_results and current_index < len(questions_for_domain):
    question_text = questions_for_domain[current_index]
    ans_data = st.session_state.extracted_results["handwritten_answers"][0]["answer_text"]
    st.subheader(f"Q{current_index+1}: {question_text}")
    st.text_area("Extracted Answer", value=ans_data, height=200)

    # Evaluate button
    if st.button("Evaluate Current Answer", key=f"eval_{current_index}"):
        eval_result = evaluate_answer(question_text, ans_data)
        st.markdown(f"**Score:** {eval_result['score']}")
        st.markdown(f"**Feedback:** {eval_result['feedback']}")

# -------------------- Next Question Button --------------------
if st.button("Next Question", key=f"next_{current_index}"):
    if current_index + 1 < len(questions_for_domain):
        st.session_state.current_q += 1
        st.session_state.extracted_results = None  # Clear extraction for the next question
    else:
        st.info("You have reached the last question in this domain.")


# -------------------- STREAMLIT UI --------------------

