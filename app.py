import streamlit as st
import os
import faiss
import torch
import pandas as pd
import numpy as np
import re
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModel
from together import Together
from google.cloud import bigquery
from google.oauth2 import service_account
import streamlit.components.v1 as components
from datetime import datetime

# --- Configs & Styling ---
st.set_page_config(page_title="Clinical Chatbot", layout="centered")
st.markdown("""
<style>
    body { background-color: #121212; color: #f1f1f1; }
    textarea { background-color: #1e1e1e !important; color: #f1f1f1 !important;
               font-family: 'Courier New', monospace; font-size: 14px; border-radius: 6px; }
    .card { background-color: #1e1e1e; padding: 12px; margin: 8px 0; border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3); font-family: monospace; font-size: 13px; }
    .section-title { font-weight: bold; color: #f1f1f1; margin-bottom: 4px; }
    .discharge-notes-box { padding: 10px; border-radius: 10px; background-color: #1e1e1e;
                            color: #f1f1f1; font-size: 13px; border: 1px solid #444; }
</style>
""", unsafe_allow_html=True)

# --- Model Setup ---
BIOBERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
SCORE_THRESHOLD = 50
MAX_CHUNKS = 15
os.environ["TOGETHER_API_KEY"] = st.secrets["TOGETHER_API_KEY"]

@st.cache_resource
def load_biobert():
    return AutoTokenizer.from_pretrained(BIOBERT_MODEL), AutoModel.from_pretrained(BIOBERT_MODEL)

tokenizer_biobert, biobert = load_biobert()

@st.cache_resource
def get_bigquery_client():
    creds_info = st.secrets["BIGQUERY_CREDENTIALS"]
    creds = service_account.Credentials.from_service_account_info(creds_info)
    return bigquery.Client(credentials=creds, project=creds.project_id)

@st.cache_data(ttl=3600)
def query_discharge_notes(subject_id: int) -> pd.DataFrame:
    client = get_bigquery_client()
    sql = f"""
      SELECT subject_id, charttime, text
      FROM `adelaide-api.clinical_RAG.discharge_notes_40_patients`
      WHERE subject_id = {subject_id}
      ORDER BY charttime DESC
    """
    df = client.query(sql).to_dataframe()
    df['charttime'] = pd.to_datetime(df['charttime'])
    return df

# --- Utilities ---
@st.cache_data
def chunk_text(note: str, charttime, max_chars: int = 500) -> List[Tuple[str, str, str]]:
    headers = [
        "Chief Complaint:", "History of Present Illness:",
        "Major Surgical or Invasive Procedure:"  
    ]
    pattern = re.compile(rf"^({'|'.join(map(re.escape, headers))})", re.MULTILINE)
    matches = list(pattern.finditer(note))
    chunks = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(note)
        section = m.group(1)
        content = note[start:end].strip().replace("\n", " ")
        # split on sentences
        sentences = re.split(r'(?<=[.?!])\s+', content)
        buffer = ""
        for sent in sentences:
            if len(buffer) + len(sent) <= max_chars:
                buffer += sent + ' '
            else:
                chunks.append((section, buffer.strip(), charttime))
                buffer = sent + ' '
        if buffer:
            chunks.append((section, buffer.strip(), charttime))
    return chunks

@st.cache_data
def get_embedding(text: str) -> np.ndarray:
    inp = tokenizer_biobert(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad(): out = biobert(**inp)
    mask = inp['attention_mask'].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
    pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
    return pooled.squeeze().cpu().numpy()

def build_faiss_index(embeds: List[np.ndarray]):
    idx = faiss.IndexFlatL2(embeds[0].shape[0])
    idx.add(np.vstack(embeds))
    return idx

def search_chunks_with_faiss(all_chunks, sid, query):
    chunks = all_chunks.get(sid, [])
    if not chunks:
        return []
    texts = [t for _, t, _ in chunks]
    embeds = [get_embedding(t) for t in texts]
    idx = build_faiss_index(embeds)
    q_emb = get_embedding(query)
    D, I = idx.search(q_emb.reshape(1, -1), MAX_CHUNKS)
    results = []
    for i, d in zip(I[0], D[0]):
        if d <= SCORE_THRESHOLD:
            results.append((*chunks[i], d))
    return results

client = Together(api_key=os.environ["TOGETHER_API_KEY"])
def generate_response(query, chunks, current_visit_summary = ""):
    if not chunks:
        return "No relevant information found."
    ctx = "\n".join([f"[{sec} {dt.strftime('%Y-%m-%d')}] {txt}" for sec, txt, dt, _ in chunks])
    # Prepend the current visit context
    full_context = f"Current Visit Context:\n{current_visit_summary}\n\nPrior Notes:\n{ctx}"

    prompt = f"""
You are a clinical assistant. Use the Context below to answer the physician's question.

=== Current Visit Context ===
{current_visit_summary}

=== Prior Discharge Notes ===
{ctx}

Query:
{query}

Instructions:
1. If the question requests any person's name or other protected detail, reply:
   “This question requests PII, which is not available in this demo.”
2. Otherwise, organize your answer under the provided headings other relevant headings (e.g., Diagnoses, Similarities, Differences, Vital Signs).
3. Use concise bullet points.
4. Include dates for each finding when available; omit dates if none are present.
"""
    resp = client.chat.completions.create(model=MODEL_NAME, messages=[{"role":"user","content":prompt}])
    return resp.choices[0].message.content
def format_note_as_sections(note_text: str) -> str:
    section_icons = {
        "Chief Complaint:": "🩺",
        "History of Present Illness:": "📜",
        "Major Surgical or Invasive Procedure:": "🛠️",
        "Past Medical History:": "📚",
        "Physical Exam:": "🧍",
        "Discharge Diagnosis:": "🔍",
        "Discharge Medications:": "💊",
        "Discharge Disposition:": "🏠",
        "Discharge Instructions:": "📋",
        "Followup Instructions:": "📅"
    }

    section_titles = list(section_icons.keys())
    pattern = re.compile(rf"({'|'.join(map(re.escape, section_titles))})", re.MULTILINE)

    matches = list(pattern.finditer(note_text))
    formatted = ""

    for i in range(len(matches)):
        title = matches[i].group(1)
        icon = section_icons.get(title, "📝")
        start = matches[i].end()
        end = matches[i+1].start() if i+1 < len(matches) else len(note_text)
        body = note_text[start:end].strip().replace("\n", "<br>")
        formatted += f"""
        <details open>
            <summary><strong>{icon} {title}</strong></summary>
            <p>{body}</p>
        </details><br>
        """

    return formatted

PII_PATTERNS = [
    r"\bwho\b", r"\bwhat is the patient name\b",
    r"\bname\b", r"\bpatient id\b"
]

def contains_pii_request(question: str) -> bool:
    q = question.lower()
    return any(re.search(pat, q) for pat in PII_PATTERNS)
# --- UI ---
st.title("🩺 Clinical Chatbot Assistant")
subject_id = st.number_input("Patient Subject ID", min_value=0, value=10001338)

notes_df = query_discharge_notes(subject_id)
if "all_chunks" not in st.session_state:
    # build all_chunks once
    all_chunks = {}
    for _, r in notes_df.iterrows():
        sid, txt, dt = r.subject_id, r.text, r.charttime
        all_chunks.setdefault(sid, []).extend(chunk_text(txt, dt))

    # remove the most recent note from indexing
    latest_chunks = all_chunks[subject_id][:len(chunk_text(notes_df.iloc[0].text, notes_df.iloc[0].charttime))]
    all_chunks[subject_id] = all_chunks[subject_id][len(latest_chunks):]

    st.session_state.all_chunks = all_chunks

# Tabs
tab1, tab2, tab3 = st.tabs(["ℹ️ Instructions", "👤 Current Visit", "📚 Discharge Notes"])
with tab1:
    st.subheader("How to Use This Demo")
    st.markdown("""
    **Overview**  
    This demo lets you explore de-identified discharge notes for a single patient.  
    - **Current Visit** shows the most recent note’s key sections in bullet-point form.  
    - **Discharge Notes** lists prior notes; click each date to expand its full text.  

    **Note structure**  
    Every discharge note is broken into sections such as:  
    - Chief Complaint  
    - History of Present Illness  
    - Major Surgical or Invasive Procedure  
    - Past Medical History  
    - Physical Exam  
    - Discharge Diagnosis & Medications  
    - Follow-up Instructions  

    **Sample Questions** (avoid PII-requests)  
    - What surgical procedure was performed?  
    - What are some relevant labs/vital signs for this patient?  
    - How does today’s pain compare to the last visit?  
    - What medications was the patient discharged on?  
    - Are there any new developments relevant to diagnoses compared to prior notes?  

    **PII restriction**  
    This demo uses MIMIC-IV data, which is fully de-identified under the data use agreement.  
    Any question asking for names or other personal identifiers will return:  
    > “This question requests PII, which is not available in this demo.”  
    """)
current_visit_summary = ""
with tab2:
    st.subheader("Current Visit Snapshot")
    row = notes_df.iloc[0]
    cc = next((t for s,t,d in chunk_text(row['text'], row['charttime']) if 'Chief Complaint' in s), 'N/A')
    proc = next((t for s,t,d in chunk_text(row['text'], row['charttime']) if 'Major Surgical or Invasive Procedure' in s), 'None')
    hpi = next((t for s,t,d in chunk_text(row['text'], row['charttime']) if 'History of Present Illness' in s), '')
    st.markdown(f"- **Date:** {row['charttime'].strftime('%b %d, %Y')}")
    st.markdown(f"- **Chief Complaint:** {cc}")
    st.markdown(f"- **Procedure:** {proc}")
    st.markdown(f"- **HPI (brief):** {hpi[:200]}…")
    # build a one‐line summary from tab2 for prompt
    current_visit_summary = (
        f"Date: {row.charttime.strftime('%b %d, %Y')}; "
        f"Chief Complaint: {cc}; "
        f"Procedure: {proc}; "
        f"HPI: {hpi[:200]}…"
    )


with tab3:
    st.subheader("Past Discharge Notes")
    for note in notes_df.iloc[1:].itertuples():
        note_formatted = format_note_as_sections(note.text)
        date = note.charttime.strftime('%b %d, %Y')
        with st.expander(f"📅 {date}", expanded=False):
            st.markdown(f"{note_formatted}", unsafe_allow_html=True)
            

# Chat
if 'messages' not in st.session_state: st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m['role']): st.markdown(m['content'])
query = st.chat_input("Ask a clinical question...")
st.caption("Demo Restriction: This data is de-identified. Questions asking for names or other PII will return a placeholder response.”")
if query:
    st.session_state.messages.append({'role':'user','content':query})
    with st.chat_message('user'): st.markdown(query)
    with st.spinner('Generating answer...'):
        if contains_pii_request(query):
            answer = "This question requests PII, which is not available in this demo."
        else:
            chunks = search_chunks_with_faiss(st.session_state.all_chunks, subject_id, query)
            if not chunks:
                answer = "No relevant information found."
            else:
                answer = generate_response(query, chunks, current_visit_summary)
    st.session_state.messages.append({'role':'assistant','content':answer})
    with st.chat_message('assistant'): st.markdown(answer)
