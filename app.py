## 1. Installation and Setup
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

# Configs 
st.set_page_config(page_title="Clinical Chatbot", layout="centered")
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #f1f1f1;
        }
        textarea {
            background-color: #1e1e1e !important;
            color: #f1f1f1 !important;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            border-radius: 6px;
        }
        .discharge-notes-box {
            height: 240px;
            overflow-y: auto;
            padding: 10px;
            border-radius: 10px;
            background-color: #1e1e1e;
            color: #f1f1f1;
            font-size: 13px;
            border: 1px solid #444;
        }
        .note-entry {
            margin-bottom: 16px;
        }
        .note-entry hr {
            border-color: #555;
        }
        .card {
            background-color: #1e1e1e;
            padding: 12px;
            margin: 8px 0;
            border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
            font-family: monospace;
            font-size: 13px;
        }
        .section-title {
            font-weight: bold;
            color: #f1f1f1;
            margin-bottom: 4px;
        }
    </style>
""", unsafe_allow_html=True)

BIOBERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
EMBED_DIM = 768
MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
SCORE_THRESHOLD = 50
MAX_CHUNKS = 15

os.environ["TOGETHER_API_KEY"] = st.secrets["TOGETHER_API_KEY"]

@st.cache_resource
def load_biobert():
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
    model = AutoModel.from_pretrained(BIOBERT_MODEL)
    return tokenizer, model

tokenizer_biobert, biobert = load_biobert()

@st.cache_resource
def get_bigquery_client():
    credentials_info = st.secrets["BIGQUERY_CREDENTIALS"]
    credentials = service_account.Credentials.from_service_account_info(credentials_info)
    return bigquery.Client(credentials=credentials, project=credentials.project_id)

@st.cache_data(ttl=3600)
def query_discharge_notes(subject_id: int):
    client = get_bigquery_client()
    query = f"""
        SELECT subject_id, charttime, text
        FROM `adelaide-api.clinical_RAG.discharge_notes_40_patients`
        WHERE subject_id = {subject_id}
        ORDER BY charttime DESC
    """
    return client.query(query).to_dataframe()

@st.cache_data
def chunk_text(note: str, charttime, max_chars: int = 500) -> List[Tuple[str, str, str]]:
    section_headers = ["Service:", "Allergies:", "Chief Complaint:", "Major Surgical or Invasive Procedure:",
                       "History of Present Illness:", "Past Medical History:", "Social History:",
                       "Family History:", "Physical Exam:", "PHYSICAL EXAM ON ADMISSION:",
                       "PHYSICAL EXAM ON DISCHARGE:", "Pertinent Results:", "Brief Hospital Course:",
                       "Medications on Admission:", "Discharge Medications:", "Discharge Disposition:",
                       "Facility:", "Discharge Diagnosis:", "Discharge Condition:",
                       "Discharge Instructions:", "Followup Instructions:"]
    pattern = re.compile(rf"^({'|'.join(map(re.escape, section_headers))})", re.MULTILINE)
    matches = list(pattern.finditer(note))
    chunks = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(note)
        header = matches[i].group(1)
        content = note[start:end].strip()
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.) (?=\w)', content)
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += sentence
            else:
                chunks.append((header, current_chunk.strip(), charttime))
                current_chunk = sentence
        chunks.append((header, current_chunk.strip(), charttime))
    return chunks

@st.cache_data
def get_embedding(text: str) -> np.ndarray:
    inputs = tokenizer_biobert(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = biobert(**inputs)
    attention_mask = inputs['attention_mask']
    last_hidden = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    mean_pooled = (last_hidden * mask).sum(1) / mask.sum(1)
    return mean_pooled.squeeze().cpu().numpy()

def build_faiss_index(embeddings: List[np.ndarray]) -> faiss.IndexFlat:
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlat(dimension)
    index.add(np.vstack(embeddings))
    return index

def search_chunks_with_faiss(all_chunks, subject_id, query, top_k=30):
    chunks = all_chunks.get(subject_id, [])
    if not chunks:
        return []
    texts = [text for section, text, date in chunks]
    embeddings = [get_embedding(t) for t in texts]
    index = build_faiss_index(embeddings)
    query_emb = get_embedding(query)
    D, I = index.search(query_emb.reshape(1, -1), top_k)
    results = []
    for i, score in zip(I[0], D[0]):
        if score <= SCORE_THRESHOLD:
            section, text, date = chunks[i]
            results.append((text, section, date, score))
            if len(results) >= MAX_CHUNKS:
                break
    return results

client = Together(api_key=os.environ["TOGETHER_API_KEY"])

def generate_response_from_chunks(query: str, retrieved_chunks: List[tuple]) -> str:
    context = "\n".join([f"[{section} {date}] {text.strip()}" for text, section, date, score in retrieved_chunks])
    prompt = f"""
    You are a clinical assistant reviewing the following clinical notes and answering specific medical questions.

    Patient Notes:
    {context}

    Question:
    {query}

    Instructions:
    - Provide a direct yes/no answer if possible.
    - Cite specific dates and symptoms mentioned.
    - If evidence is mixed or absent, explain briefly.
    - Keep the tone clinical and concise.

    Answer:"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# --- UI Logic ---
st.title("🩺 Clinical Chatbot Assistant")
subject_id = st.number_input("Patient Subject ID", value=10001338)
df = query_discharge_notes(subject_id)
df_sorted = df.sort_values("charttime", ascending=False).reset_index(drop=True)

# All chunks processing
if "all_chunks" not in st.session_state:
    all_chunks = {}
    for _, row in df.iterrows():
        sid = row['subject_id']
        chunks = chunk_text(row['text'], row['charttime'])
        if sid not in all_chunks:
            all_chunks[sid] = []
        all_chunks[sid].extend(chunks)
    st.session_state.all_chunks = all_chunks

# Display snapshot
st.markdown("## 🧠 Clinical Snapshot")
left, right = st.columns(2)

with left:
    st.markdown("### 👤 Current Visit Overview")
    st.markdown("""
        <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
            <img src='https://cdn-icons-png.flaticon.com/512/2922/2922510.png' width='60' style='border-radius: 50%; margin-right: 10px;' />
            <div style='background: #3a3a3a; color: white; padding: 10px 16px; border-radius: 16px; font-size: 14px; max-width: 320px; box-shadow: 0 0 6px rgba(0,0,0,0.2);'>
                “I’m here because of these symptoms...”
            </div>
        </div>
    """, unsafe_allow_html=True)

    latest_note = df_sorted.iloc[0]
    recent_chunks = chunk_text(latest_note["text"], latest_note["charttime"])
    for section, text, date in recent_chunks:
        if "Chief Complaint" in section or "History of Present Illness" in section:
            display_text = text.strip()
            if len(display_text) > 300:
                display_text = display_text[:300] + "..."
            st.markdown(f"""
                <div class='card'>
                    <div class='section-title'>{section.strip(':')}</div>
                    <div>{display_text}</div>
                </div>
            """, unsafe_allow_html=True)

with right:
    st.markdown("### 📚 Past Discharge Notes")
    with st.expander("Show Past Notes"):
        for i in range(1, min(3, len(df_sorted))):
            note = df_sorted.iloc[i]
            date = pd.to_datetime(note['charttime']).strftime("%b %d, %Y")
            st.markdown(f"#### 📅 {date}")
            sections_html = format_note_as_sections(note["text"][:2000])
            components.html(sections_html, height=300, scrolling=True)

# Chat and Query
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Enter your clinical question...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.spinner("Generating response..."):
        chunks = search_chunks_with_faiss(st.session_state.all_chunks, subject_id, query)
        if not chunks:
            answer = "No relevant information found."
        else:
            answer = generate_response_from_chunks(query, chunks)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
