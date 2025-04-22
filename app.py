## 1. Installation and Setup
import streamlit as st
import os
import faiss
import torch
import pandas as pd
import numpy as np
import re
import io
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModel
from together import Together
from google.cloud import bigquery
from google.oauth2 import service_account
import streamlit.components.v1 as components
import markdown
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
            height: 340px;
            overflow-y: auto;
            padding: 16px;
            border-radius: 10px;
            background-color: #1e1e1e;
            color: #f1f1f1;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            border: 1px solid #444;
        }

        .note-entry {
            margin-bottom: 20px;
        }

        .note-entry hr {
            border-color: #555;
        }

        summary {
            cursor: pointer;
            font-size: 15px;
            padding: 4px 0;
            color: #f1f1f1;
        }

        details {
            color: #f1f1f1;
            background: none;
            padding-bottom: 6px;
        }

        details[open] summary {
            font-weight: bold;
        }

        p {
            margin: 0;
            padding: 4px 0;
            color: #f1f1f1;
        }

        strong {
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)
BIOBERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
EMBED_DIM = 768
MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
SCORE_THRESHOLD = 50
MAX_CHUNKS = 15
file_id = '1JJBW3fE8GZLVZQgPt1CS7HSYWeB2uqE8-p9xtIReIxU'
# Set environment variables
os.environ["TOGETHER_API_KEY"] = st.secrets["TOGETHER_API_KEY"]
# Load models with cache
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
    df = client.query(query).to_dataframe()
    return df
@st.cache_data
def chunk_text(note: str, charttime, max_chars: int = 500) -> List[Tuple[str, str, str]]:
    section_headers = [
        "Service:", "Allergies:", "Chief Complaint:",
        "Major Surgical or Invasive Procedure:", "History of Present Illness:",
        "Past Medical History:", "Social History:", "Family History:",
        "Physical Exam:", "PHYSICAL EXAM ON ADMISSION:", "PHYSICAL EXAM ON DISCHARGE:",
        "Pertinent Results:", "Brief Hospital Course:", "Medications on Admission:",
        "Discharge Medications:", "Discharge Disposition:", "Facility:",
        "Discharge Diagnosis:", "Discharge Condition:", "Discharge Instructions:", "Followup Instructions:"
    ]
    pattern = re.compile(rf"^({'|'.join(map(re.escape, section_headers))})", re.MULTILINE)
    matches = list(pattern.finditer(note))
    chunks = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(note)
        header = matches[i].group(1)
        content = note[start:end].strip()
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', content)
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
    inputs = tokenizer_biobert(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = biobert(**inputs)
    attention_mask = inputs['attention_mask']
    last_hidden = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    masked_output = last_hidden * mask
    mean_pooled = masked_output.sum(1) / mask.sum(1)
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
    print("DEBUGGGG: "+ embeddings[0])
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
    prompt = f"""You are a clinical assistant helping summarize a patient's medical history for a physician during clinical assessment.


Patient Timeline:
{context}

Query: \"{query}\"

Instructions:
- Extract only relevant and factual information
- Associate each finding with its date
- Present in a clear, concise clinical manner

Answer:"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content



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
# steamlit app layout 
st.title("🩺 Clinical Chatbot Assistant")

subject_id_to_search = st.number_input("Patient Subject ID", value=10002430)
df = query_discharge_notes(subject_id_to_search)

# Sort discharge notes by charttime descending
df_sorted = df.sort_values("charttime", ascending=False).reset_index(drop=True)

# Extract chunks from most recent note
most_recent_note = df_sorted.iloc[0]
recent_chunks = chunk_text(most_recent_note["text"], most_recent_note["charttime"])


def extract_recent_chief_complaint_and_hpi(note_text: str) -> str:
    section_titles = [
        "Chief Complaint:", "History of Present Illness:"
    ]
    pattern = re.compile(rf"({'|'.join(map(re.escape, section_titles))})", re.MULTILINE)
    matches = list(pattern.finditer(note_text))

    sections = ""
    for i in range(len(matches)):
        title = matches[i].group(1)
        start = matches[i].end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(note_text)
        body = note_text[start:end].strip().replace("\n", "<br>")
        sections += f"<details open><summary><strong>{title}</strong></summary><p style='margin-top: 4px;'>{body}</p></details><br>"

    return sections or "Chief Complaint and HPI not found."

chief_complaint_html = extract_recent_chief_complaint_and_hpi(most_recent_note["text"])

# Prepare HTML for older notes (excluding the most recent one)
previous_notes_html = "<div class='discharge-notes-box'>"
for i in range(1, len(df_sorted)):
    note = df_sorted.iloc[i]
    
    charttime = pd.to_datetime(note["charttime"])
    date_str = charttime.strftime("%b %d, %Y")

    text = format_note_as_sections(note["text"][:2000])
    previous_notes_html += f"""
    <div class='note-entry'>
        <div style='background-color: #2a2a2a; color: #f1f1f1; padding: 6px 12px; border-radius: 6px;
                    display: inline-block; font-size: 13px; margin-bottom: 6px; box-shadow: 0 0 4px rgba(0,0,0,0.3);'>
            📅 {date_str}
        </div>
        <div>{text}</div>
        <hr>
    </div>
    """
previous_notes_html += "</div>"

st.markdown("## 🧠 Clinical Snapshot")

left, right = st.columns(2)

with left:
    st.markdown("### 👤 Current Visit Overview")

    st.markdown("""
        <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
            <img src='https://cdn-icons-png.flaticon.com/512/2922/2922510.png' width='60' style='border-radius: 50%; margin-right: 10px;'/>
            <div style='background: #3a3a3a; color: white; padding: 10px 16px; border-radius: 16px; font-size: 14px; max-width: 320px; box-shadow: 0 0 6px rgba(0,0,0,0.2);'>
                “I’m here because of these symptoms...”
            </div>
        </div>
    """, unsafe_allow_html=True)

    components.html(chief_complaint_html, height=300, scrolling=True)

with right:
    st.markdown("### 📚 Past Discharge Notes")
    components.html(previous_notes_html, height=340, scrolling=True)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "all_chunks" not in st.session_state:
    all_chunks = {}
    for _, row in df.iterrows():
        sid = row['subject_id']
        if sid == subject_id_to_search:
            note, charttime = row['text'], row['charttime']
            chunks = chunk_text(note, charttime)
            if sid not in all_chunks:
                all_chunks[sid] = []
            all_chunks[sid].extend(chunks)
    st.session_state.all_chunks = all_chunks

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Enter your clinical question...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    with st.spinner("Searching notes and generating response..."):
        chunks = search_chunks_with_faiss(
            st.session_state.all_chunks, subject_id_to_search, user_query
        )
        st.write("Retrieved FAISS Chunks:")
        st.write(chunks)
        response = generate_response_from_chunks(user_query, chunks)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
