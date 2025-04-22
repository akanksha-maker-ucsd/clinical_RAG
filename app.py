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
    return (AutoTokenizer.from_pretrained(BIOBERT_MODEL),
            AutoModel.from_pretrained(BIOBERT_MODEL))
tokenizer_biobert, biobert = load_biobert()

@st.cache_resource
def get_bigquery_client():
    cred_info = st.secrets["BIGQUERY_CREDENTIALS"]
    creds = service_account.Credentials.from_service_account_info(cred_info)
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
    return client.query(sql).to_dataframe()

# --- Utilities ---
@st.cache_data
def chunk_text(note: str, charttime, max_chars: int = 500) -> List[Tuple[str,str,str]]:
    headers = ["Chief Complaint:", "History of Present Illness:", "Discharge Notes:"]
    pattern = re.compile(rf"^({'|'.join(map(re.escape, headers))})", re.MULTILINE)
    matches = list(pattern.finditer(note))
    chunks = []
    for i in range(len(matches)):
        start = matches[i].end()
        end = matches[i+1].start() if i+1 < len(matches) else len(note)
        section = matches[i].group(1)
        content = note[start:end].strip().replace("\n", " ")
        # truncate at sentence boundaries
        snippets = re.split(r'(?<=[.?!])\s+', content)
        curr = ""
        for sent in snippets:
            if len(curr)+len(sent) <= max_chars:
                curr += sent + ' '
            else:
                chunks.append((section, curr.strip(), charttime))
                curr = sent + ' '
        if curr:
            chunks.append((section, curr.strip(), charttime))
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
    idx.add(np.vstack(embeds)); return idx

def search_chunks_with_faiss(all_chunks, sid, query):
    chunks = all_chunks.get(sid, [])
    if not chunks: return []
    texts = [t for _, t, _ in chunks]
    embeds = [get_embedding(t) for t in texts]
    idx = build_faiss_index(embeds)
    q_emb = get_embedding(query)
    D, I = idx.search(q_emb.reshape(1,-1), MAX_CHUNKS)
    out = []
    for i, d in zip(I[0], D[0]):
        if d <= SCORE_THRESHOLD:
            sec, txt, dt = chunks[i]
            out.append((sec, txt, dt, d))
    return out

client = Together(api_key=os.environ["TOGETHER_API_KEY"])
def generate_response(query, chunks):
    if not chunks: return "No relevant information found."
    ctx = "\n".join([f"[{sec} {dt}] {txt}" for sec, txt, dt, _ in chunks])
    prompt = f"""
    You are a clinical assistant reviewing the following clinical notes and answering specific medical questions.

    Patient Notes:
    {ctx}

    Question:
    {query}

    Instructions:
    - Provide a direct yes/no answer if possible.
    - Cite specific dates and symptoms mentioned.
    - If evidence is mixed or absent, explain briefly.
    - Keep the tone clinical and concise.

    Answer:"""
    return client.chat.completions.create(model=MODEL_NAME, messages=[{"role":"user","content":prompt}]).choices[0].message.content

# --- UI ---
st.title("🩺 Clinical Chatbot Assistant")
subject_id = st.number_input("Patient Subject ID", min_value=0, value=10001338)

# Query and process notes
notes_df = query_discharge_notes(subject_id)
notes_df['charttime'] = pd.to_datetime(notes_df['charttime'])

if "all_chunks" not in st.session_state:
    all_chunks = {}
    for _, r in notes_df.iterrows():
        sid, txt, dt = r['subject_id'], r['text'], r['charttime']
        c = chunk_text(txt, dt)
        all_chunks.setdefault(sid, []).extend(c)
    st.session_state.all_chunks = all_chunks

# Tabs for UX
tab1, tab2 = st.tabs(["👤 Current Visit", "📚 Discharge Notes"])

with tab1:
    st.subheader("Current Visit Overview")
    row = notes_df.iloc[0]
    st.markdown(f'''**Date:** {row['charttime'].strftime('%b %d, %Y')}  
**Chief Complaint:** {next((t for s,t,d in chunk_text(row['text'], row['charttime']) if 'Chief Complaint' in s), 'N/A')}  ''')
    hpi = next((t for s,t,d in chunk_text(row['text'], row['charttime']) if 'History of Present Illness' in s), '')
    display = hpi[:500] + '...' if len(hpi)>500 else hpi
    st.markdown(f"**HPI:** {display}")

def format_note_as_sections(note: dict) -> str:
    icons = {
        "Chief Complaint:": "🩺",
        "History of Present Illness:": "📜"
    }
    # Get all chunks for this note
    chunks = chunk_text(note["text"], note["charttime"])
    # Group text by section
    section_map: dict[str, list[str]] = {}
    for sec, txt, _ in chunks:
        if sec in icons:
            section_map.setdefault(sec, []).append(txt.strip())

    parts = []
    # Preserve the icon order
    for sec, icon in icons.items():
        texts = section_map.get(sec)
        if not texts:
            continue
        # Join all chunks of that section
        full_text = " ".join(texts)
        parts.append(f"""{icon} **{sec}**

{full_text}""")

    return "\n\n".join(parts)

with tab2:
    st.subheader("Past Discharge Notes")
    for note in notes_df.iloc[1:4].itertuples():
        date = note.charttime.strftime("%b %d, %Y")
        with st.expander(f"📅 {date}", expanded=False):
            st.markdown(format_note_as_sections(note._asdict()), unsafe_allow_html=True)

# Chat interface
if 'messages' not in st.session_state: st.session_state.messages=[]
for m in st.session_state.messages:
    with st.chat_message(m['role']): st.markdown(m['content'])
query = st.chat_input("Ask a clinical question...")
if query:
    st.session_state.messages.append({'role':'user','content':query})
    with st.chat_message('user'): st.markdown(query)
    with st.spinner('Generating answer...'):
        resp = generate_response(query, search_chunks_with_faiss(st.session_state.all_chunks, subject_id, query))
    st.session_state.messages.append({'role':'assistant','content':resp})
    with st.chat_message('assistant'): st.markdown(resp)
