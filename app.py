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
# Configs 
st.set_page_config(page_title="Clinical Chatbot", layout="centered")
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
    credentials_info = st.secrets["BIGQUERY_CREDENTIALS"]  # Add to .streamlit/secrets.toml
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

# steamlit app layout 
st.title("🩺 Clinical Chatbot Assistant")
subject_id_to_search = st.number_input("Patient Subject ID", value=10001217)
df = query_discharge_notes(subject_id_to_search)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "all_chunks" not in st.session_state:
    all_chunks = {}
    for _, row in df.iterrows():
        sid = row['subject_id']
        if sid == subject_id_to_search:
            note, charttime = row['text'], row['charttime']
            chunks = chunk_text(note, charttime)
            all_chunks[sid] = chunks
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
        response = generate_response_from_chunks(user_query, chunks)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
