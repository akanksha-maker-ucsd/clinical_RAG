import streamlit as st
import os
import faiss
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from together import Together
from typing import List, Dict, Tuple
import re
os.environ["TOGETHER_API_KEY"] = st.secrets["TOGETHER_API_KEY"]
os.environ["HUGGINGFACE_HUB_TOKEN"] = st.secrets["HUGGINGFACE_HUB_TOKEN"]
from huggingface_hub import login
login(token=os.environ["HUGGINGFACE_HUB_TOKEN"])
from google.oauth2 import service_account
from googleapiclient.discovery import build
import pandas as pd
import io
from googleapiclient.http import MediaIoBaseDownload

# Auth
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["GDRIVE_CREDENTIALS"],  # loaded from secrets.toml
    scopes=SCOPES
)

# Build service
service = build('drive', 'v3', credentials=credentials)

# File ID
file_id = '1JJBW3fE8GZLVZQgPt1CS7HSYWeB2uqE8-p9xtIReIxU'

# Download file
request = service.files().get_media(fileId=file_id)
fh = io.BytesIO()
downloader = MediaIoBaseDownload(fh, request)
done = False
while not done:
    _, done = downloader.next_chunk()

fh.seek(0)
df = pd.read_csv(fh)
# config
BIOBERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
EMBED_DIM = 768  # BioBERT output dim
# load models
tokenizer_biobert = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
biobert = AutoModel.from_pretrained(BIOBERT_MODEL)
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

def build_faiss_index(embeddings: List[np.ndarray]) -> faiss.IndexFlatL2:  # Change to faiss.IndexFlat
    """
    Builds a FAISS index using the provided embeddings.

    Args:
        embeddings: A list of NumPy arrays representing the embeddings.

    Returns:
        A FAISS index object (faiss.IndexFlat).
    """
    dimension = embeddings[0].shape[0]  # Get dimensionality from embeddings
    index = faiss.IndexFlat(dimension)  # Change to faiss.IndexFlat
    index.add(np.vstack(embeddings))  # Add embeddings to the index
    return index
def chunk_text(input_text: str, charttime, max_chars: int = 500) -> List[Tuple[str, str]]:
    """
    Chunks a clinical text into smaller pieces based on sections and a maximum
    character limit.

    Args:
        input_text: The clinical text to be chunked.
        max_chars: The maximum number of characters allowed in each sub-chunk.

    Returns:
        A list of (section, chunk text, date) tuples.
    """

    section_headers = [
        "Service:", "Allergies:",
        "Chief Complaint:", "Major Surgical or Invasive Procedure:", "History of Present Illness:",
        "Past Medical History:", "Social History:", "Family History:", "Physical Exam:",
        "PHYSICAL EXAM ON ADMISSION:", "PHYSICAL EXAM ON DISCHARGE:", "Pertinent Results:",
        "Brief Hospital Course:", "Medications on Admission:", "Discharge Medications:",
        "Discharge Disposition:", "Facility:", "Discharge Diagnosis:", "Discharge Condition:",
        "Discharge Instructions:", "Followup Instructions:"
    ]

    pattern = re.compile(rf"^({'|'.join(map(re.escape, section_headers))})", re.MULTILINE)
    matches = list(pattern.finditer(input_text))

    chunks = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(input_text)
        header = matches[i].group(1)
        content = input_text[start:end].strip()

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
# Constants
SCORE_THRESHOLD = 50   # Adjust based on empirical observation
MAX_CHUNKS = 15

def search_chunks_with_faiss(all_chunks, subject_id, query, top_k=30):
    """
    Searches FAISS and filters chunks by score and count.
    Assumes all_chunks[subject_id] = list of tuples (text, date).
    """
    all_texts_with_dates = all_chunks.get(subject_id, [])
    if not all_texts_with_dates:
        print(f"No chunks found for subject ID: {subject_id}")
        return []

    # Build index using only the chunk text part of the tuples
    all_texts = [text for section, text, date in all_texts_with_dates]

    # embed all chunks
    embeddings = [get_embedding(text) for text in all_texts]
    index = build_faiss_index(embeddings) # build FAISS index from embeddings

    # Search
    query_emb = get_embedding(query) #embed query and use to search
    D, I = index.search(query_emb.reshape(1, -1), top_k)

    # Filter by score and cap to max chunks, include date
    results = []
    for i, score in zip(I[0], D[0]):
        if score <= SCORE_THRESHOLD:
            # Get section, text, and date from the original list using i
            section, text, date = all_texts_with_dates[i]
            results.append((text, section, date, score))
            if len(results) >= MAX_CHUNKS:
                break
    return results
os.environ["TOGETHER_API_KEY"] = os.environ["TOGETHER_API_KEY"]
client = Together(api_key=os.environ["TOGETHER_API_KEY"])
MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
def generate_response_from_chunks(query: str, retrieved_chunks: List[tuple]) -> str:
    """
    Generates a clinically relevant, faithful, and date-aware summary using retrieved context.
    """
    # Format each chunk as a date-tagged clinical note
    context = "\n".join([f"[{section} {date}] {text.strip()}" for text, section, date, score in retrieved_chunks])

    # Optimized prompt
    prompt = f"""You are a clinical assistant helping summarize a patient's medical history for a physician during clinical assessment.

Patient Timeline (each entry includes a section, date, and note):
{context}

Query: "{query}"

Instructions:
- Extract only relevant and factual information from the timeline.
- Summarize findings related to the query in clinical terms.
- Associate each finding with its date.
- Do not hallucinate or infer conditions that are not explicitly mentioned.
- Present the response in a clear, concise manner suitable for use in clinical decision-making.

Answer:"""
    # Print the length of the prompt
    print("== PROMPT:==")
    print(prompt)

    # Print the length of the prompt
    print(f"Prompt length: {len(prompt)}")

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# Streamlit app structure
st.set_page_config(page_title="Clinical Chatbot", layout="centered")
st.title("🩺 Clinical Chatbot Assistant")

# Subject ID input (fixed or from dropdown in future)
subject_id_to_search = st.number_input("Patient Subject ID", value=10001217)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize chunk dictionary only once
if "all_chunks" not in st.session_state:
    all_chunks = {}
    for index, row in df.iterrows():
        subject_id = row['subject_id']
        if subject_id == subject_id_to_search:
            note = row['text']
            charttime = row['charttime']
            chunks = chunk_text(note, charttime)
            all_chunks[subject_id] = chunks
    st.session_state.all_chunks = all_chunks

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box
user_query = st.chat_input("Enter your clinical question...")

if user_query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Process user query
    with st.spinner("Searching notes and generating response..."):
        retrieved_chunks = search_chunks_with_faiss(
            st.session_state.all_chunks,
            subject_id_to_search,
            user_query
        )
        response = generate_response_from_chunks(user_query, retrieved_chunks)

    # Display assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
