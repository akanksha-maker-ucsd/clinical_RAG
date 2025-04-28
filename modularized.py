import faiss 
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel 
from typing import List, Tuple 
import re
from together import Together

#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_experimental.text_splitter import SemanticChunker

# TODO: important 
#  
# 1. Improve note preprocessing (ie remove whitespace, chunking etc.)
#
# 2. Finalize chunk metadata / size 
#    Current data not included: 
#       - hadm_id   
#       - storetime
#       - note_type (model only has support for discharge notes -> need to update if note types expand + may be useful to include to give more context to generation model)
#    Current metadata included: 
#       - subject_id (patient-specific queries so not neccesary -> inherent in data structure after user selects pid)
#       - note_id
#       - note_seq 
#       - charttime 
#       - text 
#
# 3. Test generation in pipeline (confirm Together.ai is compliant)
#
# 4. Finalize pipelines (RAG + evaluation)   
#
# 5. Clean up modularization (update preprocessing, efficiency, structure, etc.) 
#
# 6. Create agents (add query expansion, super agent)
#  
# 7. Finalize agentic workflow
# 
# 8. Work on UI for each agent's output + human-in-the-loop 

# TODO: future improvements  
#
# 1. Add error handling for incorrrect usage (ie invalid pids, etc) 
# 
# 2. Add support for other document types (multi-modal data - ie other structured data or unstructured data - radiology note)
#       - Radiology notes: df_test = pd.read_csv("mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/radiology.csv.gz", compression="gzip")
#       - Other mimic data (may require integration of other metadata ie hadm_id to support cross-database data)
#
# 3. Add support to add new patient documents (to initial database, chunked database, vector store) -> not important now, but will be in real-world use cases where the EHR is not static 


class Model(): 
    def __init__(self, data_path, embed_model, api_key, llm_model): 
        self.data = Data(data_path)
        self.embedder = Embedder(embed_model)
        self.storage = Storage(self.data, self.embedder)
        self.generator = Generation(api_key, llm_model)

class Data(): 
    # Load full dataset 
    def __init__(self, file_path): 
        self.df = pd.read_csv(file_path)                             # Initial database of notes 
        self.all_chunks = {}                                         # Stores per patient chunked data (metadata + content)  

        # Preprocess data (chunk all notes in the database to populate all_chunks)
        self.chunk_all_data()                        
    
    # Print dataset stats 
    def print_stats(self): 
        print("========================= Dataset Statistics ==========================")
        print("\nDischarge note headers:", self.df.columns.to_list())
        print("\nTotal number of notes:", len(self.df))
        id_counts = self.df['subject_id'].value_counts()
        print("Total number of patients:", len(id_counts))
        print("\nNotes per patient:\n", id_counts)

        # TODO: preview of a note?
        # print(df_test.head().to_string())  # Ensures all text is visible
        # print("\nSample Note Contents (Note 0):\n", df_test['text'][2])
        print("\n=======================================================================\n")

    # Print patient stats  
    def print_patient_stats(self, pid): 
        df = self.filter_data(pid)
        print(f"==================== Patient {pid} Statistics =============================")
        print(f"\nTotal number of notes:", len(df))

        # if patient has documents, print chunk-level stats 
        if pid in self.all_chunks:     
            chunks = self.all_chunks[pid]
            print(f"Total chunks:", len(chunks)) 
            #print(f"Max chunk len for Patient {pid}:", max(chunk_lens))  TODO: add other stats 
            #print(f"Min chunk len for Patient {pid}:", min(chunk_lens))
            #print(f"Number of chunks for Note {row['note_id']}:", len(chunks)) 
        print("\n=======================================================================\n")

    # Filter dataset based on patient id 
    def filter_data(self, pid): 
        return self.df[self.df['subject_id'] == pid]
    
    # Get sorted patient ids 
    def get_pids(self): 
        return sorted(self.df['subject_id'].unique().tolist())
    
    # Preprocess database 
    def chunk_all_data(self):
        for pid in self.get_pids(): 
            self.chunk_patient_data(pid)
        print("Chunked patient data")

    # Chunk all notes for a patient and add to all_chunks 
    def chunk_patient_data(self, pid):
        df = self.filter_data(pid)
    
        for index, row in df.iterrows():
            example_note = row['text']
            charttime = row['charttime'] # get charttime for this row 
            note_seq = row['note_seq']
            note_id = row['note_id']

            # Apply the chunk_semantically function, pass charttime as argument 
            chunks = self.chunk_text(example_note, charttime, note_id, note_seq)

            # Update the all_chunks dictionary
            if pid in self.all_chunks:
                self.all_chunks[pid].extend(chunks)
            else:
                self.all_chunks[pid] = chunks

    # Note: old splitter model TODO 
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) 

    # Chunk a single patient note 
    # TODO -> improve chunking + cleaning the text (ie new lines, punctuation, etc.)
    def chunk_text(self, input_text: str, charttime, note_id, note_seq, max_chars: int = 500) -> List[Tuple[str, str]]:
        """
        Chunks a clinical text into smaller pieces based on sections and a maximum
        character limit.

        Args:
            input_text: The clinical text to be chunked.
            max_chars: The maximum number of characters allowed in each sub-chunk.

        Returns:
            A list of (section, chunk text, date) tuples.
        """
        # TODO: figure out how to deal w/ notes that dont have content undersubheadings 

        # List of relevant headings that may start a line in the discharge note 
        section_headers = [
            "Name:", "Admission Date:", "Date of Birth:",
            "Service:", "Allergies:", "Attending:",
            "Chief Complaint:", "Major Surgical or Invasive Procedure:", "History of Present Illness:",
            "Past Medical History:", "Surgical History:", "Social History:", "Family History:", "Physical Exam:",
            "PHYSICAL EXAM ON ADMISSION:", "PHYSICAL EXAM ON DISCHARGE:", "Pertinent Results:",
            "Brief Hospital Course:", "Medications on Admission:", "Discharge Medications:",
            "Discharge Disposition:", "Facility:", "Discharge Diagnosis:", "Discharge Condition:",
            "Discharge Instructions:", "Followup Instructions:"
        ]

        # Ignore headers with PHI removed  
        # Also, some of these headers contained other headers further down the line, which is not handled by our current header matching logic 
        remove_headers = ["Name", "Admission Date", "Date of Birth", "Attending", "Facility"]

        # Clean text of spaces + newlines 
        cleaned = re.sub(r'(\n\s*|\s*\n)', '\n', input_text) # remove spaces around newlines 
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)            # condense multiple spaces into one 
        cleaned = cleaned.strip()                            # remove trailing newlines 

        # Check start of lines for match to a heading 
        pattern = re.compile(rf"^({'|'.join(map(re.escape, section_headers))})", re.MULTILINE)  
        matches = list(pattern.finditer(cleaned))

        chunks = []
        for i in range(len(matches)):
            start = matches[i].end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
            header = matches[i].group(1)[:-1]      
            content = cleaned[start:end].strip()
            
            if header not in remove_headers:

                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', content)    # TODO: improve sentence splitting (semantic splitter?)
            
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= max_chars:
                        current_chunk += sentence
                    else:
                        chunks.append((header, current_chunk.strip(), charttime, note_id, note_seq))
                        if len(sentence) <= max_chars: 
                            current_chunk = sentence                        
                        else: 
                            current_chunk = ""                                                  # Skip sentence if too long TODO improve handling of this edge case 
                chunks.append((header, current_chunk.strip(), charttime, note_id, note_seq))    # TODO: add other metadata, keep ordering cosistent across functions 

        return chunks

    # Format chunks per patient as a string for printing purposes 
    # TODO: Set default limit on max_chunk for printing in case content is too long 
    def get_formatted_chunks(self, pid, max_chunks): 

        patient_chunks = self.all_chunks[pid]
        output = [f"================= Showing up to {max_chunks} chunks for Patient {pid} ============="]

        for i, (section, chunk, date, id, seq) in enumerate(patient_chunks): 
            if i >= max_chunks: 
                output.append("\nChunk display limit reached")
                break 
            output.append(f"\nNote ID: {id} \nNote Sequence: {seq} \nDate: {date} \nSection: {section} \n{chunk}")
            output.append("\n-----------------------------------------------------------------------")
        output.append("\n=======================================================================\n")
        return "\n".join(output) 
        
class Embedder():

    # Initializes embedding model 
    def __init__(self, model_name):    
        self.options = {
            "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT", # Initialized from BioBERT & trained on all MIMIC III notes
            "BioBERT": "dmis-lab/biobert-v1.1"                 # BERT pre-trained on biomedical data and finetuned on biomedical text mining tasks
            # Add more if needed TODO 
            # Note: other models to explore: Pubmed bert, blue bert, sapbert, scibert, & gen lang models as well  
        }

        model_id = self.options[model_name]
        self.model = AutoModel.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Embed text using given model 
    def get_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        attention_mask = inputs['attention_mask']
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        masked_output = last_hidden * mask
        mean_pooled = masked_output.sum(1) / mask.sum(1)
        return mean_pooled.squeeze().cpu().numpy()

class Storage(): 
    def __init__(self, data, embedder): 
        self.pid_to_idx = {}                 # stores per patient embedded data   
        self.data = data 
        self.embedder = embedder

        # Create patient indices (use data + embedder to populate pid_to_idx)
        self.build_patient_indices()    

    # Preprocess chunk data by building indices for all patient 
    def build_patient_indices(self): 
        for pid in self.data.get_pids(): 
            self.build_faiss_index(pid)
        print("Indexed patient data")

    # Build a patients index 
    def build_faiss_index(self, pid) -> faiss.IndexFlatL2:  # Change to faiss.IndexFlat
        """
        Builds a FAISS index using the provided embeddings.

        Args:
            embeddings: A list of NumPy arrays representing the embeddings.

        Returns:
            A FAISS index object (faiss.IndexFlat).
        """
        # Build index using only the chunk text part of the tuples
        chunks_with_metadata = self.data.all_chunks[pid] 
        all_texts = [text for section, text, date, id, seq in chunks_with_metadata] # TODO: i think it makes sense to embedd the metadata too? 
        # Embed all chunks
        embeddings = [self.embedder.get_embedding(text) for text in all_texts]
        dimension = embeddings[0].shape[0]  # Get dimensionality from embeddings

        # Build FAISS index from embeddings 
        index = faiss.IndexFlat(dimension)  # Change to faiss.IndexFlat FlatL2?? # L2 distance for similarity TODO are other similarity metrics better? 
        index.add(np.vstack(embeddings))    # Add embeddings to the index
        self.pid_to_idx[pid] = index        # Add pid to index mapping 

    def search_index(self, pid, query, threshold, top_k=30): 
        """
        Searches FAISS and filters chunks by score and count.
        Assumes all_chunks[subject_id] = list of tuples (text, date).
        """
        # Constants - adjust based on empiral observation 
        #MAX_CHUNKS = 15 

        index = self.pid_to_idx[pid]          # Index w/ chunk text embeddings 
        chunks = self.data.all_chunks[pid]      # Raw chunk text w/ metadata 

        # TODO - refine approach 
        # Agentic ideas: create a knowledge graph to expand relevant search terms 
        query_emb = self.embedder.get_embedding(query) # embed query and use to search
        D, I = index.search(query_emb.reshape(1, -1), top_k)

        # Filter by score and cap to max chunks, include date
        results = []
        for i, score in zip(I[0], D[0]):
            if score <= threshold: # tunable parameter - adjust based on empiricle observation 
                # Get section, text, date, seq from the original list using i
                section, text, date, id, seq = chunks[i]
                results.append((text, section, date, id, seq, score))
            #if len(results) >= MAX_CHUNKS:
        return results
    
    # Prints the results 
    def print_retrieved_chunks(self, pid, retrieved):
        i = 0
        print(f"======================== Retrieved Chunks for Patient {pid} ===========================")
        print("\nNumber of retrieved chunks:", len(retrieved))
        for chunk_text, section, date, id, seq, score in retrieved: 
            # Print chunk 
            print(f"\nChunk {i+1} \nScore: {score} \nDate: {date} \nNote ID: {id} \nNote Sequence: {seq} \nSection: {section}")
            print(chunk_text)
            i += 1
            print("\n-----------------------------------------------------------------------")
        print("\n=======================================================================\n")

    # Evaluate retrieved chunks using QA dataset TODO - make printing optional  
    def evaluate_retrieved(self, pid, retrieved, answer_dict, correct_answer, threshold): 
        answer_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        correct = 0      # number of chunks that are above threshold and match the correct answer (ie highest sim score with answer)
        relevant = 0     # number of chunks that are above the threshold for any one of the answers 
        i = 0            # total number of chunks 

        # Embed answer choices 
        embds = []
        for answer in answer_dict: 
            embds.append(self.embedder.get_embedding(answer_dict[answer]))
        
        # TODO - avoid repeated work with prev function!!
        #print(f"======================== Retrieved Chunks for Patient {pid} ===========================")
        #print("\nNumber of retrieved chunks:", len(retrieved))
        for chunk_text, section, date, id, seq, score in retrieved: 
            # Print chunk 
            #print(f"\nChunk {i+1} \nScore: {score} \nDate: {date} \nNote ID: {id} \nNote Sequence: {seq} \nSection: {section}\n")
            #print(chunk_text)
            i += 1 

            # Calculate similarities of chunk for each answer 
            chunk_emb = self.embedder.get_embedding(chunk_text) 
            sims = []
            for answer in answer_idx: 
                idx = answer_idx[answer]
                cos_sim = np.dot(embds[idx], chunk_emb) / (np.linalg.norm(embds[idx]) *  np.linalg.norm(chunk_emb)) 
                #print(f"Cosine Similarity: {cos_sim} for {answer}")
                sims.append(cos_sim)

            # Calculate relevance of chunk 
            max_sim = max(sims)
            # relevant chunk 
            if max_sim > threshold: 
                relevant+=1 
                # correct chunk 
                if sims[answer_idx[correct_answer]] > threshold: 
                    correct+=1 
            #print("\n-----------------------------------------------------------------------")
        #print("\n=======================================================================\n")
        return relevant, correct, i 
    
    # TODO avoid repeated work ealier function - TODO make printing optional 
    def evaluate_relevant(self, pid, answer_dict, correct_answer, threshold): 
        answer_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}      
        threshold = 0.80 # TODO: need to tune after doing some data exploration 
        correct = 0      # number of chunks that are above threshold and match the correct answer (ie highest sim score with answer)
        relevant = 0     # number of chunks that are above the threshold for any one of the answers 
        total = 0        # total number of chunks 

        # Embed answer choices 
        embds = []
        for answer in answer_dict: 
            embds.append(self.embedder.get_embedding(answer_dict[answer])) 

        for section, chunk_text, date, note_id, note_seq in self.data.all_chunks[pid]: 
            total+=1 
            # Calculate similarities of chunk for each answer 
            chunk_emb = self.embedder.get_embedding(chunk_text) 
            sims = []
            for answer in answer_idx: 
                idx = answer_idx[answer]
                cos_sim = np.dot(embds[idx], chunk_emb) / (np.linalg.norm(embds[idx]) *  np.linalg.norm(chunk_emb)) 
                #print(f"Cosine Similarity: {cos_sim} for {answer}")
                sims.append(cos_sim)

            # Calculate relevance of chunk 
            max_sim = max(sims)
            # relevant chunk 
            if max_sim > threshold: 
                relevant+=1 
                # correct chunk 
                if sims[answer_idx[correct_answer]] > threshold: 
                    correct+=1  
            #print("\n-----------------------------------------------------------------------")
        #print("\n=======================================================================\n")
        return relevant, correct, total 


class Generation(): 

    def __init__(self, api_key, model_name): 
        self.client = Together(api_key=api_key)
        self.model = model_name 

    # TODO: include note id as metadata too 
    def generate_response_from_chunks(self, query: str, retrieved_chunks: List[tuple]) -> str:
        """
        Generates a clinically relevant, faithful, and date-aware summary using retrieved context.
        """
        # Format each chunk as a date-tagged clinical note
        context = "\n".join([f"[{note_id} {note_seq} {section} {date}] {text.strip()}" for text, section, date, note_id, note_seq, score in retrieved_chunks]) 

        # Optimized prompt
        prompt = f"""You are a clinical assistant helping summarize a patient's medical history for a physician during clinical assessment.

        Patient Timeline (each entry includes the note id, note sequence number, chart date, section header, and the clinical note segment):
        {context}

        Query: "{query}"

        Instructions:
        - Extract only relevant and factual information from the timeline.
        - Summarize findings related to the query in clinical terms.
        - Associate each finding with its date.
        - Do not hallucinate or infer conditions that are not explicitly mentioned.
        - Present the response in a clear, concise manner suitable for use in clinical decision-making.

        Answer:"""
        
        # Print the prompt and length 
        print("============================== PROMPT =================================")
        print(f"\nPrompt length: {len(prompt)}")
        print(f"\n{prompt}")
        print("\n=======================================================================\n")

        response = self.client.chat.completions.create(
            model= self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

'''    
formatted_docs = "\n\n".join(f"Source {i+1}: {doc}" for i, doc in enumerate(retrieved_docs))
prompt = f"""
Summarize the relevant conditions and treatments based on these clinical documents related to. Do not hallucinate or add information that is not mentioned.\n
{formatted_docs}"""

# System-level instruction
system_message = {
    "role": "system", 
    "content": "You are a medical assistant designed to summarize clinical information. Provide concise and accurate summaries strictly based on the provided documents. Avoid hallucination and ensure that all information is derived from the documents."
}

# User-level instruction 
user_message = {
    "role": "user", 
    "content": prompt
}

messages = [system_message, user_message]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
response = outputs[0]["generated_text"][-1]["content"]
print(response)

Given the following clinical documents, summarize the patient's past medical history. 
Be concise and avoid mentioning any other conditions or medical history not related.
Please ensure the summary only includes information explicitly mentioned in the documents, and do not infer anything beyond what is stated.\n\n Documents:"""
Given the retrieved patient documents, summarize key findings related to [condition/symptom]. 
Summarize the following documents"
Provide a concise assessment and plan based on the extracted information.
The summary should include: 
- Diagnoses and conditions related to 
- Relevant treatments administered, including effectiveness and outcomes. 
- Any notable test results or imaging findings that pertain to the . 
'''