# 🩺 Clinical Chatbot for Patient Note Summarization

This repository contains the implementation of a clinical chatbot system designed to facilitate efficient summarization and information retrieval from electronic discharge notes. The system integrates domain-specific embeddings from **Bio_ClinicalBERT**, efficient similarity search via **FAISS**, and controlled natural language generation using **LLaMA-style large language models** (via the Together API).

The project aims to support research and prototyping at the intersection of **clinical NLP**, **retrieval-augmented generation (RAG)**, and **medical decision support tools**.

---

## 🧠 Objectives

- Enable clinicians and researchers to query large-scale patient notes through a natural language interface  
- Retrieve relevant clinical chunks using semantic similarity search  
- Generate factual, concise, and date-aware summaries aligned with the clinical context  
- Present results through an intuitive Streamlit-based conversational interface  

---

## 🔬 Technical Overview

| Component              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `Bio_ClinicalBERT`     | Domain-tuned transformer model for medical text embeddings                 |
| `FAISS`                | High-speed approximate nearest neighbor search over dense vectors          |
| `Together API`         | Access to LLaMA-family models for instruction-tuned clinical summarization |
| `Streamlit`            | Web interface for input, interaction, and output visualization             |

---

## 📁 Project Structure

```bash

clinical-chatbot/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── content/
│   └── discharge_notes_40_patients.csv # Example Input
├── .streamlit/
│   └── config.toml         # Optional UI configuration

```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/akanksha-maker-ucsd/clinical_RAG.git
cd clinical-chatbot
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Configure API keys

Create a file named ```.streamlit/secrets.toml``` and include:
```bash
TOGETHER_API_KEY = "your_together_api_key"
HUGGINGFACE_HUB_TOKEN = "your_huggingface_token"
```

4. Launch the app

```bash
streamlit run app.py
```




## 🌐 Deploying to Streamlit Cloud
1. Push the project to a public GitHub repository.
2. Go to streamlit.io/cloud and log in.
3. Create a new app, link your GitHub repo, and set app.py as the main file.
4. In Settings → Secrets, copy the contents of .streamlit/secrets.toml.


## Example Clinical Queries
- “Are there any signs of neurological deterioration?”
-  “Was the patient prescribed anticoagulants at discharge?”
- “Any mention of fever or infection in the last hospital stay?”


## 📌 Notes
- The application uses a static CSV file containing de-identified discharge summaries for prototyping.
-  For real-world usage, the pipeline must be adapted to ensure HIPAA compliance, robust data pipelines, and model safety verification.


## 📚 Citation & Attribution

This project integrates open-access models and tools including:
- Bio_ClinicalBERT
- FAISS
- Streamlit
- Together AI API


## 📄 License

This repository is shared for academic and non-commercial use only.


## 📬 Contact

For academic collaboration, questions, or contributions, please contact:

Akanksha Pandey<br>
CSE, UC San Diego <br>
Email: apandey@ucsd.edu <br>
GitHub: github.com/akanksha-maker-ucsd <br>

Sarayu Pai<br>
CSE, UC San Diego <br>
Email: s2pai@ucsd.edu <br>
