# 📚 RAG-Powered Document Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green)
![Vector Store](https://img.shields.io/badge/Vector%20Store-InMemoryVectorStore-orange)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20%7C%20Qwen2.5%3A1.5b-black)
![Gradio](https://img.shields.io/badge/UI-Gradio-yellow)

A local, privacy-focused **Retrieval-Augmented Generation (RAG)** assistant designed to answer questions directly from custom PDF documents and study notes (supporting both Arabic and English). Instead of relying solely on an LLM's pre-trained knowledge, this pipeline retrieves the most semantically relevant chunks from your documents to generate accurate, context-grounded responses.

---

## 🎯 Project Overview

Large Language Models (LLMs) often hallucinate or lack access to private, domain-specific data. This project solves that problem by implementing an end-to-end RAG pipeline (`Load → Split → Embed → Store → Retrieve → Generate`) running completely locally using open-source tools.

### Key Features
- **Local & Zero-Cost Execution:** Powered by local multilingual embeddings (`langchain-huggingface`) and lightweight local LLM inference (`Ollama`), ensuring full data privacy and fast execution with no external API fees.
- **Multilingual Semantic Search:** Uses dense vector representations stored in `InMemoryVectorStore` (`langchain-core`) to retrieve contextually relevant passages in Arabic and English rather than simple keyword matching.
- **Interactive Web Interface:** Built with `Gradio` running locally on `127.0.0.1` for a clean, user-friendly Q&A experience.

---

## 🏗️ System Architecture

```text
+-------------------+      +---------------------+      +-----------------------+
|   PDF Document    | ---> |  PyMuPDF & Text     | ---> |  LangChain-HuggingFace|
|   (data/notes.pdf)|      |  Splitter (Chunks)  |      |  Multilingual Embeds  |
+-------------------+      +---------------------+      +-----------------------+
                                                                    |
                                                                    v
+-------------------+      +---------------------+      +-----------------------+
|   Final Grounded  | <--- |  Ollama LLM         | <--- |  InMemoryVectorStore  |
|   Answer (Gradio) |      |  (Qwen 2.5 - 1.5B)  |      |  (Top-K Search)       |
+-------------------+      +---------------------+      +-----------------------+

### How It Works

1. Document Ingestion & Chunking: PyMuPDFLoader extracts text from PDF files, and RecursiveCharacterTextSplitter breaks it into overlapping chunks (chunk_size=500, chunk_overlap=50) to preserve semantic continuity.
2. Embedding & Indexing: Each text chunk is converted into a 384-dimensional vector using sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 via HuggingFaceEmbeddings and indexed in InMemoryVectorStore.
3. **Retrieval:** When a user submits a query, it is embedded into the same vector space, and the retriever fetches the top `k=3` most similar document chunks.
4. **Augmented Generation:** The retrieved chunks are injected into a prompt template alongside the user's question and passed to `Qwen 2.5 (1.5B)` via Ollama to generate a grounded response.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Orchestration:** LangChain (langchain, langchain-community, langchain-core, langchain-text-splitters, langchain-classic)
* **PDF Parsing:** PyMuPDF (`fitz`)
* **Embeddings:** langchain-huggingface & sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
* **Vector Store:** `InMemoryVectorStore` (`langchain-core`)
* **LLM Inference:** Ollama (`qwen2.5:1.5b`)
* **Frontend UI:** Gradio

---

## 📂 Repository Structure

```text
Project-rag-powered-assistant/
│
├── data/
│   └── notes.pdf               # Sample document used for knowledge base
├── screenshots/
│   ├── query_1.jpg             # Sample Q&A output 1 (English)
│   └── query_2.jpg             # Sample Q&A output 2 (Arabic)
├── app.py                      # Main RAG pipeline & Gradio UI script
├── requirements.txt            # Project dependencies
├── .gitignore                  # Ignored local/virtual environment files
└── README.md                   # Project documentation

```

---

## 🚀 Setup & Installation

### Prerequisites

1. **Python 3.10+** installed on your system.
2. **Ollama** installed locally. Download it from [ollama.com](https://ollama.com/) and pull the lightweight Qwen 2.5 model:
```bash
ollama pull qwen2.5:1.5b

```



### Step-by-Step Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/aboodyal1996/Project-rag-powered-assistant.git](https://github.com/aboodyal1996/Project-rag-powered-assistant.git)
cd Project-rag-powered-assistant

```


2. **Create and activate a virtual environment:**
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


4. **Add your document:**
Place your target PDF file inside the `data/` folder and ensure the filename matches the path in `app.py` (default: `data/notes.pdf`).
5. **Run the application:**
```bash
python app.py

```


Open the local URL displayed in your terminal (typically `http://127.0.0.1:7860`) to start asking questions.

---

## 📸 Sample Queries & Output

Below are examples demonstrating the assistant answering questions strictly based on the uploaded document:

### Example 1: Summarizing Key Concepts (English)

> **Prompt:** *"What are the main ideas covered in the file?"*

### Example 2: Specific Detail Extraction (Arabic)

> **Prompt:** *"ما هي الأفكار الرئيسية التي يتناولها الملف؟"*

---

## 🔮 Future Improvements

* Support multi-file uploads directly through the Gradio UI.
* Add source citations (page numbers) alongside every generated answer.
* Implement conversational memory (`ConversationalRetrievalChain`) for follow-up questions.

---

## 👨‍💻 Author

**Abdulrahman**

* GitHub: [@aboodyal1996](https://github.com/aboodyal1996)
