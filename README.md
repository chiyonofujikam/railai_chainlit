# 📚 RAG Chatbot for ERTMS/ETCS Subset 026

This project is a **Retrieval-Augmented Generation (RAG) Chatbot** built with **Python**, designed to answer questions based on the **ERTMS/ETCS Subset 026** documentation.

It leverages **LangChain**, **Qdrant**, **Ollama**, and **Chainlit** to create a powerful and interactive AI assistant that can search, understand, and respond using embedded knowledge from the Subset 026 specification.

---

## 🔧 Features

- 🔍 **Document Retrieval** using Qdrant Vector Store
- 🤖 **LLM-powered Question Answering** with Ollama models
- 🧠 **Hybrid Embedding Support** (Ollama)
- 🗂️ **Cited Sources**: Track source documents and page numbers (WIP)
- ⚡ **Interactive UI** with Chainlit for chat-based interface

---

## 📁 Project Structure

```bash
.
├── bot.py               # Main application code
├── store.py             # Main qdrant store code
├── .env                 # Environment variables (e.g., QDRANT_URL_LOCALHOST)
├── .python-version      # python version
├── pyproject.toml       # dependencies
└── README.md            # Project documentation
```

---

## 🚀 How It Works

1. **Vector Embedding**:
   - Subset 026 documents are embedded into vectors using `nomic-embed-text` and stored in a **Qdrant** collection.

2. **Document Retrieval**:
   - At runtime, relevant documents are retrieved using similarity search (top 5).

3. **Prompt Composition**:
   - A prompt template is filled with the retrieved context and user’s question.

4. **Response Generation**:
   - An LLM model (`qwen2.5:14b`) hosted via **Ollama** generates a response based on the prompt.

5. **Streaming Output**:
   - Answers are streamed token-by-token via **Chainlit** to create a smooth user experience.

---

## 🧠 Concepts & Definitions

### ✅ RAG (Retrieval-Augmented Generation)

RAG combines traditional retrieval systems with generative models. It first retrieves relevant context (documents) and then uses a language model to generate answers based on that context.

### ✅ Subset 026 (ERTMS/ETCS)

Subset 026 is a specification within the **European Rail Traffic Management System (ERTMS)** and **European Train Control System (ETCS)**. It defines message structures, data formats, and protocols for safe and standardized train control communication.

### ✅ LLM (Large Language Model)

A deep learning model trained on vast text corpora, capable of understanding and generating human-like text. In this project, an Ollama-hosted model is used.

### ✅ Qdrant

A high-performance **vector database** for similarity search. Used here to store and retrieve document embeddings.

### ✅ Embeddings

Vector representations of text that encode semantic meaning. This enables similarity-based searches (e.g., finding related document fragments).

### ✅ LangChain

An open-source framework that helps orchestrate LLM pipelines, combining tools like retrievers, prompts, and output parsers.

### ✅ Chainlit

A tool for building conversational apps powered by LLMs. It provides the chat interface used to interact with the chatbot.

---

## ⚙️ Setup Instructions

1. **🛠️ Install uv (if not already installed)**
If you don't have uv on your machine, install it using one of the following methods in the documentation <https://docs.astral.sh/uv/getting-started/installation/>

2. **📦 Sync your environment**

```bash
uv sync
```

This installs all dependencies from your pyproject.toml using the Python version specified in .python-version.
3. **Run Chainlit App**:

```bash
uv run chainlit run .\bot.py
```

4. Open browser at `http://localhost:8000` to start chatting

---

## 📌 Notes

- The chatbot currently uses the `SUBSET026` collection stored in Qdrant.
- Ollama must be running locally with models `qwen2.5:14b` and `nomic-embed-text`.

---

## 🧪 Future Improvements

- ✅ Enable full source tracing in UI
- 🔐 Add authentication for secured access
- 📚 Upload and preprocess new Subset documents
- 🌍 Multilingual support (Subset 026 is often used internationally)

---

Sure! Here's a clean and professional `README.md` for your `store.py` file that explains its purpose, how it works, and how to run it:

---

# 📦 Qdrant Vector Store Indexer

This Python script processes PDF files and stores their semantic embeddings into a **Qdrant** vector database, enabling efficient similarity search for downstream tasks such as retrieval-augmented generation (RAG).

## 🚀 Features

- Loads and chunks documents using **Docling** and a **HybridChunker**.
- Converts content to Markdown and splits it by headers.
- Generates embeddings using **Ollama** and **HuggingFace Transformers**.
- Stores embeddings in a **Qdrant** vector database.
- Supports batch processing of a folder or a single document.

## 🧠 How It Works

### `process_files()`

- Iterates over files in `./data/SUBSET026/`.
- Converts files to Markdown using `DoclingLoader`.
- Splits content based on Markdown headers (`#`, `##`, `###`).
- Embeds the text using `Ollama`'s `nomic-embed-text:latest`.
- Stores the result in Qdrant under the collection `"SUBSET026"`.

## ▶️ Running the Script

The default behavior processes all documents in the `SUBSET026` folder:

```bash
uv run store.py
```

If you want to use the single-document indexing logic, modify the `__main__` section:

```python
if __name__ == "__main__":
    create_vector_database()
```

## 📚 Dependencies

- `langchain`
- `langchain_community`
- `langchain_core`
- `langchain_docling`
- `langchain_qdrant`
- `langchain_ollama`
- `sentence-transformers`
- `python-dotenv`
- `qdrant-client`

> ⚠️ Ensure your environment supports running Ollama models (e.g., `nomic-embed-text:latest`).

## 📝 Notes

- You can switch between HuggingFace and Ollama embeddings depending on your needs.
- Use `force_recreate=True` carefully, as it will delete existing data in the collection.
