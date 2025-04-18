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

---

## 🧪 Future Improvements

- ✅ Enable full source tracing in UI
- 🔐 Add authentication for secured access
- 📚 Upload and preprocess new Subset documents
- 🌍 Multilingual support (Subset 026 is often used internationally)

---

## 📫 Author

**Mustapha Elkamili**  
📍 Casablanca, Morocco  
📧 <melkamili@iksoconsulting.com>
