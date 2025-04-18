import os
from typing import Iterator
from pathlib import Path

from docling.chunking import HybridChunker
from dotenv import load_dotenv
from langchain.text_splitter import (MarkdownHeaderTextSplitter,
                                     RecursiveCharacterTextSplitter)
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

qdrant_url = os.getenv("QDRANT_URL_LOCALHOST")
EXPORT_TYPE = ExportType.MARKDOWN

# Mardown Splitter
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ],
)

def process_files():
    """ Processing pdfs """
    for idx, path in enumerate(Path(r"./data/SUBSET026").iterdir()):
        print(f"processing: {path}")
        loader = DoclingLoader(
            file_path=path,
            export_type=ExportType.MARKDOWN,
            chunker=HybridChunker(
                tokenizer="sentence-transformers/all-MiniLM-L6-v2"
            ),
        )

        splits = [
            split
            for doc in loader.load()
            for split in splitter.split_text(doc.page_content)
        ]

        embedding_ollama = OllamaEmbeddings(
            model="nomic-embed-text:latest"
        )

        if idx == 0:
            vectorstore = QdrantVectorStore.from_documents(
                documents=splits,
                embedding=embedding_ollama,
                url=qdrant_url,
                collection_name="SUBSET026",
                force_recreate=True
            )
            print(f"{path} splits added")
            continue

        vectorstore.add_documents(documents=splits)
        print(f"{path} splits added")

    print('Vector DB created successfully !')

# Create vector database
def create_vector_database():

    loader = DoclingLoader(
        file_path="./data/DeepSeek_R1.pdf",
        export_type=EXPORT_TYPE,
        chunker=HybridChunker(tokenizer="sentence-transformers/all-MiniLM-L6-v2"),
    )

    docling_documents = loader.load()
    
    # Determining the splits
    
    if EXPORT_TYPE == ExportType.DOC_CHUNKS:
        splits = docling_documents

    elif EXPORT_TYPE == ExportType.MARKDOWN:
        splits = [
            split
            for doc in docling_documents
            for split in splitter.split_text(doc.page_content)
        ]
    else:
        raise ValueError(f"Unexpected export type: {EXPORT_TYPE}")

    with open('data/output_docling.md', 'a', encoding='utf-8') as f:
        for doc in docling_documents:
            f.write(doc.page_content + '\n')

    # Initialize Embeddings
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    embedding_llm = OllamaEmbeddings(
        model="nomic-embed-text:latest"
    )

    # Create and persist a Qdrant vector database from the chunked documents
    vectorstore = QdrantVectorStore.from_documents(
        documents=splits,
        embedding=embedding_llm,
        url=qdrant_url,
        collection_name="rag",
        force_recreate=True
    )

    print('Vector DB created successfully !')

if __name__ == "__main__":
    process_files()
