import os
from typing import Iterable

import chainlit as cl
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import (Runnable, RunnableConfig,
                                       RunnablePassthrough)
from langchain_core.documents import Document as LCDocument
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_qdrant import QdrantVectorStore

load_dotenv()


qdrant_url = os.getenv("QDRANT_URL_LOCALHOST")
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

llm = OllamaLLM(
    model= "qwen2.5:14b" #deepseek-r1:14b"
)
embedding_llm = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

class PostMessageHandler(BaseCallbackHandler):
    """
    Callback handler for handling the retriever and LLM processes.
    Used to post the sources of the retrieved documents as a Chainlit element.
    """

    def __init__(self, msg: cl.Message):
        BaseCallbackHandler.__init__(self)
        self.msg = msg
        self.sources = set()

    def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
        for d in documents:
            source_page_pair = (d.metadata['source'], d.metadata['dl_meta']['doc_items'][0]['prov'][0]['page_no'])
            self.sources.add(source_page_pair)

    def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
        if len(self.sources):
            sources_text = "\n".join([f"{source}#page={page}" for source, page in self.sources])
            self.msg.elements.append(
                cl.Text(name="Sources", content=sources_text, display="inline")
            )

@cl.on_chat_start
async def on_chat_start():
    template = """Answer the question based only on the following context:

    {context}

    Question: {question}
    """

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embedding_llm,
        collection_name="SUBSET026",
        url = qdrant_url
    )

    retriever = vectorstore.as_retriever(search_kwargs={'k': 20})

    runnable = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | ChatPromptTemplate.from_template(template) # prompt
        | llm
        | StrOutputParser()
    )

    cl.user_session.set("runnable", runnable)
    
    
@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")
    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        message.content,
        config=RunnableConfig(callbacks=[
            cl.LangchainCallbackHandler(),
            # PostMessageHandler(msg)
        ]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
