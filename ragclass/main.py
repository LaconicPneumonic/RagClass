import logging
import os
import uuid

import fastapi
import qdrant_client
import torch
from fastapi import File, UploadFile
from llama_index.core import Document, Response, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter, LangchainNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.qdrant import QdrantVectorStore
from transformers import AutoTokenizer, BitsAndBytesConfig
from langchain.document_loaders import PyPDFLoader
import pathlib

logging.basicConfig(level=logging.INFO)
app = fastapi.FastAPI()


llm_name = os.getenv("LLM_NAME", "HuggingFaceH4/zephyr-7b-alpha")
embedding_name = "BAAI/bge-base-en-v1.5"
Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_name)
Settings.tokenizer = AutoTokenizer.from_pretrained(llm_name)
Settings.llm = HuggingFaceLLM(
    model_name=llm_name,
    tokenizer_name=llm_name,
    context_window=int(os.getenv("CONTEXT_WINDOW", "3900")),
    max_new_tokens=256,
    model_kwargs={
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    },
)

client = qdrant_client.QdrantClient("http://localhost:6333")


@app.post("/upload")
async def put(file: UploadFile = File(...)) -> dict:
    file_content = await file.read()
    documents = [Document(text=file_content)]

    nodes = SentenceSplitter(chunk_size=512).get_nodes_from_documents(documents)

    collection_name = file.filename + "_" + uuid.uuid4().hex

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )

    logging.info(f"Uploading {len(nodes)} nodes to collection {collection_name}")
    index = VectorStoreIndex.from_vector_store(vector_store)

    index.insert_nodes(nodes)

    return {
        "collection_name": collection_name,
    }


@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)) -> dict:

    filename = f"./{uuid.uuid4().hex}.pdf"
    with open(filename, "wb") as f:
        f.write(await file.read())

    # delete file after reading

    documents = PyPDFLoader(filename).load_and_split()

    collection_name = file.filename + "_" + uuid.uuid4().hex

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )

    index = VectorStoreIndex.from_vector_store(vector_store)

    nodes = SentenceSplitter(chunk_size=512).get_nodes_from_documents(
        [Document(text=doc.page_content) for doc in documents]
    )

    logging.info(f"Uploading {len(nodes)} nodes to collection {collection_name}")
    index.insert_nodes(nodes)

    pathlib.Path(filename).unlink()

    return {
        "collection_name": collection_name,
    }


@app.post("/query/{collection_name}/{question}")
async def get(collection_name: str, question: str) -> dict:
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )

    index = VectorStoreIndex.from_vector_store(vector_store)

    query_engine = index.as_query_engine()

    response: Response = query_engine.query(question)

    return {
        "response": response.response,
    }
