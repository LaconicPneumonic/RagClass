import logging

import requests

file_content = requests.get(
    "https://gist.githubusercontent.com/LaconicPneumonic/b7fb7ce6ac9845ffa69a9c14402d0ffb/raw/1824ce3d7e10ada3efe75c6568a76ec4b343af61/civilrights.txt"
).text

from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)


documents = [Document(text=file_content)]

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")
Settings.llm = HuggingFaceLLM(
    model_name="HuggingFaceH4/zephyr-7b-alpha",
    tokenizer_name="HuggingFaceH4/zephyr-7b-alpha",
)
index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine()
response = query_engine.query("When did the supreme court order ole miss to integrate?")
print(response)
