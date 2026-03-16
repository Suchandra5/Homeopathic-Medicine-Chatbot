import pandas as pd
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from langchain_google_genai import (
    ChatGoogleGenerativeAI    
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

# Load .env file
load_dotenv()

# Get API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Load dataset
df = pd.read_csv("homeopathic_medicines_dataset_1700.csv")


# Convert rows → documents
documents = []

for _, row in df.iterrows():

    text = f"""
    Homeopathic Medicine: {row['medicine_name']}

    Symptoms: {row['symptom_name']}

    Indications: {row['symptom_description']}

    Potency: {row['potency']}
    """

    documents.append(Document(page_content=text))


# Gemini embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# Create FAISS vector DB
vectorstore = FAISS.from_documents(documents, embeddings)


# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)


# RAG chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)


# Ask questions
while True:

    question = input("\nAsk your medical question: ")

    if question.lower() == "exit":
        break

    result = qa.invoke({"query": question})

    print("\nAnswer:")
    print(result["result"])