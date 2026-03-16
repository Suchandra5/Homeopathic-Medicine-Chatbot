import os
import pandas as pd
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA


# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


# Load dataset
file_path = "Homeopathic Chatbot/homeopathic_medicines_dataset_1700.csv"

df = pd.read_csv(file_path)


# Convert dataset rows to LangChain Documents
docs = []

for _, row in df.iterrows():

    text = " | ".join([str(v) for v in row.values])

    docs.append(Document(page_content=text))


# Create embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create FAISS vector store
vectorstore = FAISS.from_documents(docs, embeddings)


# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=api_key,
    temperature=0.3
)


# Create RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)


# Function used by Streamlit app
def ask_homeopathy(question):

    result = qa.invoke({"query": question})

    return result["result"]


# import pandas as pd
# import os
# from dotenv import load_dotenv
# import warnings
# warnings.filterwarnings("ignore")
# from langchain_core.documents import Document
# from langchain_community.vectorstores import FAISS

# from langchain_google_genai import (
#     ChatGoogleGenerativeAI    
# )
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_classic.chains.retrieval_qa.base import RetrievalQA

# # Load .env file
# load_dotenv()

# # Get API key
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# # Load dataset
# df = pd.read_csv("homeopathic_medicines_dataset_1700.csv")


# # Convert rows → documents
# documents = []

# for _, row in df.iterrows():

#     text = f"""
#     Homeopathic Medicine: {row['medicine_name']}

#     Symptoms: {row['symptom_name']}

#     Indications: {row['symptom_description']}

#     Potency: {row['potency']}
#     """

#     documents.append(Document(page_content=text))


# # Gemini embeddings
# embeddings = HuggingFaceEmbeddings(
#     model_name="all-MiniLM-L6-v2"
# )


# # Create FAISS vector DB
# vectorstore = FAISS.from_documents(documents, embeddings)


# # Retriever
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# # Gemini LLM
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0.7,
#     google_api_key=GOOGLE_API_KEY
# )


# # RAG chain
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever
# )


# # Ask questions
# while True:

#     question = input("\nAsk your medical question: ")

#     if question.lower() == "exit":
#         break

#     result = qa.invoke({"query": question})

#     print("\nAnswer:")
#     print(result["result"])
