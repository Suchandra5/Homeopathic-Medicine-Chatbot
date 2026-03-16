import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# Load dataset
data = pd.read_csv("Homeopathic Chatbot/homeopathic_medicines_dataset_1700.csv")

# Get API key
api_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-pro")

st.title("🌿 Homeopathic Medicine Chatbot")

st.write("Describe your symptoms to get homeopathic suggestions.")

user_input = st.text_input("Enter symptoms")

if st.button("Get Medicine Suggestion"):

    # Use dataset as context
    dataset_context = data.head(50).to_string()

    prompt = f"""
You are a homeopathic medicine expert.

Symptoms:
{user_input}

Use the following dataset for reference:

{dataset_context}

Suggest a medicine and explain briefly.
"""

    response = model.generate_content(prompt)

    st.write(response.text)
