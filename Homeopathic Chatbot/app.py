import streamlit as st
import pandas as pd
import os
from google import genai

# Load API key from environment
api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)

# Load dataset
data = pd.read_csv("Homeopathic Chatbot/homeopathic_medicines_dataset_1700.csv")

st.title("🌿 Homeopathic Medicine Chatbot")

symptoms = st.text_input("Enter your symptoms")

if st.button("Get Medicine Suggestion"):

    prompt = f"""
You are a homeopathic medicine expert.

Symptoms:
{symptoms}

Suggest a homeopathic medicine and explain briefly.
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    st.write(response.text)

# import streamlit as st
# import pandas as pd
# import google.generativeai as genai
# import os

# # Load dataset
# data = pd.read_csv("Homeopathic Chatbot/homeopathic_medicines_dataset_1700.csv")

# # Get API key
# api_key = os.getenv("GOOGLE_API_KEY")

# genai.configure(api_key=api_key)

# model = genai.GenerativeModel("gemini-1.5-flash")

# st.title("🌿 Homeopathic Medicine Chatbot")

# st.write("Describe your symptoms to get homeopathic suggestions.")

# user_input = st.text_input("Enter symptoms")

# if st.button("Get Medicine Suggestion"):

#     # Use dataset as context
#     dataset_context = data.head(50).to_string()

#     prompt = f"""
# You are a homeopathic medicine expert.

# Symptoms:
# {user_input}

# Use the following dataset for reference:

# {dataset_context}

# Suggest a medicine and explain briefly.
# """

#     response = model.generate_content(prompt)

#     st.write(response.text)
