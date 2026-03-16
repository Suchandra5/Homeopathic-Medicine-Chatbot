import streamlit as st
from rag import ask_homeopathy

st.set_page_config(page_title="Homeopathic AI", page_icon="🌿")

st.title("🌿 Homeopathic Medicine Chatbot")

st.write("Describe your symptoms and get homeopathic suggestions.")

question = st.text_input("Enter symptoms")

if st.button("Get Medicine Suggestion"):

    if question:

        answer = ask_homeopathy(question)

        st.write("### Suggested Remedy")
        st.write(answer)

    else:
        st.warning("Please enter symptoms")
