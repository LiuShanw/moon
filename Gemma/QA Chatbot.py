import streamlit as st
from langchain import HuggingFaceHub
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the Hugging Face API token from the environment
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Dictionary mapping model variants to their IDs
model_variants = {
    "Gemma 7B": "google/gemma-7b",
    "Gemma 7B Instruction Tuned": "google/gemma-7b-it",
    "Gemma 2B": "google/gemma-2b",
    "Gemma 2B Instruction Tuned": "google/gemma-2b-it"
}

# Streamlit UI
st.title("Gemma Chatbot")

# Dropdown menu to select the Gemma model variant
selected_variant = st.selectbox("Select Gemma Model Variant:", list(model_variants.keys()))

# Initialize the chatbot with the selected Gemma model variant
chatbot = HuggingFaceHub(
    repo_id=model_variants[selected_variant],
    model_kwargs={"temperature": 0.7, "max_length": 65000, "top_p": 0.9},
    huggingfacehub_api_token=huggingfacehub_api_token
)

# Define a function for chatting
def chat(llm, text):
    return llm(text)

# Text input for user to type messages
user_input = st.text_input("You:", "")

# Button to send message
if st.button("Send"):
    # Get response from chatbot
    response = chat(chatbot, user_input)
    # Display response
    st.text_area("Gemma:", value=response, height=100)