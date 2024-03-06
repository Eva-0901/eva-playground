# streamlit_huggingface_app.py
import streamlit as st

st.title("Hugging Face + Streamlit App")

# Streamlit 输入框，接收用户输入的文本
user_input = st.text_input("Enter text for generation:", "Once upon a time")

