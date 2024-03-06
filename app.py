# streamlit_huggingface_app.py
import streamlit as st
from transformers import pipeline

st.title("Hugging Face + Streamlit App")

# 使用 Hugging Face Transformers 中的模型（这里以文本生成模型为例）
text_generator = pipeline("text-generation")

# Streamlit 输入框，接收用户输入的文本
user_input = st.text_input("Enter text for generation:", "Once upon a time")

# 生成文本并显示在 Streamlit 上
generated_text = text_generator(user_input, max_length=50, num_return_sequences=1)[0]['generated_text']
st.write("Generated Text:")
st.write(generated_text)
