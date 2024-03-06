import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import os

CHUNK_SIZE = 1000

@st.cache_data
def load_data(vector_store_dir: str = "faiss_index"):
    load_dotenv()
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://pvg-azure-openai-uk-south.openai.azure.com"

    client = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key="8d1daadc333e42b18e26d861588cfd43",
        api_version="2023-05-15"
    )

    db = FAISS.load_local(vector_store_dir, client)
    llm = AzureChatOpenAI(model_name="gpt-35-turbo", temperature=0.5, api_key="8d1daadc333e42b18e26d861588cfd43")

    print("Loading data...")

    AMAZON_REVIEW_BOT = RetrievalQA.from_chain_type(llm,
                                                    retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                              search_kwargs={"score_threshold": 0.7}))
    AMAZON_REVIEW_BOT.return_source_documents = True

    return AMAZON_REVIEW_BOT


def chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")

    AMAZON_REVIEW_BOT = load_data()

    ans = AMAZON_REVIEW_BOT.invoke({"query": message})
    if ans["source_documents"]:
        return ans["result"]
    else:
        return "I don't know."


if __name__ == "__main__":

    st.title('IRM Help Review')

    prompt = st.chat_input("Enter your questions here")

    if "user_prompt_history" not in st.session_state:
        st.session_state["user_prompt_history"] = []
    if "chat_answers_history" not in st.session_state:
        st.session_state["chat_answers_history"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if prompt:
        with st.spinner("Generating......"):
            output = chat(prompt, st.session_state["chat_history"])

            st.session_state["chat_answers_history"].append(output)
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_history"].append((prompt, output))

    # Displaying the chat history

    if st.session_state["chat_answers_history"]:
        for i, j in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
            message1 = st.chat_message("user")
            message1.write(j)
            message2 = st.chat_message("assistant")
            message2.write(i)
