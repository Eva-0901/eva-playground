from langchain.vectorstores import FAISS
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import os
from dotenv import load_dotenv

CHUNK_SIZE = 1000


# CREATE A VECTOR DATABASE - FAISS
def create_vector_db(pdf) -> FAISS:
    load_dotenv()
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://pvg-azure-openai-uk-south.openai.azure.com"

    client = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version="2023-05-15"
    )

    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                   chunk_overlap=100)
    texts = text_splitter.split_text(text)

    vector_db = FAISS.from_texts(texts, client)  # create vector db for similarity search
    vector_db.save_local("faiss_index")  # save the vector db to avoid repeated calls to it
    return vector_db

directory_name = "IRM Help.pdf"
create_vector_db(directory_name)
