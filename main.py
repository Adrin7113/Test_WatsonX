import os
import bs4
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")


loader = WebBaseLoader(
    web_path=("https://cleartax.in/s/how-to-efile-itr"),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer())
)
web_load = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(web_load)

embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(documents[:15], embedder)

llm = ChatGroq(temperature=0, model="llama3-8b-8192")

chain = llm|StrOutputParser()
print(chain.invoke('''Tell me everything about how to fill out a form for Income Tax Return (ITR) in India. Only use the information from the website https://cleartax.in/s/how-to-efile-itr.'''))