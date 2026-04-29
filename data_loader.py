from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter



load_dotenv()

client = OpenAI()

reader = PDFReader()
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75
)

def clean_text(text: str) -> str:
    text = text.encode("ascii", "ignore").decode()
    text = text.replace("\xa0", " ")
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def load_then_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]

    chunks = []
    for t in texts:
        cleaned_text = clean_text(t)
        chunks.extend(splitter.split_text(cleaned_text))

    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model = EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]