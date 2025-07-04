from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2
import os

app = FastAPI()

# === Load model and embeddings ===
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Load PDF and build FAISS ===
def load_pdf_text(path="data.pdf"):
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])

text = load_pdf_text()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_text(text)
doc_embeddings = embedder.encode(docs)

dimension = doc_embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

def retrieve_relevant_docs(query, k=3):
    q_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(q_embedding), k)
    return [docs[i] for i in indices[0]]

# === API input format ===
class QueryInput(BaseModel):
    query: str
    length: str = "short"

@app.post("/chat")
def chat(input: QueryInput):
    query = input.query
    length = input.length

    context_docs = retrieve_relevant_docs(query, k=3)
    context = "\n\n".join(context_docs)

    prompt = f"""<|system|>You are a helpful assistant.<|end|>
<|user|>
Context:
{context}

Question: {query}
Give a {'short (1-2 sentence)' if length == 'short' else 'detailed (long paragraph)'} answer based only on the context above.
<|end|>
<|assistant|>"""

    output = generator(prompt, max_new_tokens=300 if length == "long" else 80,
                       do_sample=True, temperature=0.7, top_p=0.9, return_full_text=False)
    
    return {"answer": output[0]["generated_text"].strip()}
