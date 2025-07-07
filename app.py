import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import gradio as gr
import requests
import os

# 🔐 Read API key from env
HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = "google/flan-t5-base"

pdf_path = "your_file.pdf"  # Change this to your PDF file name

def extract_text_from_pdf(path):
    reader = PyPDF2.PdfReader(path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t
    return text

raw_text = extract_text_from_pdf(pdf_path)

splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100)
chunks = splitter.split_text(raw_text)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_texts(chunks, embedding=embedding, persist_directory="./chroma_api_db")

def query_hf_api(prompt):
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    data = {"inputs": prompt}

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        output = response.json()
        if isinstance(output, list) and "generated_text" in output[0]:
            return output[0]["generated_text"]
        elif "answer" in output:
            return output["answer"]
        else:
            return output
    else:
        return f"[API Error] {response.status_code}: {response.text}"

def answer_question(user_input):
    retriever = db.as_retriever()
    docs = retriever.get_relevant_documents(user_input)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Context:
{context}

Question: {user_input}
"""
    return query_hf_api(prompt)

demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="Ask your question..."),
    outputs="text",
    title="📘 RAG Chatbot with HF API",
)

demo.launch(share=True)
