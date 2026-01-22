import os
import io
import pytesseract
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pdf2image import convert_from_path
from gtts import gTTS
import gradio as gr

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. Backend: API & Model Setup ---
from google.colab import userdata
groq_api_key = userdata.get('GROQ_API_KEY')

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.1
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = None

# --- 2. Backend: Logic Functions ---

def process_document(file_path):
    """Handles OCR, indexes for FAISS, and creates a downloadable text file."""
    global vector_db
    text = ""
    
    if file_path.endswith('.pdf'):
        pages = convert_from_path(file_path)
        for page in pages:
            text += pytesseract.image_to_string(page)
    else:
        text = pytesseract.image_to_string(Image.open(file_path))
    
    # Save the raw OCR text to a file for download
    ocr_filename = "extracted_text.txt"
    with open(ocr_filename, "w", encoding="utf-8") as f:
        f.write(text)
    
    # Indexing logic
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    vector_db = FAISS.from_texts(chunks, embeddings)
    
    return "✅ Document Processed!", ocr_filename

def chat_and_tts(question):
    if vector_db is None:
        return "Please upload a document first.", None

    retriever = vector_db.as_retriever()
    template = """Answer the question based ONLY on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    
    response_text = chain.invoke(question)
    
    tts = gTTS(response_text)
    audio_path = "response.mp3"
    tts.save(audio_path)
    
    return response_text, audio_path

# --- 3. Frontend: Updated UI ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ SmartDoc AI: OCR & Voice RAG")
    
    with gr.Row():
        with gr.Column(scale=1):
            doc_input = gr.File(label="Upload (PDF/Image)", file_types=[".pdf", ".png", ".jpg"])
            process_btn = gr.Button("🔍 Process & Index", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False)
            # NEW: Download component for OCR text
            ocr_download = gr.File(label="Download Extracted Text")
            
        with gr.Column(scale=2):
            chat_input = gr.Textbox(label="Ask a Question", placeholder="Analyze the document...")
            ask_btn = gr.Button("🤖 Get AI Response", variant="secondary")
            
            text_output = gr.Textbox(label="AI Text Answer", lines=6)
            audio_output = gr.Audio(label="Audio Response", type="filepath")

    # Event Wiring: process_document now returns two values
    process_btn.click(
        process_document, 
        inputs=[doc_input], 
        outputs=[status_output, ocr_download]
    )
    
    ask_btn.click(chat_and_tts, inputs=[chat_input], outputs=[text_output, audio_output])

demo.launch(debug=True)
