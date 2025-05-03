# main.py

import os
import torch
import docx
import PyPDF2
import streamlit as st
from transformers import (
    BartTokenizer, BartForConditionalGeneration,
    RobertaTokenizer, RobertaForSequenceClassification
)

# =========================
# Load Models (cached)
# =========================
@st.cache_resource
def load_models():
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    roberta_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    return bart_model, bart_tokenizer, roberta_model, roberta_tokenizer

# =========================
# File Handling Functions
# =========================
def read_docx(file):
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def read_txt(file):
    return file.read().decode("utf-8")

def extract_text(file):
    ext = os.path.splitext(file.name)[1].lower()
    if ext == ".docx":
        return read_docx(file)
    elif ext == ".pdf":
        return read_pdf(file)
    elif ext == ".txt":
        return read_txt(file)
    else:
        st.warning(f"Unsupported file type: {file.name}")
        return ""

# =========================
# NLP Functions
# =========================
def summarize_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=40,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def detect_bias(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs[0][1].item()  # Probability of being biased

# =========================
# Streamlit UI
# =========================
def main():
    st.set_page_config(page_title="AI Document Analyzer", layout="centered")
    st.title("📄 AI-Powered Document Analyzer")
    st.markdown("Upload a `.pdf`, `.docx`, or `.txt` document to summarize it and check for potential bias.")

    uploaded_files = st.file_uploader("Upload files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("Loading models..."):
            bart_model, bart_tokenizer, roberta_model, roberta_tokenizer = load_models()

        full_text = ""
        for file in uploaded_files:
            file_text = extract_text(file)
            if file_text:
                st.success(f"✅ Extracted text from: {file.name}")
                full_text += file_text + "\n"

        if not full_text.strip():
            st.error("❌ No readable text found in uploaded documents.")
            return

        # Show extracted text
        with st.expander("📜 Show Extracted Text"):
            st.text_area("Extracted Text", value=full_text.strip(), height=300)

        # Summarization
        st.subheader("📝 Summary")
        summary = summarize_text(full_text, bart_model, bart_tokenizer)
        st.success(summary)

        # Bias Detection
        st.subheader("🧠 Bias Detection")
        bias_score = detect_bias(summary, roberta_model, roberta_tokenizer)
        st.metric(label="Bias Score", value=f"{bias_score:.2f}")
        st.caption("0 = No bias, 1 = High bias")

# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    main()
