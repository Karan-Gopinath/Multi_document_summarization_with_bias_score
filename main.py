import os
import torch
import docx
import PyPDF2
import streamlit as st
from transformers import (
    BartTokenizer, BartForConditionalGeneration,
    RobertaTokenizer, RobertaForSequenceClassification
)

# ---------------------------
# Load models (with caching)
# ---------------------------
@st.cache_resource
def load_models():
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    roberta_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    return bart_model, bart_tokenizer, roberta_model, roberta_tokenizer

# ---------------------------
# Read file content
# ---------------------------
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

# ---------------------------
# NLP Functions
# ---------------------------
def summarize_chunks(text, model, tokenizer, max_chunk=1024):
    """Split long text into chunks and summarize each."""
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"][0]
    
    summaries = []
    for i in range(0, len(input_ids), max_chunk):
        chunk = input_ids[i:i + max_chunk]
        input_dict = {"input_ids": chunk.unsqueeze(0)}
        summary_ids = model.generate(
            input_dict["input_ids"],
            max_length=150,
            min_length=40,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return "\n\n".join(summaries)

def detect_bias(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs[0][1].item()  # Probability of bias

# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.set_page_config(page_title="🧠 AI Document Analyzer", layout="centered")
    st.title("📄 AI Document Analyzer")
    st.markdown("Analyze documents for **summarization** and **bias detection** using powerful NLP models.")

    with st.sidebar:
        st.header("📂 Upload Document")
        uploaded_files = st.file_uploader(
            "Choose .pdf, .docx, or .txt files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )
        st.markdown("---")
        st.caption("Built with [🤗 Transformers](https://huggingface.co/) and Streamlit")

    if uploaded_files:
        with st.spinner("🔄 Loading models..."):
            bart_model, bart_tokenizer, roberta_model, roberta_tokenizer = load_models()

        full_text = ""
        for file in uploaded_files:
            text = extract_text(file)
            if text:
                st.success(f"✅ Processed: {file.name}")
                full_text += text + "\n"

        if not full_text.strip():
            st.error("❌ No text could be extracted.")
            return

        # Show extracted content
        with st.expander("🧾 View Extracted Text"):
            st.text_area("Extracted Text", full_text.strip(), height=250)

        # Summarization Section
        st.subheader("📝 Document Summary")
        with st.spinner("Generating summary..."):
            summary = summarize_chunks(full_text, bart_model, bart_tokenizer)

        st.markdown("#### 📌 Summary Result")
        st.info(summary)

        # Bias Detection Section
        st.subheader("🔍 Bias Detection")
        with st.spinner("Analyzing bias..."):
            bias_score = detect_bias(summary, roberta_model, roberta_tokenizer)

        st.metric(
            label="🧠 Bias Probability",
            value=f"{bias_score:.2f}",
            delta=None,
            delta_color="off",
        )

        if bias_score > 0.7:
            st.warning("⚠️ High potential bias detected.")
        elif bias_score > 0.4:
            st.info("🔎 Some bias detected.")
        else:
            st.success("✅ Minimal bias detected.")

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    main()
