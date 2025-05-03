import os
import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer, RobertaTokenizer, RobertaForSequenceClassification
import torch
import docx
import PyPDF2

# Load models with caching to prevent reloading
@st.cache_resource
def load_models():
    bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    return bart_model, bart_tokenizer, roberta_model, roberta_tokenizer

bart_model, bart_tokenizer, roberta_model, roberta_tokenizer = load_models()

# File reading utilities
def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "".join([page.extract_text() or "" for page in reader.pages])

def read_txt(file):
    return file.read().decode("utf-8")

# Summarize concatenated text
def summarize_texts(text_list):
    text = ' '.join(text_list)
    inputs = bart_tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = bart_model.generate(inputs['input_ids'], max_length=130, min_length=30, 
                                      length_penalty=2.0, num_beams=4, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Bias detection
def detect_bias(text):
    inputs = roberta_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    outputs = roberta_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs[0][1].item()  # Probability of "biased"

# Streamlit UI
st.set_page_config(page_title="Multi-Doc Summarizer & Bias Detector", layout="wide")
st.title("📚 Multi-Document Summarization and Bias Detection")

uploaded_files = st.file_uploader("Upload .pdf, .docx, or .txt files", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)

if uploaded_files:
    text_list = []
    with st.spinner("Reading and extracting text..."):
        for file in uploaded_files:
            file_type = file.name.lower()
            if file_type.endswith('.pdf'):
                text_list.append(read_pdf(file))
            elif file_type.endswith('.docx'):
                text_list.append(read_docx(file))
            elif file_type.endswith('.txt'):
                text_list.append(read_txt(file))
            else:
                st.warning(f"Unsupported file: {file.name}")

    if text_list:
        with st.spinner("Generating summary..."):
            summary = summarize_texts(text_list)
            st.subheader("📝 Summary")
            st.write(summary)

        with st.spinner("Analyzing bias..."):
            bias_score = detect_bias(summary)
            st.subheader("⚖️ Bias Score")
            st.metric(label="Bias Probability (0 = Neutral, 1 = Biased)", value=f"{bias_score:.2f}")

    else:
        st.error("No valid documents to summarize.")
else:
    st.info("Please upload one or more documents to get started.")
