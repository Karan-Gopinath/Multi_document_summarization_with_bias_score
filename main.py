
import os
from transformers import BartForConditionalGeneration, BartTokenizer, RobertaTokenizer, RobertaForSequenceClassification
import torch
import docx  # For reading .docx files
import PyPDF2  # For reading .pdf files
from google.colab import files  # For file upload

# Load Models
def load_models():
    bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    return bart_model, bart_tokenizer, roberta_model, roberta_tokenizer

bart_model, bart_tokenizer, roberta_model, roberta_tokenizer = load_models()

# Summarization Function
def summarize_texts(text_list):
    concatenated_text = ' '.join(text_list)
    inputs = bart_tokenizer(concatenated_text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = bart_model.generate(inputs['input_ids'], max_length=130, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Bias Detection Function
def detect_bias(text):
    inputs = roberta_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    outputs = roberta_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs[0][1].item()

# File Reading Helpers
def read_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# Main Program Logic
def main():
    print("Upload your documents for Multi-Document Summarization and Bias Detection.")

    # Upload files
    uploaded_files = files.upload()
    text_list = []

    for file_name in uploaded_files.keys():
        ext = os.path.splitext(file_name)[1].lower()
        if ext == ".docx":
            text_list.append(read_docx(file_name))
        elif ext == ".pdf":
            text_list.append(read_pdf(file_name))
        elif ext == ".txt":
            with open(file_name, "r", encoding="utf-8") as file:
                text_list.append(file.read())
        else:
            print(f"Unsupported file type: {file_name}. Skipping...")

    if not text_list:
        print("No valid documents provided. Exiting...")
        return

    # Generate Summary
    print("\nGenerating summary...")
    summary = summarize_texts(text_list)
    print("\nSummary:")
    print(summary)

    # Detect Bias
    print("\nDetecting bias in the summary...")
    bias_score = detect_bias(summary)
    print(f"\nBias Score: {bias_score:.2f}")

if __name__ == "__main__":
    main()
