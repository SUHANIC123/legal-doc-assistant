import streamlit as st
from transformers import pipeline
import PyPDF2
from rouge_score import rouge_scorer
import pandas as pd

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="Legal Document Evaluator", layout="wide")

st.title("⚖️ AI Legal Document Summarization & Evaluation")

# --------------------------------------------------
# Load DistilBART Model (Cached)
# --------------------------------------------------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

@st.cache_resource
def load_model():
    model_name = "sshleifer/distilbart-cnn-12-6"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model

# --------------------------------------------------
# Extract Text from PDF
# --------------------------------------------------
def extract_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# --------------------------------------------------
# Generate Summary (Chunking for Long Docs)
# --------------------------------------------------
def generate_bart_summary(text):
    tokenizer, model = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# --------------------------------------------------
# ROUGE Evaluation
# --------------------------------------------------
def calculate_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )
    scores = scorer.score(reference, generated)

    return {
        "ROUGE-1": round(scores['rouge1'].fmeasure, 3),
        "ROUGE-2": round(scores['rouge2'].fmeasure, 3),
        "ROUGE-L": round(scores['rougeL'].fmeasure, 3)
    }


# --------------------------------------------------
# UI Section
# --------------------------------------------------

uploaded_file = st.file_uploader("📄 Upload Legal PDF", type=["pdf"])

if uploaded_file is not None:

    text = extract_text(uploaded_file)

    st.success("PDF uploaded successfully!")
    st.write("Document length:", len(text), "characters")

    # Always show button after upload
    generate = st.button("🚀 Generate Summary")

    if generate:
        with st.spinner("Generating summary using DistilBART..."):
            bart_summary = generate_bart_summary(text)

        st.subheader("📝 DistilBART Summary")
        st.write(bart_summary)

        st.session_state["bart"] = bart_summary


# --------------------------------------------------
# Evaluation Section
# --------------------------------------------------

st.subheader("📊 Evaluation (ROUGE Scores)")

reference_summary = st.text_area(
    "Paste Human-Written Reference Summary Here"
)

if "bart" in st.session_state and reference_summary:
    bart_scores = calculate_rouge(
        reference_summary,
        st.session_state["bart"]
    )

    df = pd.DataFrame(
        [bart_scores],
        index=["DistilBART"]
    )

    st.table(df)