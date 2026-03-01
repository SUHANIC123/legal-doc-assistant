import os
from transformers import pipeline
from PyPDF2 import PdfReader

# =========================
# 1️⃣ LOAD DOCUMENT
# =========================
print("Loading and preparing the document...")

file_path = "my_contract.txt"  # change to your file

text = ""

if file_path.endswith(".pdf"):
    reader = PdfReader(file_path)
    for page in reader.pages:
        text += page.extract_text() + "\n"
else:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

# =========================
# 2️⃣ SPLIT INTO CHUNKS
# =========================
max_chunk_length = 1000
chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]

print(f"Document split into {len(chunks)} chunks.")

# =========================
# 3️⃣ LOAD MODEL (SUPPORTED TASK)
# =========================
print("Loading model...")

generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=-1  # CPU
)

# =========================
# 4️⃣ SUMMARIZE CHUNKS
# =========================
print("Running chunk-based summarization...")

chunk_summaries = []

for i, chunk in enumerate(chunks):
    print(f"Summarizing chunk {i+1}/{len(chunks)}...")

    prompt = f"""
You are a legal expert.

Summarize the following legal text clearly and concisely.
Highlight obligations, risks, and key clauses.

Legal text:
{chunk}

Summary:
"""

    output = generator(
        prompt,
        max_new_tokens=200,
        do_sample=False
    )[0]["generated_text"]

    # Extract only the summary part
    summary = output.split("Summary:")[-1].strip()
    chunk_summaries.append(summary)

# =========================
# 5️⃣ FINAL SUMMARY
# =========================
print("Generating final summary...")

final_prompt = f"""
You are a legal expert.

Combine the following summaries into a clear final legal summary.

Summaries:
{chunk_summaries}

Final Summary:
"""

final_output = generator(
    final_prompt,
    max_new_tokens=300,
    do_sample=False
)[0]["generated_text"]

final_summary = final_output.split("Final Summary:")[-1].strip()

# =========================
# 6️⃣ OUTPUT
# =========================
print("\n" + "="*50)
print("          FINAL DOCUMENT SUMMARY")
print("="*50)
print(final_summary)
print("="*50) 
