# methods/rag.py

import numpy as np
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
import faiss
import torch

retriever_model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = None
retrieval_docs = None

def generate_answer(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def build_faiss_index(contexts):
    global faiss_index, retrieval_docs
    retrieval_docs = contexts
    doc_embeddings = retriever_model.encode(contexts, show_progress_bar=False, convert_to_numpy=True)
    faiss_index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    faiss_index.add(doc_embeddings)

def retrieve_context(question, top_k=1):
    question_embedding = retriever_model.encode([question], convert_to_numpy=True)
    _, indices = faiss_index.search(np.array(question_embedding), top_k)
    return " ".join([retrieval_docs[i] for i in indices[0]])

# --- Новая сигнатура ---
def evaluate(data: list, model, tokenizer) -> dict:
    print("[RAG] Building FAISS index...")
    build_faiss_index([ex["context"] for ex in data])

    predictions = []
    references = []

    for ex in data:
        retrieved = retrieve_context(ex["question"])
        prompt = f"Context: {retrieved}\nQ: {ex['question']}\nA:"
        answer = generate_answer(prompt, tokenizer, model)
        predictions.append(answer.strip())
        references.append(ex["answer"].strip())

    print("[RAG] Computing BERTScore...")
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)

    return {
        "bert_score_f1": round(F1.mean().item(), 4),
    }
