# methods/pln.py

import numpy as np
from nltk import ngrams
from collections import Counter
from bert_score import score as bert_score
import torch

def build_ngram_model(contexts, n=2):
    ngram_counts = Counter()
    for context in contexts:
        tokens = context.lower().split()
        ngram_counts.update(ngrams(tokens, n))
    total_ngrams = sum(ngram_counts.values())
    ngram_probs = {ng: count / total_ngrams for ng, count in ngram_counts.items()}
    return ngram_probs

def extract_key_phrases(context, ngram_probs, top_k=5):
    tokens = context.lower().split()
    context_ngrams = list(ngrams(tokens, 2))
    scored_phrases = [(ng, ngram_probs.get(ng, 0)) for ng in context_ngrams]
    scored_phrases.sort(key=lambda x: x[1], reverse=True)
    top_phrases = [' '.join(ng) for ng, _ in scored_phrases[:top_k]]
    return top_phrases

def generate_answer(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# --- Унифицированная сигнатура ---
def evaluate(data: list, model, tokenizer) -> dict:
    print("[PLN] Building n-gram model...")
    ngram_probs = build_ngram_model([ex["context"] for ex in data], n=2)

    predictions = []
    references = []

    for ex in data:
        key_phrases = extract_key_phrases(ex["context"], ngram_probs)
        prompt = f"Key Phrases: {', '.join(key_phrases)}\nQ: {ex['question']}\nA:"
        answer = generate_answer(prompt, tokenizer, model)
        predictions.append(answer.strip())
        references.append(ex["answer"].strip())

    print("[PLN] Computing BERTScore...")
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)

    return {
        "bert_score_f1": round(F1.mean().item(), 4),
    }
