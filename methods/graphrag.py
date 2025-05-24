# methods/graphrag.py

import numpy as np
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
import networkx as nx
import spacy
import torch

# --- Инициализация ---
retriever_model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

# --- Построение и использование графа знаний ---
def build_knowledge_graph(contexts):
    graph = nx.Graph()
    for context in contexts:
        doc = nlp(context)
        entities = [ent.text for ent in doc.ents]
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                graph.add_edge(entities[i], entities[j], context=context)
    return graph

def retrieve_relevant_subgraph(graph, question, top_k=5):
    doc = nlp(question)
    entities = [ent.text for ent in doc.ents if ent.text in graph.nodes()]
    subgraph_nodes = set(entities)
    for ent in entities:
        neighbors = list(graph.neighbors(ent))
        subgraph_nodes.update(neighbors[:top_k])
    return graph.subgraph(subgraph_nodes)

def convert_subgraph_to_text(subgraph):
    contexts = set()
    for u, v, data in subgraph.edges(data=True):
        if "context" in data:
            contexts.add(data["context"])
    return " ".join(contexts)

def generate_answer(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# --- Унифицированная сигнатура ---
def evaluate(data: list, model, tokenizer) -> dict:
    print("[GraphRAG] Building knowledge graph...")
    graph = build_knowledge_graph([ex["context"] for ex in data])

    predictions = []
    references = []

    for ex in data:
        subgraph = retrieve_relevant_subgraph(graph, ex["question"])
        graph_context = convert_subgraph_to_text(subgraph)
        prompt = f"Context: {graph_context}\nQ: {ex['question']}\nA:"
        answer = generate_answer(prompt, tokenizer, model)
        predictions.append(answer.strip())
        references.append(ex["answer"].strip())

    print("[GraphRAG] Computing BERTScore...")
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)

    return {
        "bert_score_f1": round(F1.mean().item(), 4),
    }
