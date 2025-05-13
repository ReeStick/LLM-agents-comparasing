import torch
from transformers import RagTokenizer, RagTokenForGeneration, RagRetriever, RagSequenceForGeneration
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Загрузка датасета
dataset = load_dataset('squad', split='train[:5000]')  # Используем только 5000 примеров для обучения
test_dataset = load_dataset('squad', split='validation[:1000]')  # Используем 1000 примеров для тестирования

# Инициализация моделей и токенизаторов
rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
rag_retriever = RagRetriever.from_pretrained("facebook/rag-token-base", index_name="exact", use_dummy_dataset=True)
rag_model = RagTokenForGeneration.from_pretrained("facebook/rag-token-base", retriever=rag_retriever)

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Функция для подготовки данных
def prepare_data(batch):
    questions = [q.strip() for q in batch['question']]
    contexts = [c.strip() for c in batch['context']]
    answers = [a['text'][0].strip() for a in batch['answers']]
    return questions, contexts, answers

# Подготовка данных
train_questions, train_contexts, train_answers = prepare_data(dataset)
test_questions, test_contexts, test_answers = prepare_data(test_dataset)

# Функция для вычисления точности
def compute_accuracy(model, tokenizer, questions, contexts, answers, model_type='rag'):
    correct = 0
    total = len(questions)
    
    for i in tqdm(range(total)):
        if model_type == 'rag':
            input_text = f"Question: {questions[i]} Context: {contexts[i]}"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            outputs = model.generate(inputs["input_ids"], max_length=50)
            predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            input_text = f"Question: {questions[i]} Context: {contexts[i]}"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            outputs = model.generate(inputs["input_ids"], max_length=50)
            predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if predicted_answer.strip() == answers[i].strip():
            correct += 1
    
    return correct / total

# Вычисление точности для RAG
rag_accuracy = compute_accuracy(rag_model, rag_tokenizer, test_questions, test_contexts, test_answers, model_type='rag')
print(f"RAG Accuracy: {rag_accuracy:.4f}")

# Вычисление точности для GPT-2
gpt2_accuracy = compute_accuracy(gpt2_model, gpt2_tokenizer, test_questions, test_contexts, test_answers, model_type='gpt2')
print(f"GPT-2 Accuracy: {gpt2_accuracy:.4f}")