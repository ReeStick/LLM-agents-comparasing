# LLM Preprocessing Methods Evaluation

Это проект для сравнения качества генерации ответов моделей LLM в зависимости от способа предварительной обработки контекста: RAG, GraphRAG и PLN. Поддерживаются несколько датасетов (SQuAD v2, AG News, SNLI, TREC, WMT14), метрика — BERTScore.

## 🔧 Установка

```bash
git clone https://github.com/your-repo/llm-preprocessing-eval.git
cd llm-preprocessing-eval

# Установка зависимостей
pip install -r requirements.txt

# Загрузка моделей и данных (по необходимости вручную)
python prepare.py  # если есть
```

## 🚀 Запуск сервера

```bash
uvicorn server:app --reload
```

После запуска сервер будет доступен по адресу:  
`http://127.0.0.1:8000`

## 🧪 Запуск клиента

```bash
python frontend.py
```

Этот скрипт отправляет POST-запрос на `/evaluate` с выбранным методом и датасетом.

## Пример запроса

```json
POST /evaluate
Content-Type: application/json

{
  "dataset": "squad_v2",
  "method": "rag"
}
```

## 📁 Структура проекта

```
.
├── server.py            # FastAPI сервер
├── frontend.py          # Клиент для отправки запросов
├── methods/
│   ├── rag.py
│   ├── graph_rag.py
│   └── pln.py
├── requirements.txt     # Зависимости
└── README.md
```

## 🧠 Описание методов

- **RAG** — Извлекает релевантный контекст по эмбеддингам.
- **GraphRAG** — Строит граф сущностей и извлекает подграф на основе запроса.
- **PLN (PLM)** — Извлекает ключевые фразы по n-граммной вероятности.

## 📊 Метрики

- `BERTScore_F1` — сравнивает семантическое сходство между предсказанием и референсом.

## 📌 Требования

- Python 3.8+
- `transformers`, `datasets`, `faiss-cpu`, `spacy`, `sentence-transformers`, `bert-score`, `nltk`

---

**Примечание:** Убедитесь, что модель `en_core_web_sm` загружена для spaCy:

```bash
python -m spacy download en_core_web_sm
```
