import random
import torch
import numpy as np
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Literal
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import importlib
from fastapi.responses import JSONResponse

# Constants
SEED = 42
DATASET_FRACTION = 50
MODEL_NAME = "Qwen/Qwen3-14B"

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if str(device) == 'cpu':
    raise RuntimeError('Device in use is not cuda')

# Init FastAPI
app = FastAPI()

# Load model and datasets at startup
@app.on_event("startup")
def startup_event():
    print("Loading model...")
    app.state.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    app.state.model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    app.state.model.eval()

    print("Loading datasets...")
    datasets = {
        "squad_v2": load_dataset("squad_v2", split=f"validation[:{DATASET_FRACTION}]").shuffle(seed=SEED),
        "ag_news": load_dataset("ag_news", split=f"test[:{DATASET_FRACTION}]").shuffle(seed=SEED),
        "snli": load_dataset("snli", split=f"validation[:{DATASET_FRACTION}]").shuffle(seed=SEED),
        "trec": load_dataset("trec", split=f"test[:{DATASET_FRACTION}]", trust_remote_code=True).shuffle(seed=SEED),
        "wmt14": load_dataset("wmt14", "de-en", split=f"test[:{DATASET_FRACTION}]").shuffle(seed=SEED)
    }

    unified_data = {}
    for name, dataset in datasets.items():
        entries = []
        for example in dataset:
            if name == "squad_v2" and example["answers"]["text"]:
                entries.append({"question": example["question"], "context": example["context"], "answer": example["answers"]["text"][0]})
            elif name == "ag_news":
                entries.append({"question": example["text"], "context": example["text"], "answer": str(example["label"])})
            elif name == "snli" and example["label"] != -1:
                entries.append({"question": example["premise"], "context": example["hypothesis"], "answer": str(example["label"])})
            elif name == "trec":
                entries.append({"question": example["text"], "context": example["text"], "answer": str(example["coarse_label"])})
            elif name == "wmt14":
                entries.append({"question": example["translation"]["de"], "context": example["translation"]["de"], "answer": example["translation"]["en"]})
        unified_data[name] = entries

    app.state.unified_data = unified_data
    print("Datasets and model loaded successfully.")

# Endpoint for dataset selection
@app.get("/dataset")
def get_dataset(name: Literal["squad_v2", "ag_news", "snli", "trec", "wmt14"] = Query(...)):
    data = app.state.unified_data.get(name)
    if data is None:
        return {"error": "Dataset not found"}
    return {"dataset_name": name, "sample_count": len(data), "samples": data[:5]}

class EvaluationRequest(BaseModel):
    dataset: Literal["squad_v2", "ag_news", "snli", "trec", "wmt14"]
    method: Literal["rag", "graph_rag", "pln"]

@app.post("/evaluate")
def evaluate_method(request: EvaluationRequest):
    dataset = request.dataset
    method = request.method

    data = app.state.unified_data.get(dataset)
    if not data:
        return JSONResponse(status_code=404, content={"error": "Dataset not found"})

    try:
        module = importlib.import_module(f"methods.{method}")
        result = module.evaluate(
            data=data,
            model=app.state.model,
            tokenizer=app.state.tokenizer
        )
    except ImportError:
        return JSONResponse(status_code=404, content={"error": f"Method '{method}' not found"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {"method": method, "dataset": dataset, "result": result}