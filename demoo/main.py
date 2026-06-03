import os
import re
import unicodedata
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

app = FastAPI(title="Kannada WSD Dashboard")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Global dataset state
words_db = {} # target_word -> {"senses": [], "examples": []}

def load_datasets():
    DOCS_DIR = os.path.join(BASE_DIR, "docs")
    # Parse kannada_wsd_words.md for words and senses
    with open(os.path.join(DOCS_DIR, "kannada_wsd_words.md"), "r", encoding="utf-8") as f:
        content = f.read()
    
    # regex to find words: ### 1. **ಮಾತು** (mātu)
    word_blocks = re.split(r'### \d+\. \*\*([^*]+)\*\*', content)[1:]
    for i in range(0, len(word_blocks), 2):
        word = word_blocks[i].strip()
        block_text = word_blocks[i+1]
        
        senses = []
        for line in block_text.split('\n'):
            line = line.strip()
            if re.match(r'\d+\.\s+(.*)', line):
                senses.append(line)
        
        words_db[word] = {"senses": senses, "examples": []}

    # Parse kannada_wic_dataset.md for pairs
    with open(os.path.join(DOCS_DIR, "kannada_wic_dataset.md"), "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if "|" in line and not line.startswith("pair_id") and not line.startswith("- "):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 4:
                s1, s2, w, label = parts
                if w in words_db:
                    words_db[w]["examples"].append({"s1": s1, "s2": s2, "label": label})

load_datasets()

# Cache for loaded models
loaded_models = {}

class SimilarityRequest(BaseModel):
    model_name: str
    target_word: str
    sentence1: str
    sentence2: str

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def locate_target_span(sentence: str, target: str):
    sentence = normalize_text(sentence)
    target = normalize_text(target)
    start_idx = sentence.find(target)
    if start_idx == -1:
        return None
    return (start_idx, start_idx + len(target))

def get_model(model_name: str):
    if model_name not in loaded_models:
        model_path = os.path.join(MODELS_DIR, model_name.replace("/", "_"))
        if not os.path.exists(model_path):
            raise Exception(f"Model {model_name} not found locally at {model_path}.")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        model.eval()
        loaded_models[model_name] = (tokenizer, model)
    return loaded_models[model_name]

def extract_embedding(sentence: str, target_word: str, tokenizer, model):
    enc = tokenizer([sentence], return_tensors="pt", padding=True, truncation=True, max_length=128, return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    
    with torch.no_grad():
        outputs = model(**enc)
        hidden = outputs.last_hidden_state[0] # (seq_len, hidden_size)
        
    span = locate_target_span(sentence, target_word)
    token_indices = []
    
    if span:
        s, e = span
        for ti, (ts, te) in enumerate(offsets):
            if ts < e and te > s:
                token_indices.append(ti)
                
    if token_indices:
        vec = hidden[token_indices, :].mean(dim=0)
    else:
        # fallback to CLS
        vec = hidden[0, :]
        
    vec = vec.numpy()
    return vec / np.linalg.norm(vec)

@app.get("/api/words")
def list_words():
    return {"words": list(words_db.keys())}

@app.get("/api/word/{word}")
def get_word(word: str):
    if word in words_db:
        return words_db[word]
    return {"error": "Word not found"}

@app.post("/api/similarity")
def calculate_similarity(req: SimilarityRequest):
    try:
        tokenizer, model = get_model(req.model_name)
        vec1 = extract_embedding(req.sentence1, req.target_word, tokenizer, model)
        vec2 = extract_embedding(req.sentence2, req.target_word, tokenizer, model)
        sim = float(np.dot(vec1, vec2))
        return {"similarity": sim}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/models")
def list_models():
    if not os.path.exists(MODELS_DIR):
        return {"models": []}
    dirs = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
    # Map back to original names for display
    mapping = {
        "xlm-roberta-base": "xlm-roberta-base",
        "bert-base-multilingual-cased": "bert-base-multilingual-cased",
        "google_muril-base-cased": "google/muril-base-cased",
        "ai4bharat_indic-bert": "ai4bharat/indic-bert"
    }
    return {"models": [mapping.get(d, d) for d in dirs]}

# Simple static HTML hosting
@app.get("/")
def read_root():
    with open(os.path.join(BASE_DIR, "demoo", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
