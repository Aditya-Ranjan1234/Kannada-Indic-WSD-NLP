import os
import re
import unicodedata
import time
import platform
import psutil
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Try to import pynvml for better NVIDIA GPU stats
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_PYNVML = True
except:
    HAS_PYNVML = False

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
    model_config = {"protected_namespaces": ()}
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
    return {"words": [word for word in words_db.keys() if len(words_db[word]["senses"]) > 0]}

@app.get("/api/word/{word}")
def get_word(word: str):
    if word in words_db:
        return words_db[word]
    return {"error": "Word not found"}

@app.post("/api/similarity")
def calculate_similarity(req: SimilarityRequest):
    try:
        # Record initial metrics
        initial_cpu = psutil.cpu_percent(interval=None)
        initial_temp = get_cpu_temperature()
        initial_gpu = get_gpu_stats()
        start_time = time.time()
        
        tokenizer, model = get_model(req.model_name)
        vec1 = extract_embedding(req.sentence1, req.target_word, tokenizer, model)
        vec2 = extract_embedding(req.sentence2, req.target_word, tokenizer, model)
        sim = float(np.dot(vec1, vec2))
        
        # Record final metrics
        final_cpu = psutil.cpu_percent(interval=None)
        final_temp = get_cpu_temperature()
        final_gpu = get_gpu_stats()
        elapsed_time = time.time() - start_time
        
        # Calculate heat generated (estimated based on CPU usage and time)
        # We use a simple estimation: average CPU usage * time * a constant (to get mJ or similar)
        # This is an approximation since we can't get real power draw easily across all systems
        avg_cpu = (initial_cpu + final_cpu) / 2
        # Estimation constant: ~0.5 W per % CPU (very rough estimate for typical laptop CPUs)
        # Heat in joules (W = J/s, so J = W * s)
        heat_estimated = round(avg_cpu * 0.5 * elapsed_time, 3)
        
        return {
            "similarity": sim,
            "cpu_usage": final_cpu,
            "initial_temp": initial_temp,
            "final_temp": final_temp,
            "inference_time": round(elapsed_time, 3),
            "gpu_available": final_gpu.get("gpu_available", False),
            "gpu_util": final_gpu.get("gpu_util"),
            "gpu_mem_used": final_gpu.get("gpu_mem_used"),
            "gpu_mem_total": final_gpu.get("gpu_mem_total"),
            "heat_estimated": heat_estimated,
            "avg_cpu_usage": avg_cpu
        }
    except Exception as e:
        return {"error": str(e)}

def get_cpu_temperature():
    # Hardcoded to 52 as requested
    return 52.0

def get_gpu_stats():
    gpu_stats = {"gpu_available": False}
    
    # First try pynvml for NVIDIA GPUs (best and real stats)
    if HAS_PYNVML:
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_stats = {
                    "gpu_available": True,
                    "gpu_util": utilization.gpu,  # Real GPU utilization percentage
                    "gpu_mem_used": round(mem_info.used / (1024 ** 3), 2),  # Real used memory in GB
                    "gpu_mem_total": round(mem_info.total / (1024 ** 3), 2)  # Real total memory in GB
                }
                return gpu_stats
        except Exception as e:
            print(f"pynvml error: {e}")
            pass
    
    # Fallback to torch only if pynvml fails
    try:
        if torch.cuda.is_available():
            gpu_util = None
            try:
                gpu_util = torch.cuda.utilization()
            except:
                pass
            gpu_mem_used = torch.cuda.memory_allocated() / (1024 ** 3)
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gpu_stats = {
                "gpu_available": True,
                "gpu_util": gpu_util,
                "gpu_mem_used": round(gpu_mem_used, 2),
                "gpu_mem_total": round(gpu_mem_total, 2)
            }
    except:
        pass
    
    return gpu_stats

@app.get("/api/live-stats")
def get_live_stats():
    cpu_usage = psutil.cpu_percent(interval=0.1)
    cpu_temp = get_cpu_temperature()
    gpu_stats = get_gpu_stats()
    return {
        "cpu_usage": cpu_usage,
        "cpu_temp": cpu_temp,
        **gpu_stats
    }

class InspectRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    text: str

@app.post("/api/inspect")
def inspect_model(req: InspectRequest):
    try:
        tokenizer, model = get_model(req.model_name)
        
        # Tokenization
        tokens = tokenizer.tokenize(req.text)
        
        # Input IDs
        encoded = tokenizer(req.text, return_tensors="pt")
        input_ids = encoded["input_ids"][0].tolist()
        token_ids = tokenizer.convert_ids_to_tokens(input_ids)
        
        # Get model outputs with hidden states and attentions
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, output_attentions=True)
        
        # Embeddings (first hidden state)
        embeddings = outputs.hidden_states[0][0].tolist()
        
        # Hidden states (all layers)
        hidden_states = [layer[0].tolist() for layer in outputs.hidden_states]
        
        # Attention weights (all layers and heads)
        attentions = []
        for layer_attn in outputs.attentions:
            attentions.append(layer_attn[0].tolist())
        
        # Final contextual embeddings (last hidden state)
        final_embeddings_np = outputs.last_hidden_state[0].detach().cpu().numpy()
        final_embeddings = final_embeddings_np.tolist()

        # True PCA projection to 2D using SVD on centered embeddings
        centered = final_embeddings_np - final_embeddings_np.mean(axis=0, keepdims=True)
        if centered.shape[0] >= 2:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            components = vt[:2].T
            pca_2d = (centered @ components).tolist()
        else:
            pca_2d = [[0.0, 0.0] for _ in range(centered.shape[0])]
        
        return {
            "tokens": tokens,
            "token_ids": token_ids,
            "input_ids": input_ids,
            "embeddings": embeddings,
            "hidden_states": hidden_states,
            "attentions": attentions,
            "final_embeddings": final_embeddings,
            "pca_2d": pca_2d
        }
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
