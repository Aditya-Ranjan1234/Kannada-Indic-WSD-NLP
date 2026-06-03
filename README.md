# Cross-lingual Sense Transfer with Low-Resource Indic Languages: Kannada Word Sense Disambiguation Benchmark

## Introduction
Word Sense Disambiguation (WSD) in low-resource languages remains an unresolved bottleneck for high-quality natural language understanding. This repository presents a structured, reproducible Kannada WSD benchmark that evaluates four multilingual encoders (XLM-R, mBERT, MuRIL, and IndicBERT) across three complementary evaluation tasks.

## Purpose
The goal of this project is to provide:
1. A curated, reproducible Kannada WiC (Word-in-Context) benchmark (627 cleaned sentence pairs over 44 polysemous target words)
2. A comprehensive sense inventory (45 target words with 150 sense entries)
3. Multi-task evaluation of multilingual models for Kannada WSD
4. Deployment-oriented analysis for compute-constrained Indic-language settings

## Project Structure
```
Kannada-Indic-WSD-NLP/
├── data/                  # Cleaned dataset files
├── demoo/                 # FastAPI demo dashboard
├── docs/                  # Documentation and guides
├── models/                # (not committed) Downloaded Hugging Face models
├── paper access/         # Extended IEEE Access paper
├── paper cssitss/        # CSSITSS conference paper
├── report/               # Project report
├── download_models.py    # Script to download pre-trained models
├── requirements.txt     # Python dependencies
└── run_pipeline.py      # Main pipeline script
```

## Getting Started

### Prerequisites
- Python 3.10+
- Virtual environment (venv)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd Kannada-Indic-WSD-NLP
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv

   # Windows
   .\venv\Scripts\activate

   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pre-trained models**
   ```bash
   python download_models.py
   ```
   (Uses `HF_HUB_TOKEN` environment variable can be set for authenticated downloads from Hugging Face Hub.

## How to Use

### Run Demo Dashboard
Run the FastAPI-based demo dashboard:
```bash
cd demoo
uvicorn main:app --reload
```
Then open http://localhost:8000 in your browser.

## Model Comparison
| Model               | Parameters | WiC Accuracy | Gloss Ranking MRR |
|---------------------|------------|--------------|-----------------|
| mBERT               | 110M      | 53.92%      | 0.530         |
| MuRIL               | 180M      | 47.06%      | 0.619         |
| XLM-RoBERTa        | 278M      | 51.96%      | 0.513         |
| IndicBERT            | 33M       | 45.10%      | 0.416         |

## Citation
If you use this dataset or code, please cite:
```bibtex
@inproceedings{ranjan2026cross,
  title={Cross-lingual Sense Transfer with Low-Resource Indic Languages: A Reproducible Kannada Word Sense Disambiguation Benchmark},
  author={Ranjan, Aditya and Gupta, Arindam and Naidu, Gnanendra N and Vijayalakshmi, M N and Kumar, S Anupama},
  booktitle={CSSITSS},
  year={2026}
}
```
