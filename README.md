# RAG Optimizer Tool 🧠📚

A production-style Retrieval-Augmented Generation (RAG) experimentation platform with:

- 🔎 Live RAG API (FastAPI)
- 📊 Optimization dashboard (Streamlit)
- ⚡ Two-stage optimization (fast retrieval screening -> full RAG evaluation)
- 🔁 Hot-swappable local corpora
- 🏆 Experiment tracking & leaderboard

This repository is designed to be cloned and run locally in minutes.

---

# 🔧 Installation (Recommended: Conda)

This project uses NumPy, SciPy, PyTorch, and Transformers.  
To avoid binary compatibility issues (especially on Windows), use the provided Conda environment.

### Create the environment

```
conda env create -f environment.yml
conda activate rag-optimizer
```

This installs:
- Python 3.10
- NumPy (pinned < 2.0)
- SciPy / scikit-learn (compatible versions)
- PyTorch (CPU build)
- Transformers stack
- FastAPI
- Streamlit

---

# Running the Application

You need to run two processes:\
1. The FastAPI backend\
2. The Streamlit dashboard

## 1. Prepare the dataset

For the sake of testing, try out the included Wikipedia Corpus from Huggingface (rag-mini-wikipedia)

To build a small local Wikipedia corpus + QA evaluation set (It is already available in the data folder):
```
python scripts/prepare_wikipedia_corpus.py
```

This creates:
```
data/
  wiki_docs.jsonl
  wiki_eval.jsonl
```

You can try out your own corpus by modifying `prepare_wikipedia_corpus.py` script to generate a suitable corpus of the JSONL format:
```json
{"id": 0, "text": "Document text here"}
```

## 2. Start the FastAPI Backend

