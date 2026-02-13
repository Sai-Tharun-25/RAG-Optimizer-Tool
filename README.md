# RAG Optimizer Tool 🧠📚

A production-style Retrieval-Augmented Generation (RAG) experimentation platform with:

- 🔎 Live RAG API (FastAPI)
- 📊 Optimization dashboard (Streamlit)
- ⚡ Two-stage optimization (fast retrieval screening → full RAG evaluation)
- 🔁 Hot-swappable local corpora
- 🏆 Experiment tracking & leaderboard

This repository is designed to be cloned and run locally in minutes.

---

# 🛠 Installation

## Recommended: Conda (Stable & Reproducible)

This project uses NumPy, SciPy, PyTorch, and Transformers.  
To avoid binary compatibility issues (especially on Windows), use the provided Conda environment.

### 1️⃣ Create the environment

```bash
conda env create -f environment.yml
conda activate rag-optimizer
