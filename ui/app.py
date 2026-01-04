import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.title("AutoRAG – RAG Optimization Tool (MVP)")

dataset_file = st.file_uploader("Upload dataset (.jsonl)", type=["jsonl"])

st.subheader("Search Space")
chunk_sizes = st.text_input("Chunk sizes (comma-separated)", "256,512")
overlaps = st.text_input("Overlaps (comma-separated)", "32,64")
ks = st.text_input("Top-k values (comma-separated)", "3,5,10")

if st.button("Run Experiment") and dataset_file:
    files = {"dataset_file": dataset_file}
    payload = {
        "chunk_size": [int(x.strip()) for x in chunk_sizes.split(",")],
        "overlap": [int(x.strip()) for x in overlaps.split(",")],
        "k": [int(x.strip()) for x in ks.split(",")],
    }
    # Streamlit has to send JSON separately
    res = requests.post(
        f"{API_BASE}/experiments",
        files=files,
        data={"search_space": ""}  # If needed adjust backend to accept form+json
    )
    st.write("Raw response:", res.json())
