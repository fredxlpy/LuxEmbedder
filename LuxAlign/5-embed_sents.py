import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import argparse
from sentence_transformers import SentenceTransformer

try:
    from LuxAlign.config import DATA_FOLDER
except ImportError:
    from config import DATA_FOLDER

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
LANGS = ["lb", "en", "fr"]
MODEL_NAME ="sentence-transformers/LaBSE"
DEFAULT_BATCH_SIZE = 500         # Batching: API limits, memory, checkpointing
SENTS_FILE_TMPL = "{lang}_sents.json"
BATCH_DIR = f"{DATA_FOLDER}/batched_sentence_embeddings"
FINAL_EMB_TMPL = "{lang}_sentence_embeddings.npy"

os.makedirs(BATCH_DIR, exist_ok=True)

model = SentenceTransformer(MODEL_NAME).to("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_sentences(lang: str) -> list[str]:
    path = f"{DATA_FOLDER}/{SENTS_FILE_TMPL.format(lang=lang)}"
    df = pd.read_json(path, lines=True)
    return df["text"].astype(str).tolist()

def embed_batches(texts: list[str], lang: str, batch_size: int) -> None:
    num_docs = len(texts)
    if num_docs == 0:
        print(f"[{lang}] No sentences to embed.")
        return
    num_batches = (num_docs + batch_size - 1) // batch_size
    for i in tqdm(range(num_batches), desc=f"Embedding {lang}"):
        batch_file = f"{BATCH_DIR}/{lang}_sentence_embeddings_{i}.npy"
        if os.path.exists(batch_file):
            continue  # resume support
        start = i * batch_size
        end = min((i + 1) * batch_size, num_docs)
        batch_docs = texts[start:end]
        arr = model.encode(batch_docs)
        np.save(batch_file, arr)

def collect_batches(lang: str) -> np.ndarray:
    files = sorted(
        [f for f in os.listdir(BATCH_DIR) if f.startswith(f"{lang}_sentence_embeddings_") and f.endswith(".npy")],
        key=lambda x: int(x.rsplit("_", 1)[1].split(".")[0])
    )
    if not files:
        return np.empty((0, model.get_sentence_embedding_dimension()))
    arrays = [np.load(f"{BATCH_DIR}/{f}") for f in tqdm(files, desc=f"Loading {lang} batches")]
    return np.concatenate(arrays, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Compute sentence embeddings for one or more languages.")
    parser.add_argument("--langs", nargs="*", choices=LANGS, help="Subset of languages. Default: all.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    target_langs = args.langs if args.langs else LANGS
    for lang in target_langs:
        print(f"\n=== {lang.upper()} sentences ===")
        texts = load_sentences(lang)
        embed_batches(texts, lang, args.batch_size)
        embeddings = collect_batches(lang)
        out_path = f"{DATA_FOLDER}/{FINAL_EMB_TMPL.format(lang=lang)}"
        np.save(out_path, embeddings)
        print(f"[{lang}] Saved embeddings {embeddings.shape} -> {out_path}")

if __name__ == "__main__":
    main()
