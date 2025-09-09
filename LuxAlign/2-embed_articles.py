from openai import OpenAI
import pandas as pd
import numpy as np
from tqdm import tqdm
import tiktoken
import os
import argparse

try:
    from LuxAlign.config import DATA_FOLDER
except ImportError:
    from config import DATA_FOLDER

"""Embed cleaned articles for a single language.

Usage:
    python src/2-embed_articles.py --lang fr
"""

# ---------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------
MODEL_NAME = "text-embedding-3-small"
MAX_TOKENS = 8192
DEFAULT_BATCH_SIZE = 300
PRICE_PER_M_TOKEN = 0.02

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
ENCODING = tiktoken.get_encoding("cl100k_base")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_data(path: str) -> list[str]:
    df = pd.read_excel(path)
    return df["text"].astype(str).tolist()

def truncate_text_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
    tokens = ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return ENCODING.decode(tokens[:max_tokens])

def estimate_cost(docs: list[str]) -> tuple[int, float]:
    token_counts = [len(ENCODING.encode(d)) for d in docs]
    total = int(np.sum(token_counts))
    price = total * PRICE_PER_M_TOKEN / 1_000_000
    return total, price

def embed_batches(docs: list[str], lang: str, batch_size: int) -> None:
    # Batch = avoid API/request limits, huge RAM use, and loss on crash
    num_docs = len(docs)
    if num_docs == 0:
        print("No documents to embed.")
        return
    num_batches = (num_docs + batch_size - 1) // batch_size
    for i in tqdm(range(num_batches), desc=f"Embedding {lang}"):
        start = i * batch_size
        end = min((i + 1) * batch_size, num_docs)
        batch_docs = docs[start:end]
        response = client.embeddings.create(model=MODEL_NAME, input=batch_docs)
        batch_embeddings = np.array([r.embedding for r in response.data])
        np.save(f"{DATA_FOLDER}/batched_embeddings/{lang}_article_embeddings_{i}.npy", batch_embeddings)

def load_all_batches(lang: str) -> np.ndarray:
    prefix = f"{lang}_article_embeddings_"
    batch_dir = f"{DATA_FOLDER}/batched_embeddings"
    files = sorted(
        [f for f in os.listdir(batch_dir) if f.startswith(prefix) and f.endswith(".npy")],
        key=lambda x: int(x.rsplit("_", 1)[1].split(".")[0])
    )
    arrays = [np.load(os.path.join(batch_dir, f)) for f in files]
    return np.concatenate(arrays, axis=0) if arrays else np.empty((0,))

def main():
    parser = argparse.ArgumentParser(description="Embed cleaned article texts for a language.")
    parser.add_argument("--lang", choices=["en", "fr", "lb"], required=True)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    lang = args.lang
    batch_dir = f"{DATA_FOLDER}/batched_embeddings"
    os.makedirs(batch_dir, exist_ok=True)
    data_file = f"{DATA_FOLDER}/data_clean_{lang}.xlsx"
    final_path = f"{DATA_FOLDER}/{lang}_article_embeddings.npy"

    docs = load_data(data_file)
    docs = [truncate_text_tokens(d) for d in tqdm(docs, desc="Truncating")]
    total_tokens, est_price = estimate_cost(docs)
    print(f"[{lang}] Total tokens: {total_tokens} | Est cost: ${est_price:.4f}")
    embed_batches(docs, lang, args.batch_size)
    embeddings = load_all_batches(lang)
    np.save(final_path, embeddings)
    print(f"[{lang}] Saved embeddings shape {embeddings.shape} -> {final_path}")

if __name__ == "__main__":
    main()

