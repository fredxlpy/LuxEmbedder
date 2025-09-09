import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import argparse

try:
    from LuxAlign.config import DATA_FOLDER
except ImportError:
    from config import DATA_FOLDER
    
# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
EMBEDDING_FILE_TMPL = "{lang}_article_embeddings.npy"
CLEAN_FILE_TMPL = "data_clean_{lang}.csv"
OUTPUT_FILE_TMPL = "{la}_{lb}_parallel_articles.csv"

LANG_PAIRS = [
    ("lb", "en"),
    ("lb", "fr"),
    ("en", "fr"),
]

DATE_WINDOW_DAYS = 1          # +/- days for candidate selection
COS_SIM_THRESHOLD = 0.5       # Minimum similarity to keep a match
NORMALIZE_EMBEDDINGS = False  # Set True if raw embeddings are unnormalized and you want cosine faster

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_embeddings(lang: str) -> np.ndarray:
    path = f"{DATA_FOLDER}/{EMBEDDING_FILE_TMPL.format(lang=lang)}"
    arr = np.load(path)
    if NORMALIZE_EMBEDDINGS:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
    return arr

def load_data(lang: str) -> pd.DataFrame:
    path = f"{DATA_FOLDER}/{CLEAN_FILE_TMPL.format(lang=lang)}"
    df = pd.read_csv(path)
    # Ensure date column is datetime.date (no time component)
    df["public_date"] = pd.to_datetime(df["public_date"], errors="coerce").dt.date
    df = df.dropna(subset=["public_date"]).reset_index(drop=True)
    return df

def find_best_match(source_row: pd.Series,
                    target_df: pd.DataFrame,
                    src_emb: np.ndarray,
                    tgt_emb: np.ndarray) -> tuple[int | None, int | None, float | None]:
    """
    For one source article: restrict targets to +/- DATE_WINDOW_DAYS by date, then cosine similarity.
    Returns (source_index, target_index, similarity) or (None, None, None).
    """
    src_idx = source_row.name
    sdate = source_row["public_date"]
    min_date = sdate - pd.Timedelta(days=DATE_WINDOW_DAYS)
    max_date = sdate + pd.Timedelta(days=DATE_WINDOW_DAYS)
    # Filter candidates
    # Convert back to Timestamp for comparison ease
    candidate_mask = (pd.to_datetime(target_df["public_date"]) >= pd.to_datetime(min_date)) & (pd.to_datetime(target_df["public_date"]) <= pd.to_datetime(max_date))
    candidate_indices = target_df.index[candidate_mask]
    if len(candidate_indices) == 0:
        return (None, None, None)
    sims = cosine_similarity(
        src_emb[src_idx].reshape(1, -1),
        tgt_emb[candidate_indices]
    )[0]
    best_pos = sims.argmax()
    return (src_idx, candidate_indices[best_pos], float(sims[best_pos]))

def enrich_matches(match_df: pd.DataFrame,
                   df_a: pd.DataFrame,
                   df_b: pd.DataFrame,
                   la: str,
                   lb: str) -> pd.DataFrame:
    # Pull metadata columns
    cols = ["article_id", "title", "header", "text"]
    match_df[[f"{la}_article_id", f"{la}_title", f"{la}_header", f"{la}_text"]] = \
        df_a.loc[match_df[f"{la}_article_index"], cols].values
    match_df[[f"{lb}_article_id", f"{lb}_title", f"{lb}_header", f"{lb}_text"]] = \
        df_b.loc[match_df[f"{lb}_article_index"], cols].values
    return match_df

def process_pair(la: str, lb: str) -> None:
    out_path = f"{DATA_FOLDER}/{OUTPUT_FILE_TMPL.format(la=la, lb=lb)}"
    # Skip if already exists (optional)
    if Path(out_path).exists():
        print(f"[SKIP] {out_path} already exists.")
        return

    print(f"\n=== Matching {la.upper()} -> {lb.upper()} ===")
    emb_a = load_embeddings(la)
    emb_b = load_embeddings(lb)
    df_a = load_data(la)
    df_b = load_data(lb)

    matches = []
    for _, row in tqdm(df_a.iterrows(), total=len(df_a), desc=f"{la}->{lb}"):
        s_idx, t_idx, sim = find_best_match(row, df_b, emb_a, emb_b)
        if s_idx is not None:
            matches.append((s_idx, t_idx, sim))

    if not matches:
        print(f"No matches found for pair {la}-{lb}.")
        return

    match_df = pd.DataFrame(matches, columns=[f"{la}_article_index", f"{lb}_article_index", "cos_sim"])
    match_df = match_df[match_df["cos_sim"] >= COS_SIM_THRESHOLD].reset_index(drop=True)
    if len(match_df) == 0:
        print(f"No matches above threshold {COS_SIM_THRESHOLD} for {la}-{lb}.")
        return

    match_df = enrich_matches(match_df, df_a, df_b, la, lb)
    match_df.to_csv(out_path, index=False)
    print(f"Saved {len(match_df)} matches -> {out_path}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Match parallel / comparable articles across languages.")
    parser.add_argument("--pairs", nargs="*", help="Optional explicit pairs like lb-en en-fr (hyphen separated). If omitted, use default list.")
    args = parser.parse_args()

    if args.pairs:
        to_run = []
        for p in args.pairs:
            if '-' not in p:
                continue
            a, b = p.split('-', 1)
            to_run.append((a, b))
    else:
        to_run = LANG_PAIRS

    for la, lb in to_run:
        process_pair(la, lb)

if __name__ == "__main__":
    main()