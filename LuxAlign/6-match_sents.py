import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import argparse

try:
    from LuxAlign.config import DATA_FOLDER
except ImportError:
    from config import DATA_FOLDER

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
BILINGUAL_LANG_PAIRS = [
    ("lb", "en"),
    ("lb", "fr"),
    ("en", "fr"),
]

ARTICLE_MATCH_FILE_TMPL = "{la}_{lb}_parallel_articles.xlsx"
SENTS_FILE_TMPL = "{lang}_sents.json"
SENT_EMB_FILE_TMPL = "{lang}_sentence_embeddings.npy"
BILINGUAL_OUT_TMPL = "{la}_{lb}_parallel_sents.csv"
MONO_OUT_TMPL = "{lang}_mono_parallel_sents.csv"

# Thresholds
ARTICLE_PAIR_MIN_SIM = 0.65            # further filtering on article-level pairs file
BILINGUAL_SENT_MIN_SIM = 0.70    # final bilingual sentence similarity threshold
MONO_MIN_SIM = 0.80
MONO_MAX_SIM = 0.95
BILINGUAL_LEN_RATIO_MAX = 1.5
MONO_LEN_RATIO_MAX = 2.0

# Monolingual advanced filters
MONO_SHARED_WORD_FRAC_MAX = 0.5        # each direction
MONO_MIN_UNIQUE_DIFF = 8               # min unique words not shared (each side) for logic

BENCHMARK_PATH = "benchmark_texts.txt"  # optional file with texts to exclude from final outputs (in order to avoid training/test leakage)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_sentence_embeddings(lang: str) -> np.ndarray:
    path = f"{DATA_FOLDER}/{SENT_EMB_FILE_TMPL.format(lang=lang)}"
    return np.load(path)

def load_sentences(lang: str) -> pd.DataFrame:
    path = f"{DATA_FOLDER}/{SENTS_FILE_TMPL.format(lang=lang)}"
    return pd.read_json(path, lines=True)

def load_article_matches(la: str, lb: str) -> pd.DataFrame:
    path = f"{DATA_FOLDER}/{ARTICLE_MATCH_FILE_TMPL.format(la=la, lb=lb)}"
    df = pd.read_excel(path)
    # expect columns: {la}_article_id, {lb}_article_id, cos_sim (from earlier pipeline)
    # filter further
    if "cos_sim" in df.columns:
        df = df[df["cos_sim"] >= ARTICLE_PAIR_MIN_SIM]
    return df[[f"{la}_article_id", f"{lb}_article_id"]].drop_duplicates()

def build_article_index(df: pd.DataFrame) -> dict:
    # returns mapping article_id -> np.array(row_indices)
    groups = df.groupby("article_id").indices
    return {k: np.fromiter(v, dtype=int) for k, v in groups.items()}

def safe_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.empty((0, 0))
    return cosine_similarity(a, b)

def length_ratio_ok(a: str, b: str, max_ratio: float) -> bool:
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return False
    return (la / lb) < max_ratio and (lb / la) < max_ratio

def low_word_overlap(a: str, b: str, threshold: float = 0.8) -> bool:
    w1 = set(a.lower().split())
    w2 = set(b.lower().split())
    if not w1 or not w2:
        return True
    inter = w1 & w2
    shorter = min(len(w1), len(w2))
    if shorter == 0:
        return True
    return (len(inter) / shorter) < threshold

def deduplicate_pairs(df: pd.DataFrame, col1: str, col2: str, sim_col: str = "cos_sim") -> pd.DataFrame:
    if sim_col not in df.columns:
        raise KeyError(f"{sim_col!r} column not found")

    out = df.copy()
    out["_row_order"] = np.arange(len(out))  # to preserve stable ordering

    def keep_best_per_group(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
        # Treat NaN scores as -inf so they don't win
        scores = frame[sim_col].fillna(float("-inf"))
        # Rank within each group (higher is better), ties -> first by original order
        rank = scores.groupby(frame[group_col], dropna=False).rank(method="first", ascending=False)
        kept = frame[rank == 1].copy()
        return kept

    # Stage 1: resolve duplicates by col1
    out = keep_best_per_group(out, col1)
    # Stage 2: resolve duplicates by col2 (on the already filtered frame)
    out = keep_best_per_group(out, col2)

    # Restore original order and cleanup
    out = out.sort_values("_row_order").drop(columns=["_row_order"])
    return out

def load_benchmark_texts() -> set:
    if not os.path.exists(f"{DATA_FOLDER}/{BENCHMARK_PATH}"):
        Warning(f"Benchmark file {BENCHMARK_PATH} not found, skipping benchmark filtering.")
        return set()
    with open(f"{DATA_FOLDER}/{BENCHMARK_PATH}", "r") as f:
        texts = f.read().splitlines()
    return set(t for t in texts if isinstance(t, str))

# ---------------------------------------------------------------------
# Bilingual sentence matching
# ---------------------------------------------------------------------
def process_bilingual_pair(la: str, lb: str) -> None:
    out_path = f"{DATA_FOLDER}/{BILINGUAL_OUT_TMPL.format(la=la, lb=lb)}"
    if os.path.exists(out_path):
        print(f"[SKIP] {out_path} exists.")
        return

    print(f"\n=== Bilingual sentence matching {la.upper()}-{lb.upper()} ===")
    # Load resources
    try:
        art_pairs = load_article_matches(la, lb)
    except FileNotFoundError:
        print(f"[WARN] Article match file missing for {la}-{lb}, skipping.")
        return
    if art_pairs.empty:
        print(f"[INFO] No article pairs after filtering for {la}-{lb}.")
        return

    sents_a = load_sentences(la)
    sents_b = load_sentences(lb)
    emb_a = load_sentence_embeddings(la)
    emb_b = load_sentence_embeddings(lb)

    if len(sents_a) != emb_a.shape[0] or len(sents_b) != emb_b.shape[0]:
        print(f"[ERROR] Sentence count and embeddings misaligned for {la}-{lb}.")
        return

    idx_a = build_article_index(sents_a)
    idx_b = build_article_index(sents_b)

    pairs = []
    for ida, idb in tqdm(art_pairs.values, desc=f"{la}-{lb} article pairs"):
        rows_a = idx_a.get(ida)
        rows_b = idx_b.get(idb)
        if rows_a is None or rows_b is None:
            continue
        sim_matrix = safe_cosine(emb_a[rows_a], emb_b[rows_b])
        if sim_matrix.size == 0:
            continue
        cand = np.where(sim_matrix >= BILINGUAL_SENT_MIN_SIM)
        for i, j in zip(*cand):
            sim = sim_matrix[i, j]
            sa = sents_a.iloc[rows_a[i]]["text"]
            sb = sents_b.iloc[rows_b[j]]["text"]
            if not length_ratio_ok(sa, sb, BILINGUAL_LEN_RATIO_MAX):
                continue
            if not low_word_overlap(sa, sb):
                continue
            pairs.append([ida, idb, sa, sb, float(sim)])

    if not pairs:
        print(f"[INFO] No bilingual sentence pairs found for {la}-{lb}.")
        return

    df_pairs = pd.DataFrame(
        pairs,
        columns=[f"{la}_article_id", f"{lb}_article_id", f"{la}_text", f"{lb}_text", "cos_sim"]
    )

    # Deduplicate
    df_pairs = deduplicate_pairs(df_pairs, f"{la}_text", f"{lb}_text")

    # Remove empty
    df_pairs = df_pairs[(df_pairs[f"{la}_text"].str.len() > 0) & (df_pairs[f"{lb}_text"].str.len() > 0)]

    # Benchmark exclusion
    bench_texts = load_benchmark_texts()
    if bench_texts:
        df_pairs = df_pairs[~df_pairs[f"{la}_text"].isin(bench_texts)]
        df_pairs = df_pairs[~df_pairs[f"{lb}_text"].isin(bench_texts)]

    # Final rename
    df_pairs = df_pairs.rename(columns={
        f"{la}_text": "text_1",
        f"{lb}_text": "text_2"
    })

    df_pairs.to_csv(out_path, index=False)
    print(f"[OK] Saved {len(df_pairs)} bilingual pairs -> {out_path}")

# ---------------------------------------------------------------------
# Monolingual (intra-article) sentence matching
# ---------------------------------------------------------------------
def process_monolingual(lang: str) -> None:
    out_base = f"{DATA_FOLDER}/{MONO_OUT_TMPL.format(lang=lang)}"
    if os.path.exists(out_base):
        print(f"[SKIP] {lang} monolingual outputs exist.")
        return

    print(f"\n=== Monolingual sentence matching {lang.upper()} ===")
    sents = load_sentences(lang)
    emb = load_sentence_embeddings(lang)
    if len(sents) != emb.shape[0]:
        print(f"[ERROR] Sentence count and embeddings misaligned for {lang}.")
        return

    idx_map = build_article_index(sents)
    collected = []
    for art_id, rows in tqdm(idx_map.items(), desc=f"{lang} articles"):
        if len(rows) < 2:
            continue
        sims = safe_cosine(emb[rows], emb[rows])
        # upper triangle only
        tri_i, tri_j = np.triu_indices(len(rows), k=1)
        for i, j in zip(tri_i, tri_j):
            val = sims[i, j]
            if val < MONO_MIN_SIM or val > MONO_MAX_SIM:
                continue
            t1 = sents.iloc[rows[i]]["text"]
            t2 = sents.iloc[rows[j]]["text"]
            if not length_ratio_ok(t1, t2, MONO_LEN_RATIO_MAX):
                continue
            if ":" in t1 or ":" in t2:
                continue
            collected.append([art_id, t1, t2, float(val)])

    if not collected:
        print(f"[INFO] No monolingual pairs for {lang}.")
        return

    mono_df = pd.DataFrame(collected, columns=["article_id", "text_1", "text_2", "cos_sim"])
    mono_df = deduplicate_pairs(mono_df, "text_1", "text_2")

    # Filtering
    def shared_ok(row) -> bool:
        w1 = set(row["text_1"].split())
        w2 = set(row["text_2"].split())
        inter = w1 & w2
        return (len(inter) / len(w1) < MONO_SHARED_WORD_FRAC_MAX) and (len(inter) / len(w2) < MONO_SHARED_WORD_FRAC_MAX)

    def unique_diff_ok(row) -> bool:
        w1 = set(row["text_1"].split())
        w2 = set(row["text_2"].split())
        inter = w1 & w2
        return (len(w1) - len(inter) > MONO_MIN_UNIQUE_DIFF) and (len(w2) - len(inter) > MONO_MIN_UNIQUE_DIFF)

    mono_df = mono_df[mono_df.apply(shared_ok, axis=1)]
    mono_df = mono_df[mono_df.apply(unique_diff_ok, axis=1)]

    mono_df.to_csv(out_base, index=False)
    print(f"[OK] Saved monolingual base: {len(mono_df)} -> {out_base}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Match parallel sentences (bilingual + monolingual).")
    parser.add_argument("--bi", nargs="*", help="Optional bilingual pairs like lb-en en-fr. Default: all.")
    parser.add_argument("--mono", nargs="*", help="Optional monolingual languages. Default: configured list.")
    args = parser.parse_args()

    bi_pairs = []
    if args.bi:
        for p in args.bi:
            if '-' in p:
                a, b = p.split('-', 1)
                bi_pairs.append((a, b))
    else:
        bi_pairs = BILINGUAL_LANG_PAIRS

    mono_langs = args.mono if args.mono else []

    for la, lb in bi_pairs:
        process_bilingual_pair(la, lb)
    for lang in mono_langs:
        process_monolingual(lang)

if __name__ == "__main__":
    main()
