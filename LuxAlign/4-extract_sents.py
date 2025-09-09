import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk
import re
import fasttext
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from pathlib import Path
from typing import Iterable
import argparse

try:
    from LuxAlign.config import DATA_FOLDER
except ImportError:
    from config import DATA_FOLDER

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
LANGS = ["en", "lb", "fr"]
MIN_CHAR_LEN = 10
MIN_WORDS = 4
OUT_FILE_TMPL = "{lang}_sents.json"

LID_TARGET = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "lb": "ltz_Latn",
}

NLTK_LANG = {
    "en": "english",
    "fr": "french",
    "lb": "german",  # lb unsupported => fallback (close delimiter behavior)
}

# Precompiled regex patterns
EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
BRACKET_TAG_RE = re.compile(r'\[.*?\]')
AT_TAG_RE = re.compile(r'\[at\]')
PHONE_MULTI_RE = re.compile(
    '|'.join([
        r'\(?\+352\)?[\s-]*\d+[\s-]*\d+[\s-]*\d+',
        r'\b\d{2}[ -]\d{2}[ -]\d{2}[ -]\d{2}\b',
        r'\b\d{2}[ -]\d{2}[ -]\d{2}[ -]\d{1}\b',
        r'\b\d{2}[ -]\d{2}[ -]\d{2}\b',
        r'\b\d{3}[ -]\d{3}[ -]\d{3}\b',
        r'\b\d{2}[ -]\d{3}[ -]\d{3}\b',
        r'\b\d{3}[ -]\d{2}[ -]\d{2}[ -]\d{2}\b',
        r'\b\d{9}\b',
        r'\b\d{6}\b',
        r'\b\d{3}[ -]\d{3}[ -]\d{2}[ -]\d{2}\b',
        r'\b\d{2}[ -]\d{2}[ -]\d{2}[ -]\d{4}\b',
        r'\b\d{4}[ -]\d{4}\b',
        r'\b\d{3}[ -]\d{4}\b',
        r'\b\d{5}[ -]\d{4}\b',
        r'\b\d{3}[ -]\d{2}[ -]\d{4}\b',
        r'\b\d{3}[ -]\d{5}\b',
    ])
)
URL_PRE_INSERT_RE = re.compile(r'(?<=[a-zA-Z])(?=(?:http[s]?://|www\.))')
URL_HTTP_RE = re.compile(r'http[s]?://[a-zA-Z0-9$-_@.&+!*(),%]+')
URL_WWW_RE = re.compile(r'www\.[a-zA-Z0-9$-_@.&+!*(),%]+')
MIS_SPLIT_END_RE = re.compile(r'.*(?<!\d)[0-9]{1,2}\.$')

# ---------------------------------------------------------------------
# Sentence post-splitting helper (legacy behavior preserved)
# ---------------------------------------------------------------------
def split_sentences_with_dot(sentences: list[str]) -> list[str]:
    result = []
    pattern = re.compile(r'(?<=[a-z]{2})\.(?=[A-Z][a-z])|(?<=[a-z]{2})\.(?=[A-Z]{3})')
    for sentence in sentences:
        if any(tok in sentence for tok in ['[at]', '@', 'www', '...']):
            result.append(sentence)
            continue
        matches = list(pattern.finditer(sentence))
        if matches:
            parts = []
            last = 0
            for m in matches:
                split_idx = m.end() - 1
                parts.append(sentence[last:split_idx + 1])
                last = split_idx + 1
            parts.append(sentence[last:].strip())
            result.extend(parts)
        else:
            result.append(sentence)
    return result

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_articles(lang: str) -> pd.DataFrame:
    dfs = []
    fname = f"{DATA_FOLDER}/data_clean_{lang}.xlsx"
    df = pd.read_excel(fname)
    if "text" not in df.columns and "clean_text" in df.columns:
        df = df.rename(columns={"clean_text": "text"})
    dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No input files found for {lang} in {DATA_FOLDER}")
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["article_id"]).reset_index(drop=True)
    # Basic normalization
    for col in ["title", "header", "text"]:
        if col in df_all.columns:
            df_all[col] = df_all[col].astype(str).str.replace('\n', ' ').str.strip()
    # Ensure title ends with punctuation
    if "title" in df_all.columns:
        df_all["title"] = df_all["title"].apply(lambda x: x + '.' if x and x[-1] not in ".!?" else x)
    return df_all

# ---------------------------------------------------------------------
# Sentence extraction
# ---------------------------------------------------------------------
def sentence_iter(row: pd.Series, lang: str) -> Iterable[tuple[str, str]]:
    tokenizer_lang = NLTK_LANG[lang]
    for col in ["title", "header", "text"]:
        if col not in row or not isinstance(row[col], str) or not row[col].strip():
            continue
        try:
            sents = sent_tokenize(row[col], language=tokenizer_lang)
        except Exception:
            continue
        sents = split_sentences_with_dot(sents)
        for s in sents:
            yield row["article_id"], s

# ---------------------------------------------------------------------
# Cleaning / filtering
# ---------------------------------------------------------------------
def clean_sentence(text: str) -> str:
    # Replace markers
    text = AT_TAG_RE.sub('@', text)
    text = BRACKET_TAG_RE.sub('', text)
    text = EMAIL_RE.sub('[email]', text)
    # URLs (ensure space before)
    text = URL_PRE_INSERT_RE.sub(' ', text)
    text = URL_HTTP_RE.sub('[url]', text)
    text = URL_WWW_RE.sub('[url]', text)
    # Phones
    text = PHONE_MULTI_RE.sub('[phone]', text)
    # Strip stray single leading / trailing quote
    if text.startswith('"') and text.count('"') == 1:
        text = text[1:]
    if text.endswith('"') and text.count('"') == 1:
        text = text[:-1]
    return text.strip()

def filter_basic(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["text"].str.len() > MIN_CHAR_LEN]
    df = df[df["text"].str.split().map(len) >= MIN_WORDS]
    df = df[~df["text"].str.match(MIS_SPLIT_END_RE)]
    return df

# ---------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------
def detect_language_filter(df: pd.DataFrame, lid_model, target_code: str) -> pd.DataFrame:
    langs = []
    for txt in tqdm(df["text"], desc="Language filter"):
        if not txt:
            langs.append(False)
            continue
        pred = lid_model.predict(txt)[0][0].replace("__label__", "")
        langs.append(pred == target_code)
    return df[pd.Series(langs, index=df.index)]

# ---------------------------------------------------------------------
# Per-language processing
# ---------------------------------------------------------------------
def process_language(lang: str, lid_model) -> None:
    print(f"\n=== Processing {lang.upper()} ===")
    df_articles = load_articles(lang)
    rows = []
    for _, row in tqdm(df_articles.iterrows(), total=len(df_articles), desc="Splitting"):
        rows.extend(sentence_iter(row, lang))
    if not rows:
        print("No sentences extracted.")
        return
    sents_df = pd.DataFrame(rows, columns=["article_id", "text"])
    # Clean
    sents_df["text"] = sents_df["text"].map(clean_sentence)
    sents_df = filter_basic(sents_df)
    # Language filter
    target_code = LID_TARGET[lang]
    sents_df = detect_language_filter(sents_df, lid_model, target_code)
    # Final prune after LID
    sents_df = filter_basic(sents_df)
    out_path = f"{DATA_FOLDER}/{OUT_FILE_TMPL.format(lang=lang)}"
    sents_df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    print(f"Saved {len(sents_df)} sentences -> {out_path}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract and clean sentences per language from cleaned article files.")
    parser.add_argument("--langs", nargs="*", choices=LANGS, help="Subset of languages. Default: all.")
    args = parser.parse_args()

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    model_path = hf_hub_download(repo_id="laurievb/OpenLID", filename="model.bin")
    lid_model = fasttext.load_model(model_path)
    target_langs = args.langs if args.langs else LANGS
    for lang in target_langs:
        process_language(lang, lid_model)

if __name__ == "__main__":
    main()

