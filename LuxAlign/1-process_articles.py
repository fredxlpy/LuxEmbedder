import json
import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from tqdm import tqdm
import re
import datetime
from huggingface_hub import hf_hub_download
import fasttext
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
import warnings
import argparse

try:
    from LuxAlign.config import DATA_FOLDER
except ImportError:
    from config import DATA_FOLDER

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

"""Article preprocessing script.

Now supports argparse:
    python src/1-process_articles.py --lang en
    python src/1-process_articles.py --lang lb

Outputs cleaned article CSV file for the selected language.
"""

# ---------------------------------------------------------------------
# Configuration (language-agnostic constants; language-specific paths are built at runtime)
# ---------------------------------------------------------------------

MIN_DATE_BY_LANG = {
    'lb': datetime.datetime(1999, 1, 1),
    'fr': datetime.datetime(2011, 9, 1),
    'en': datetime.datetime(2018, 1, 1)
}
END_DATE = datetime.datetime(2025, 8, 31)
MIN_CHARS = 100

TARGET_LANG_CODE = {
    'fr': 'fra_Latn',
    'en': 'eng_Latn',
    'lb': 'ltz_Latn'
}

HTML_SIMPLE_REPLACEMENTS = [
    (r'\[attachment.*?\]', ' '),
    (r'<strong>|</strong>|<b>|</b>', ''),
    (r'<i>|</i>|<em>|</em>', ''),
    (r'<li>|</li>', ' '),
    (r'<br ?/?>|</br>', ' '),
    (r'\[comments\]|\[gallery\]', '')
]
COMPILED_SUBS = [(re.compile(pat), repl) for pat, repl in HTML_SIMPLE_REPLACEMENTS]
WHITESPACE_RE = re.compile(r'\s+')

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_raw(path: str) -> pd.DataFrame:
    with open(path, 'r') as f:
        return pd.DataFrame(json.load(f))

def normalize_encoding(df: pd.DataFrame, cols) -> pd.DataFrame:
    # Fix encoding issues (latin1 -> utf8) gracefully
    df[cols] = df[cols].map(lambda x: x.encode('latin1').decode('utf8', errors='ignore') if isinstance(x, str) else x)
    return df

def strip_illegal(df: pd.DataFrame, cols) -> pd.DataFrame:
    df[cols] = df[cols].map(lambda x: ILLEGAL_CHARACTERS_RE.sub(r'', x) if isinstance(x, str) else x)
    return df

def clean_html(raw_html: str) -> str:
    if not isinstance(raw_html, str) or raw_html.strip() == '':
        return ''
    text = raw_html
    # Fast regex-based removals
    for pattern, repl in COMPILED_SUBS:
        text = pattern.sub(repl, text)
    # Parse residual HTML
    soup = BeautifulSoup(text, 'html.parser')
    # Keep only top-level relevant tags
    p_tags = [p for p in soup.children if getattr(p, 'name', None) in ['p', None, 'div', 'ul', 'a']]
    if p_tags:
        cleaned = ' '.join(p.get_text(strip=True) for p in p_tags)
    else:
        cleaned = soup.get_text(strip=True)
    # Normalize whitespace
    cleaned = WHITESPACE_RE.sub(' ', cleaned).strip()
    return cleaned

def detect_language(model, txt: str) -> str:
    if not txt:
        return ''
    return model.predict(txt)[0][0].replace('__label__', '')

def process_articles(lang: str) -> None:
    # Build input/output file paths
    input_file = f"{DATA_FOLDER}/raw_articles/news_articles_{lang}.json"
    output_file = f"{DATA_FOLDER}/data_clean_{lang}.csv"

    print(f"=== Processing raw articles for {lang.upper()} ===")

    # Load raw JSON -> DataFrame
    data = load_raw(input_file)

    # Parse publication date and drop rows with invalid dates
    data['public_date'] = pd.to_datetime(data['public_date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    data = data.dropna(subset=['public_date'])

    # Date range filtering per language
    start_date = MIN_DATE_BY_LANG[lang]
    data = data[(data['public_date'] >= start_date) & (data['public_date'] <= END_DATE)].reset_index(drop=True)

    # Normalize encoding for selected text fields
    text_cols = ['category_name', 'title', 'header', 'text']
    data = normalize_encoding(data, text_cols)

    # Clean raw HTML/body text
    clean_texts = [clean_html(t) for t in tqdm(data['text'], desc="Cleaning articles")]
    data['clean_text'] = clean_texts

    # Strip illegal Excel characters
    data = strip_illegal(data, ['title', 'header', 'text', 'clean_text'])

    # Drop very short or empty cleaned texts
    data = data[(data['clean_text'].str.len() >= MIN_CHARS) & (data['clean_text'] != '')]

    # Load FastText language ID model (downloaded from HF if not cached)
    model_path = hf_hub_download(repo_id="laurievb/OpenLID", filename="model.bin")
    model = fasttext.load_model(model_path)
    target_code = TARGET_LANG_CODE[lang]

    # Predict language for each cleaned article
    data['lang'] = [detect_language(model, t) for t in tqdm(data['clean_text'], desc="Language detection")]

    # Keep only rows matching desired language code
    data = data[data['lang'] == target_code]

    # Standardize column names: keep original + cleaned
    data = data.rename(columns={'text': 'text_original', 'clean_text': 'text'})

    # Write result to CSV
    data.to_csv(output_file, index=False)
    print(f"Saved {len(data)} rows -> {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process raw news articles into cleaned CSV file.")
    parser.add_argument("--lang", choices=["en", "fr", "lb"], default="en", help="Language to process")
    args = parser.parse_args()
    process_articles(args.lang)

if __name__ == "__main__":
    main()

