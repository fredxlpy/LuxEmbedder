# LuxAlign Creation Pipeline

This repository provides a lightweight pipeline to build multilingual (lb / en / fr) parallel dataset `LuxAlign`. The core preprocessing & alignment workflow is implemented in six Python scripts under `src/`, each exposing a minimal, consistent CLI via `argparse`.

## Pipeline Stages

1. `1-process_articles.py`  
	Clean news articles per language (HTML stripping, date & length filtering, language ID) -> `data_clean_{lang}.csv`.
2. `2-embed_articles.py`  
	Create article-level embeddings (OpenAI `text-embedding-3-small`) for a single language -> `{lang}_article_embeddings.npy` (batched checkpoints in `data/batched_embeddings`).
3. `3-match_articles.py`  
	Find cross-lingual comparable / parallel article pairs using cosine similarity within a publication date window -> `{la}_{lb}_parallel_articles.csv`.
4. `4-extract_sents.py`  
	Sentence segmentation + cleaning + language filtering from cleaned news articles -> `{lang}_sents.json` (JSONL with `article_id`, `text`).
5. `5-embed_sents.py`  
	Compute sentence embeddings (SentenceTransformer `sentence-transformers/LaBSE`) for sentences per language -> `{lang}_sentence_embeddings.npy` (chunks in `data/batched_sentence_embeddings`).
6. `6-match_sents.py`  
	Produce bilingual sentence pairs (from matched articles) and monolingual paraphrase-like pairs inside an article -> CSV outputs.

## Setup

Install dependencies:
```bash
# (optional) create & activate a fresh conda env
conda create -n luxalign python=3.11 -y
conda activate luxalign
pip install -r requirements.txt
```
Set your OpenAI key (required for stage 2):
```bash
export OPENAI_API_KEY=sk-YOURKEY
```

Set data folder path in `src/LuxAlign/config.py`.

## Usage Examples

1. Process raw articles
```bash
python LuxAlign/1-process_articles.py --lang en
python LuxAlign/1-process_articles.py --lang fr
python LuxAlign/1-process_articles.py --lang lb
```
2. Article embeddings
```bash
python LuxAlign/2-embed_articles.py --lang en --batch_size 300
python LuxAlign/2-embed_articles.py --lang fr --batch_size 300
python LuxAlign/2-embed_articles.py --lang lb --batch_size 300
```
3. Article matching
```bash
python LuxAlign/3-match_articles.py --pairs lb-en lb-fr
```
4. Sentence extraction
```bash
python LuxAlign/4-extract_sents.py --langs lb en fr
```
5. Sentence embeddings
```bash
python LuxAlign/5-embed_sents.py --langs lb en fr --batch_size 500
```
6. Sentence matching
```bash
python LuxAlign/6-match_sents.py --bi lb-en lb-fr
```

## Outputs

| Stage | Output Pattern |
|-------|----------------|
| 1 | data_clean_{lang}.csv |
| 2 | {lang}_article_embeddings.npy (+ batched files) |
| 3 | {la}_{lb}_parallel_articles.csv |
| 4 | {lang}_sents.json |
| 5 | {lang}_sentence_embeddings.npy (+ batched files) |
| 6 | {la}_{lb}_parallel_sents.csv / {lang}_mono_parallel_sents.csv |

