import argparse

import pandas as pd
from datasets import load_dataset
from sentence_transformers import InputExample, SentenceTransformer, losses
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import set_seed

from utils import ContrastiveLossEvaluator

SUBSETS = ["lb-en", "lb-fr"]

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-6)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=3)
args = parser.parse_args()

set_seed(args.seed)


def load_subset(name):
    """Load one LuxAlign subset as a DataFrame with columns text_1, text_2."""
    src, tgt = name.split("-")  # e.g. "lb", "en"
    df = load_dataset("fredxlpy/LuxAlign", name, split="train").to_pandas()
    return df.rename(columns={src: "text_1", tgt: "text_2"})[["text_1", "text_2"]]


def add_negatives(df, seed):
    """Label all pairs as positive and add one shuffled negative per pair."""
    pos = df.assign(label=1)
    neg = df.assign(label=0)
    neg["text_2"] = df["text_2"].sample(frac=1, random_state=seed).values
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed)


# Load data, split each language pair, add negatives
train_parts, eval_parts = [], []
for subset in SUBSETS:
    df = load_subset(subset)
    train_df, eval_df = train_test_split(df, test_size=0.02, random_state=args.seed)
    train_parts.append(add_negatives(train_df, args.seed))
    eval_parts.append(add_negatives(eval_df, args.seed))

train_data = pd.concat(train_parts, ignore_index=True)
eval_data = pd.concat(eval_parts, ignore_index=True)
print("Number of training samples:", len(train_data))

# Model (truncation to 512 tokens is handled by max_seq_length)
model = SentenceTransformer("sentence-transformers/LaBSE")
model.max_seq_length = 512

train_examples = [
    InputExample(texts=[t1, t2], label=label)
    for t1, t2, label in zip(train_data["text_1"], train_data["text_2"], train_data["label"])
]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
train_loss = losses.ContrastiveLoss(model)

evaluator = ContrastiveLossEvaluator(
    sentences1=eval_data["text_1"].tolist(),
    sentences2=eval_data["text_2"].tolist(),
    labels=eval_data["label"].tolist(),
)


def callback(score, epoch, steps):
    print(f"Score: {score} at epoch {epoch} and step {steps}")


output_path = f"models/Luxembedder_{args.batch_size}-{args.lr}-{args.seed}"
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=args.epochs,
    evaluator=evaluator,
    evaluation_steps=500,
    save_best_model=True,
    output_path=output_path,
    scheduler="constantlr",
    optimizer_params={"lr": args.lr},
    callback=callback,
    show_progress_bar=True,
)