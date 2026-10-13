from sentence_transformers.evaluation import SentenceEvaluator
import logging
import os
import csv
from typing import List
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from contextlib import nullcontext
from sentence_transformers.evaluation import SentenceEvaluator
import logging
import os
import csv
from typing import List, Optional

logger = logging.getLogger(__name__)

class ContrastiveLossEvaluator(SentenceEvaluator):
    """
    Computes the mean squared error (x100) between the computed sentence embedding
    and some target sentence embedding.

    The MSE is computed between ||teacher.encode(source_sentences) - student.encode(target_sentences)||.

    For multilingual knowledge distillation (https://arxiv.org/abs/2004.09813), source_sentences are in English
    and target_sentences are in a different language like German, Chinese, Spanish...

    :param source_sentences: Source sentences are embedded with the teacher model
    :param target_sentences: Target sentences are ambedding with the student model.
    :param show_progress_bar: Show progress bar when computing embeddings
    :param batch_size: Batch size to compute sentence embeddings
    :param name: Name of the evaluator
    :param write_csv: Write results to CSV file
    :param truncate_dim: The dimension to truncate sentence embeddings to. `None` uses the model's current truncation
        dimension. Defaults to None.
    """

    def __init__(
        self,
        sentences1: List[str],
        sentences2: List[str],
        labels: List[int],
        margin: float = 0.5,
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = "",
        write_csv: bool = True,
        truncate_dim: Optional[int] = None,
    ):
        self.truncate_dim = truncate_dim

        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = torch.Tensor(labels)
        self.margin = margin
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name

        self.csv_file = "cl_evaluation_" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "CLoss"]
        self.write_csv = write_csv

    def __call__(self, model: SentenceTransformer, output_path, epoch=-1, steps=-1):
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(self.truncate_dim):

            embeddings1 = model.encode(
                self.sentences1,
                show_progress_bar=self.show_progress_bar,
                batch_size=self.batch_size,
                convert_to_numpy=True,
            )

            embeddings2 = model.encode(
                self.sentences2,
                show_progress_bar=self.show_progress_bar,
                batch_size=self.batch_size,
                convert_to_numpy=True,
            )

        distances = 1 - F.cosine_similarity(torch.Tensor(embeddings1), torch.Tensor(embeddings2))
        losses = 0.5 * (
            self.labels.float() * distances.pow(2) + (1 - self.labels).float() * F.relu(self.margin - distances).pow(2)
        )

        loss = losses.mean()

        logger.info(f"Loss evaluation (lower = better) on the {self.name} dataset{out_txt}:")
        logger.info("Contrastive Loss:\t{:4f}".format(loss))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline="", mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, loss])

        return -loss  # Return negative score as SentenceTransformers maximizes the performance
