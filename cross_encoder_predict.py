import json
from pathlib import Path

import torch
from tqdm import tqdm
import typer
from sentence_transformers.cross_encoder import CrossEncoder
from annoy import AnnoyIndex
from safetensors.numpy import load_file

from utils import Collection, Split


def main(
    collection: Collection = typer.Option(
        ..., help="Collection to process, either 'tib-core-subjects' or 'all-subjects'"
    ),
    split: Split = typer.Option(
        ..., help="Split to process, either 'train', 'test', or 'dev'"
    ),
):

    model = CrossEncoder(
        "./tmp-cross-encoder", num_labels=1, max_length=256, device="cuda"
    )

    embedding_dim = 1024
    n_preds_ann = 512
    n_preds_ce = 100
    batch_size = 512
    in_path = Path(f"docs_dataset_{collection}_{split}.json").resolve()
    out_path = Path(f"cross_encoder_predictions_{collection}_{split}.jsonl").resolve()

    subjects_collection = (
        collection if "all" not in collection else collection.strip("-subjects")
    )
    subject_index = AnnoyIndex(embedding_dim, "angular")
    subject_index.load(f"subject_embeddings_{subjects_collection}.ann")

    subjects = []
    with open(f"extracted_subjects_{subjects_collection}.jsonl", "r") as f:
        for line in f:
            subjects.append(json.loads(line))

    doc_embeddings = load_file(f"doc_embeddings_{collection}_{split}.safetensors")[
        "embeddings"
    ]

    if out_path.exists():
        with open(out_path, "r") as f:
            start_position = sum(1 for _ in f)
    else:
        start_position = 0

    with open(in_path, "r") as file_in, open(out_path, "a") as file_out:
        total_docs = sum(1 for _ in file_in)
        file_in.seek(0)
        print(f"Starting at position {start_position} of {total_docs}")
        for i, line in tqdm(enumerate(file_in), total=total_docs):
            if i >= start_position:
                cur_doc = json.loads(line)
                cur_embedding = doc_embeddings[i]
                nearest_indices, _ = subject_index.get_nns_by_vector(
                    cur_embedding, n_preds_ann, include_distances=True, search_k=50000
                )
                candidates = [subjects[index] for index in nearest_indices]
                cur_pairs = []
                for candidate in candidates:
                    cur_pairs.append((cur_doc["text"], candidate["description"]))
                ce_predictions = (
                    model.predict(
                        cur_pairs,
                        show_progress_bar=False,
                        batch_size=batch_size,
                        convert_to_tensor=True,
                    )
                    .detach()
                    .cpu()
                )
                ce_predictions = torch.argsort(
                    ce_predictions, descending=True
                ).tolist()[:n_preds_ce]
                ce_predicted_codes = [candidates[i]["code"] for i in ce_predictions]
                cur_doc["subjects"] = ce_predicted_codes
                file_out.write(json.dumps(cur_doc) + "\n")
                if i % 100 == 0:
                    file_out.flush()
