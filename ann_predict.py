import json
import copy

from tqdm import tqdm
from annoy import AnnoyIndex
from safetensors.numpy import load_file
import typer

from utils import Collection, Split


def main(collection: Collection, split: Split):
    embedding_dim = 1024
    n_preds = 1000

    doc_embeddings = load_file(f"doc_embeddings_{collection}_{split}.safetensors")[
        "embeddings"
    ]
    docs = []
    with open(f"extracted_docs_{collection}_{split}.jsonl", "r") as f:
        for line in f:
            docs.append(json.loads(line))

    subjects_collection = (
        collection if "all" not in collection else collection.strip("-subjects")
    )

    subjects = []
    with open(f"extracted_subjects_{subjects_collection}.jsonl", "r") as f:
        for line in f:
            subjects.append(json.loads(line))

    subject_index = AnnoyIndex(embedding_dim, "angular")
    subject_index.load(f"subject_embeddings_{subjects_collection}.ann")

    predictions = copy.deepcopy(docs)
    for i, (doc, embedding) in enumerate(
        tqdm(zip(docs, doc_embeddings), total=len(docs))
    ):
        nearest_indices, distances = subject_index.get_nns_by_vector(
            embedding, n_preds, include_distances=True, search_k=50000
        )
        subject_codes = [subjects[idx]["code"] for idx in nearest_indices]
        predictions[i]["subjects"] = subject_codes

    with open(f"predictions_{collection}_{split}.jsonl", "w") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")


if __name__ == "__main__":
    typer.run(main)
