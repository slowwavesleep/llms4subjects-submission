import json

from sentence_transformers import SentenceTransformer
from safetensors.torch import save_file
import typer

from utils import Collection


def main(collection: Collection):
    model_name = "intfloat/multilingual-e5-large-instruct"
    model = SentenceTransformer(model_name)

    subjects = []
    with open(f"extracted_subjects_{collection}.jsonl", "r") as f:
        for line in f:
            subjects.append(json.loads(line))

    if model_name == "intfloat/multilingual-e5-large":
        texts = [f"query: {subject['description']}" for subject in subjects]
    else:
        texts = [subject["description"] for subject in subjects]

    batch_size = 64

    all_embeddings = (
        model.encode(
            texts, convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size
        )
        .detach()
        .cpu()
    )

    tensors_dict = {"embeddings": all_embeddings}
    save_file(tensors_dict, f"subject_embeddings_{collection}.safetensors")


if __name__ == "__main__":
    typer.run(main)
