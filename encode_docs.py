import json

from sentence_transformers import SentenceTransformer
from safetensors.torch import save_file
import typer

from utils import Collection, Split


def main(collection: Collection, split: Split):
    model_name = "intfloat/multilingual-e5-large-instruct"
    model = SentenceTransformer(model_name)

    docs = []
    with open(f"extracted_docs_{collection}_{split}.jsonl", "r") as f:
        for line in f:
            docs.append(json.loads(line))
    if model_name == "intfloat/multilingual-e5-large":
        texts = [f"passage: \n{doc['title']}\n\n{doc['abstract']}" for doc in docs]
    elif model_name == "intfloat/multilingual-e5-large-instruct":
        texts = [
            f"Instruct: Given the following title and abstract for the document, retrieve the relevant subjects classifying the document.\nQuery: Title: {doc['title']}\n Abstract: {doc['abstract']}"
            for doc in docs
        ]
    else:
        texts = [doc["title"] + "\n" + doc["abstract"] for doc in docs]

    batch_size = 64

    all_embeddings = model.encode(
        texts, convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size
    )

    tensors_dict = {"embeddings": all_embeddings}
    save_file(tensors_dict, f"doc_embeddings_{collection}_{split}.safetensors")


if __name__ == "__main__":
    typer.run(main)
