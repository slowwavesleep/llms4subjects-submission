from annoy import AnnoyIndex
from safetensors.numpy import load_file
import typer


def build_annoy_index(embeddings_path: str, n_trees: int = 100):
    tensors = load_file(embeddings_path)
    embeddings = tensors["embeddings"]

    embedding_dim = embeddings.shape[1]

    index = AnnoyIndex(embedding_dim, "angular")

    for i, embedding in enumerate(embeddings):
        index.add_item(i, embedding)

    index.build(n_trees)

    index_path = embeddings_path.replace(".safetensors", ".ann")
    index.save(index_path)
    return index_path


def main(embeddings_path: str, n_trees: int = 10):
    index_path = build_annoy_index(embeddings_path, n_trees)
    print(f"Built and saved index to: {index_path}")


if __name__ == "__main__":
    typer.run(main)
