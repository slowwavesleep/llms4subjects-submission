import json
from pathlib import Path

from tqdm import tqdm
import typer

from utils import Split, Collection


def extract_article_info(json_data):
    entry = None
    for cur_entry in json_data["@graph"]:
        if "@type" in cur_entry and cur_entry["@type"] in (
            "bibo:Article",
            "bibo:Book",
            "bibo:Conference",
            "bibo:Report",
            "bibo:Thesis",
        ):
            entry = cur_entry

    if not entry:
        return None

    # Extract title and abstract
    title = entry.get("title", None)
    abstract = entry.get("abstract", "")

    # Extract and resolve dcterms:subjects using GND mappings
    subjects = []
    if "dcterms:subject" in entry:
        if isinstance(entry["dcterms:subject"], list):
            for item in entry["dcterms:subject"]:
                subjects.append(item["@id"])
        else:
            subjects.append(entry["dcterms:subject"]["@id"])

    if title:
        return {"title": title, "abstract": abstract, "subjects": subjects}

    else:
        return None


def main(
    split: Split = typer.Option(
        ..., help="Split to process, either 'train', 'test', or 'dev'"
    ),
    collection: Collection = typer.Option(
        ..., help="Collection to process, either 'tib-core-subjects' or 'all-subjects'"
    ),
    dataset_repo_path: str = typer.Option(..., help="Path to the dataset repository"),
):
    root_docs_path = Path(
        f"{dataset_repo_path}/shared-task-datasets/TIBKAT/{collection.value}/data/{split.value}"
    ).resolve()

    if root_docs_path.exists():
        print(f"Directory exists at: {root_docs_path}")
        print("\nContents:")
        for item in root_docs_path.iterdir():
            print(f"- {item}")
    else:
        print(f"Directory not found at: {root_docs_path}")

    extracted_docs = []
    all_files = list(root_docs_path.rglob("*jsonld"))
    for file_path in tqdm(all_files):
        with open(file_path, "r") as file:
            cur_data = json.load(file)
        extracted_data = extract_article_info(cur_data)
        if extracted_data:
            extracted_data["file_name"] = file_path.name
            extracted_data["language"] = str(file_path.parent.name)
            extracted_data["record_type"] = str(file_path.parent.parent.name)
            extracted_docs.append(extracted_data)

    print(len(extracted_docs))

    with open(f"extracted_docs_{collection.value}_{split.value}.jsonl", "w") as file:
        for doc in extracted_docs:
            file.write(json.dumps(doc) + "\n")


if __name__ == "__main__":
    typer.run(main)
