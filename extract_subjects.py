import json
from pathlib import Path

from tqdm import tqdm
import typer

from utils import Collection


def main(
    collection: Collection = typer.Option(
        ..., help="Collection to process, either 'tib-core-subjects' or 'all-subjects'"
    ),
    subjects_path: str = typer.Option(..., help="Path to the subjects file"),
):
    # llms4subjects/shared-task-datasets/GND/dataset/"
    root_subj_path = Path(subjects_path)

    if collection == Collection.tib_core_subjects:
        collection_name = "tib-core"
    else:
        collection_name = "all"

    with open(root_subj_path / f"GND-Subjects-{collection_name}.json", "r") as file:
        data = json.load(file)

    extracted_subjects = []
    for item in tqdm(data):
        code = item["Code"]
        name = item["Name"]
        classification_name = item["Classification Name"]
        description = " ".join([name, classification_name])
        if "Definition" in item:
            description = f"{description}: {item['Definition']}"
        extracted_subjects.append({"code": code, "description": description})

    with open(f"extracted_subjects_{collection.value}.jsonl", "w") as file:
        for subject in extracted_subjects:
            file.write(json.dumps(subject) + "\n")


if __name__ == "__main__":
    typer.run(main)
