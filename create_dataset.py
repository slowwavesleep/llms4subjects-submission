import json

from datasets import Dataset
import typer


def ds_from_docs(docs: list[dict[str, str]], test: bool = False) -> Dataset:
    cur_ds = []
    for item in docs:
        if test:
            item["subjects"] = []
        if item["subjects"] or test:
            if isinstance(item["title"], list):
                title = "\n".join(item["title"])
            else:
                title = item["title"]
            if isinstance(item["abstract"], list):
                abstract = "\n".join(item["abstract"])
            else:
                abstract = item["abstract"]
            cur_ds.append(
                {
                    "text": title + "\n\n" + abstract,
                    "label": item["subjects"],
                    "file_name": item["file_name"],
                    "language": item["language"],
                    "record_type": item["record_type"],
                }
            )
    return Dataset.from_list(cur_ds)


def main(
    collection: str = typer.Option(
        ...,
        choices=["tib-core-subjects", "all-subjects"],
        help="Collection to process, either 'tib-core-subjects' or 'all-subjects'",
    )
):

    dev = []
    with open(f"extracted_docs_{collection}_dev.jsonl", "r") as file:
        for line in file:
            data = json.loads(line)
        dev.append(data)

    train = []
    with open(f"extracted_docs_{collection}_train.jsonl", "r") as file:
        for line in file:
            data = json.loads(line)
            train.append(data)

    test = []
    with open(f"extracted_docs_{collection}_test.jsonl", "r") as file:
        for line in file:
            data = json.loads(line)
            test.append(data)

    dev_ds = ds_from_docs(dev)
    train_ds = ds_from_docs(train)
    test_ds = ds_from_docs(test, test=True)
    dev_ds.to_json(f"docs_dataset_{collection}_dev.json")
    train_ds.to_json(f"docs_dataset_{collection}_train.json")
    test_ds.to_json(f"docs_dataset_{collection}_test.json")


if __name__ == "__main__":
    typer.run(main)
