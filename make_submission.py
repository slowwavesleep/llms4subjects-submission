import json

from tqdm import tqdm
from pathlib import Path

from utils import Prefix, Split, Collection


def main(
    prefix: Prefix,
    split: Split,
    collection: Collection,
    k: int = 50,
):
    submission_dir = Path(f"submission_{prefix}_{collection}_{split}").resolve()
    Path(submission_dir).mkdir(parents=True, exist_ok=False)
    predictions_path = f"predictions_{collection}_{split}.jsonl"

    with open(predictions_path, "r") as f:
        for line in tqdm(f):
            prediction = json.loads(line)
            file_name = prediction["file_name"]
            prediction["subjects"] = prediction["subjects"][:k]
            language = prediction["language"]
            record_type = prediction["record_type"]
            submission = {"dcterms:subject": prediction["subjects"]}
            file_dir = submission_dir / record_type / language
            file_dir.mkdir(parents=True, exist_ok=True)
            with open(file_dir / file_name, "w") as f:
                f.write(json.dumps(submission))
