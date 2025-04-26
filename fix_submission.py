from pathlib import Path
import shutil

import typer
from tqdm import tqdm

from utils import Prefix, Collection, Split


def main(prefix: Prefix, collection: Collection, split: Split):
    submission_folder = f"submission_{prefix}_{collection}_{split}"
    fixed_submission_folder = f"fixed_{submission_folder}"
    test_dir = Path(
        f"../llms4subjects/shared-task-datasets/TIBKAT/{collection}/data/{split}"
    ).resolve()
    submission_dir = Path(submission_folder).resolve()
    fixed_submission_dir = Path(fixed_submission_folder).resolve()

    fixed_submission_dir.mkdir(parents=True, exist_ok=False)

    assert submission_dir.exists()
    test_file_paths = []
    for path in test_dir.rglob("*"):
        if path.is_file():
            test_file_paths.append(path.relative_to(test_dir).with_suffix(""))
    test_file_paths = set(test_file_paths)

    submission_file_paths = []
    for path in submission_dir.rglob("*"):
        if path.is_file():
            submission_file_paths.append(
                path.relative_to(submission_dir).with_suffix("")
            )
    submission_file_paths = set(submission_file_paths)

    files_to_keep = test_file_paths & submission_file_paths
    files_to_remove = submission_file_paths - test_file_paths
    assert len(submission_file_paths) == len(files_to_keep) + len(files_to_remove)

    for file_path in tqdm(files_to_keep):
        if "DS_Store" not in str(file_path):
            cur_full_path = submission_dir / file_path.with_suffix(".jsonld")
            new_full_path = fixed_submission_dir / file_path.with_suffix(".json")
            assert cur_full_path.exists(), f"File {cur_full_path} does not exist"
            new_full_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(cur_full_path, new_full_path)


if __name__ == "__main__":
    typer.run(main)
