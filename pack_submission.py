import zipfile
from pathlib import Path

import typer


def main(
    team_name: str = "tartunlp",
):
    # team_name = "tartunlp"
    core = "tib-core-subjects"
    all = "all-subjects"

    dir_to_run = {
        f"fixed_submission_bi-encoder_{all}_test": Path(f"{all}/run_1"),
        f"fixed_submission_bi-encoder_{core}_test": Path(f"{core}/run_1"),
        f"fixed_submission_cross_encoder_{all}_test": Path(f"{all}/run_2"),
        f"fixed_submission_cross_encoder_{core}_test": Path(f"{core}/run_2"),
    }

    with zipfile.ZipFile(f"{team_name}.zip", "w") as zip_file:
        for source_dir, target_path in dir_to_run.items():
            source_path = Path(source_dir)
            if not source_path.exists():
                print(f"Warning: Source directory {source_path} does not exist")
                continue

            for file_path in source_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(source_path)
                    archive_path = target_path / relative_path
                    zip_file.write(file_path, archive_path)


if __name__ == "__main__":
    typer.run(main)
