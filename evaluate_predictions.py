import json
from pathlib import Path

import typer


def main(predictions_path: str, ground_truth_path: str, k: int, label_column: str):

    predictions_path = Path(predictions_path).resolve()
    ground_truth_path = Path(ground_truth_path).resolve()
    assert (
        predictions_path.exists() and ground_truth_path.exists()
    ), "Predictions or ground truth file does not exist"

    predictions = []
    with open(predictions_path, "r") as f:
        for line in f:
            predictions.append(json.loads(line))

    ground_truth = []
    with open(ground_truth_path, "r") as f:
        for line in f:
            ground_truth.append(json.loads(line))

    total_precision = 0
    total_recall = 0
    total_mrr = 0
    for pred, truth in zip(predictions, ground_truth):
        pred_subjects = set(pred[label_column][:k])
        true_subjects = set(truth[label_column])
        if not pred_subjects:
            continue
        else:
            precision = len(pred_subjects.intersection(true_subjects)) / len(
                pred_subjects
            )
            total_precision += precision

        recall = len(pred_subjects.intersection(true_subjects)) / min(
            len(true_subjects), k
        )
        total_recall += recall

        for rank, subject in enumerate(pred[label_column], 1):
            if subject in true_subjects:
                total_mrr += 1.0 / rank
                break

    avg_precision = total_precision / len(predictions)
    avg_recall = total_recall / len(predictions)
    if not avg_precision + avg_recall:
        f1 = 0
    else:
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    mrr = total_mrr / len(predictions)

    print(f"Precision@{k}: {avg_precision:.5f}")
    print(f"Recall@{k}: {avg_recall:.5f}")
    print(f"F1@{k}: {f1:.5f}")
    print(f"MRR@{k}: {mrr:.5f}")


if __name__ == "__main__":
    typer.run(main)
