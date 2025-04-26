import math
import random
import json

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import (
    CEBinaryClassificationEvaluator,
)
from sentence_transformers.readers import InputExample
from tqdm import tqdm
from torch.utils.data import DataLoader
import typer

from utils import Collection, Split


def create_sentence_pairs(
    docs: list[dict[str, str]], subjects: dict[str, str]
) -> list[tuple[InputExample]]:
    subjects_list = list(subjects.keys())
    sentence_pairs = []
    for doc in tqdm(docs):
        pairs = []
        cur_subjects = doc["label"]
        sentence_1 = doc["text"]
        for subject in cur_subjects:
            if subject in subjects:
                pairs.append(
                    InputExample(texts=[sentence_1, subjects[subject]], label=1)
                )
        num_pairs = len(pairs)
        for _ in range(num_pairs):
            while random_subject := subjects_list[
                random.randint(0, len(subjects_list) - 1)
            ]:
                if random_subject not in cur_subjects:
                    cur_subjects.append(random_subject)
                    pairs.append(
                        InputExample(
                            texts=[sentence_1, subjects[random_subject]], label=0
                        )
                    )
                    break
        sentence_pairs.extend(pairs)
    return sentence_pairs


def main(collection: Collection, split: Split):
    train = []
    with open(f"docs_dataset_{collection.value}_{split.value}.json", "r") as f:
        for line in f:
            train.append(json.loads(line))

    dev = []
    with open(f"docs_dataset_{collection.value}_{split.value}.json", "r") as f:
        for line in f:
            dev.append(json.loads(line))

    subjects = {}
    with open(f"extracted_subjects_{collection.value}.jsonl", "r") as f:
        for line in f:
            cur_data = json.loads(line)
            subjects[cur_data["code"]] = cur_data["description"]

    sentence_pairs_train = create_sentence_pairs(train, subjects)
    sentence_pairs_dev = create_sentence_pairs(dev, subjects)

    train_batch_size = 64
    num_epochs = 10

    model_name = "microsoft/mdeberta-v3-base"
    model = CrossEncoder(model_name, num_labels=1, max_length=256, device="cuda")

    train_dataloader = DataLoader(
        sentence_pairs_train, shuffle=True, batch_size=train_batch_size
    )
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        sentence_pairs_dev, name=f"subjects-{collection.value}-{split.value}"
    )

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
    evaluation_steps = 50

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=evaluation_steps,
        warmup_steps=warmup_steps,
        output_path="./tmp-cross-encoder",
    )


if __name__ == "__main__":
    typer.run(main)
