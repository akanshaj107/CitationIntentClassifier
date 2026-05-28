import pandas as pd
from datasets import Dataset, DatasetDict


def load_jsonl(file_path):

    df = pd.read_json(
        file_path,
        lines=True
    )

    # Keep ONLY needed columns
    df = df[["string", "label"]]

    return Dataset.from_pandas(df)


def load_scicite(data_dir):

    train_dataset = load_jsonl(
        f"{data_dir}/train.jsonl"
    )

    validation_dataset = load_jsonl(
        f"{data_dir}/dev.jsonl"
    )

    test_dataset = load_jsonl(
        f"{data_dir}/test.jsonl"
    )

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })

    return dataset