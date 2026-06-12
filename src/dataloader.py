import pandas as pd
from datasets import Dataset, DatasetDict
from pathlib import Path
from scripts.build_acl_arc_dataset import build_acl_arc_dataset
from scripts.build_soft_dataset import (
    build_soft_dataset
)

def load_jsonl(file_path):

    df = pd.read_json(
        file_path,
        lines=True
    )

    # Keep ONLY needed columns
    #df = df[["string", "label"]]

    return Dataset.from_pandas(df)

def load_scicite(data_dir):

    return load_dataset_from_jsonl(
        data_dir
    )

def load_acl_arc(data_dir):

    train_file = Path(
        f"{data_dir}/train.jsonl"
    )

    dev_file = Path(
        f"{data_dir}/dev.jsonl"
    )

    test_file = Path(
        f"{data_dir}/test.jsonl"
    )

    if not (
        train_file.exists()
        and dev_file.exists()
        and test_file.exists()
    ):

        print(
            "Processed ACL-ARC dataset not found."
        )

        print(
            "Building dataset from raw files..."
        )

        build_acl_arc_dataset()

    return load_dataset_from_jsonl(
        data_dir
    )
    
def load_soft(data_dir):

    train_file = Path(
        f"{data_dir}/train.jsonl"
    )

    dev_file = Path(
        f"{data_dir}/dev.jsonl"
    )

    test_file = Path(
        f"{data_dir}/test.jsonl"
    )

    if not (
        train_file.exists()
        and dev_file.exists()
        and test_file.exists()
    ):

        print(
            "Building SOFT dataset..."
        )

        build_soft_dataset()

    return DatasetDict({

        "train":
        load_jsonl(
            f"{data_dir}/train.jsonl"
        ),

        "validation":
        load_jsonl(
            f"{data_dir}/dev.jsonl"
        ),

        "test":
        load_jsonl(
            f"{data_dir}/test.jsonl"
        )
    })

def load_soft_cross_domain(
    data_dir
):

    cross_file = Path(
        f"{data_dir}/cross_domain.jsonl"
    )

    if not cross_file.exists():

        print(
            "Cross-domain dataset not found."
        )

        print(
            "Building SOFT dataset..."
        )

        build_soft_dataset()

    cross_domain_dataset = (
        load_jsonl(
            f"{data_dir}/cross_domain.jsonl"
        )
    )

    return cross_domain_dataset

def load_dataset_from_jsonl(data_dir):

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