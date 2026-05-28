from datasets import load_dataset


def load_scicite(data_dir):

    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{data_dir}/train.jsonl",
            "validation": f"{data_dir}/dev.jsonl",
            "test": f"{data_dir}/test.jsonl"
        }
    )

    return dataset