import json
from datasets import Dataset


def load_external_dataset(file_path):

    paper_ids = []
    texts = []

    with open(file_path, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            sample = json.loads(line)

            paper_ids.append(sample["paper_id"])

            texts.append(sample["citation_context"])

    dataset = Dataset.from_dict({

        "paper_id": paper_ids,

        "string": texts

    })

    return dataset