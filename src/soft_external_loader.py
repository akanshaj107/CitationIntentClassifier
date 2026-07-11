import json
from datasets import Dataset


def load_soft_external_dataset(file_path):

    paper_ids = []
    texts = []

    with open(file_path, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            sample = json.loads(line)

            paper_ids.append(sample["paper_id"])

            texts.append(sample["citation_context"])

    return Dataset.from_dict({

        "paper_id": paper_ids,

        "citation_context": texts

    })