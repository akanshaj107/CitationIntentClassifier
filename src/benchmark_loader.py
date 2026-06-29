import json
from datasets import Dataset


def load_benchmark_dataset(
    benchmark_file,
    taxonomy
):
    """
    Load the benchmark dataset and return a HuggingFace Dataset.

    Parameters
    ----------
    benchmark_file : str
        Path to benchmark JSON.

    taxonomy : str
        One of:
            "scicite"
            "acl_arc"
            "soft_intent"
            "soft_content"

    Returns
    -------
    Dataset
        Dataset containing:
            text
            label
    """

    with open(benchmark_file, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    texts = []
    labels = []

    for sample in benchmark:

        text = sample["citation_context"]

        if taxonomy == "scicite":
            label = sample["scicite"]

        elif taxonomy == "acl_arc":
            label = sample["acl_arc"]

        elif taxonomy == "soft_intent":
            label = sample["soft_intent"]

        elif taxonomy == "soft_content":
            label = sample["soft_content"]

        else:
            raise ValueError(
                f"Unknown taxonomy: {taxonomy}"
            )

        # Skip unlabeled instances if any
        if label is None:
            continue

        texts.append(text)
        labels.append(label)

    dataset = Dataset.from_dict(
        {
            "string": texts,
            "label": labels
        }
    )

    return dataset