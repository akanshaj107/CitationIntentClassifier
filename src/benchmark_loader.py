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

        elif taxonomy == "aclarc":
            label = sample["aclarc"]
            
            label = sample["aclarc"].lower()

            label_mapping = {
                "compares": "compareorcontrast",
                "compare": "compareorcontrast",
                "compare_or_contrast": "compareorcontrast",
            }

            label = label_mapping.get(label, label)
            

        elif taxonomy == "soft_intent":
            label = sample["soft_intent"]

        elif taxonomy == "soft_content":
            label = sample["soft_content_type"]

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


def load_soft_benchmark_dataset(benchmark_file):

    with open(benchmark_file, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    texts = []
    citation_functions = []
    citation_objects = []

    for sample in benchmark:

        if sample["soft_intent"] is None:
            continue

        if sample["soft_content_type"] is None:
            continue

        text = sample["citation_context"]
        intent = sample["soft_intent"]
        content = sample["soft_content_type"]
        
        intent_mapping = {
        "JustifyDesignChoice": "Justify Design Choice",
        "HighlightLimitation": "Highlight Limitation",
        "SignalGap": "Signal Gap",
        "EvaluateAgainst": "Evaluate Against",
    }

        intent = intent_mapping.get(intent, intent)

        content_mapping = {
            "PerformedWork": "Performed Work",
            "ProducedResource": "Produced Resource"
        }

        content = content_mapping.get(content, content)

        texts.append(text)
        citation_functions.append(intent)
        citation_objects.append(content)

    return Dataset.from_dict({

        "citation_context": texts,
        "citation_function": citation_functions,
        "citation_object": citation_objects

    })