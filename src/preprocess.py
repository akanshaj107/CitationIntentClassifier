from transformers import AutoTokenizer
from datasets import Value
from configs.current_config import (
    MODEL_NAME,
    MAX_LENGTH
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# -------------------------
# PREPROCESS FUNCTION
# -------------------------

def preprocess_function(example, label2id):

    encoding = tokenizer(
        example["string"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    # Convert string label -> integer
    if "label" in example:
        encoding["label"] = label2id[example["label"].lower()]

    return encoding


# -------------------------
# TOKENIZE DATASET
# -------------------------

def tokenize_dataset(dataset, labels):

    label2id = {
        label: idx
        for idx, label in enumerate(labels)
    }

    tokenized_dataset = dataset.map(
        lambda example: preprocess_function(
            example,
            label2id
        )
    )
    tokenized_dataset = tokenized_dataset.cast_column(
        "label",
        Value("int64")
    )
    
    return tokenized_dataset