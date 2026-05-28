from transformers import AutoTokenizer

MODEL_NAME = "allenai/scibert_scivocab_uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(example):

    return tokenizer(
        example["string"],
        truncation=True,
        padding="max_length",
        max_length=256
    )


def tokenize_dataset(dataset):

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True
    )

    return tokenized_dataset