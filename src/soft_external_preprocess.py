from transformers import AutoTokenizer
from configs.current_config import *

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def preprocess_soft_external(example):

    encoding = tokenizer(
        example["citation_context"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    return encoding


def tokenize_soft_external_dataset(dataset):

    tokenized_dataset = dataset.map(
        preprocess_soft_external
    )

    return tokenized_dataset