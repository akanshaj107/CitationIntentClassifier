from transformers import AutoTokenizer
from datasets import Dataset

from configs.current_config import *

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def preprocess_external(example):

    encoding = tokenizer(
        example["string"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    return encoding


def tokenize_external_dataset(dataset):

    tokenized_dataset = dataset.map(
        preprocess_external
    )

    return tokenized_dataset