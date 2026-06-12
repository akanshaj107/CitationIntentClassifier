from datasets import Value

from transformers import AutoTokenizer

from configs.current_config import (
    MODEL_NAME,
    MAX_LENGTH,
    INTENT_LABELS,
    CONTENT_LABELS
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


intent2id = {

    label: idx

    for idx, label in enumerate(
        INTENT_LABELS
    )
}


content2id = {

    label: idx

    for idx, label in enumerate(
        CONTENT_LABELS
    )
}


def preprocess_soft(
    example
):

    encoding = tokenizer(

        example[
            "citation_context"
        ],

        truncation=True,

        padding="max_length",

        max_length=MAX_LENGTH
    )

    encoding[
        "intent_label"
    ] = intent2id[
        example[
            "citation_function"
        ]
    ]

    encoding[
        "content_label"
    ] = content2id[
        example[
            "citation_object"
        ]
    ]

    return encoding


def tokenize_soft_dataset(
    dataset
):

    tokenized_dataset = dataset.map(
        preprocess_soft
    )

    tokenized_dataset = (
        tokenized_dataset.cast_column(
            "intent_label",
            Value("int64")
        )
    )

    tokenized_dataset = (
        tokenized_dataset.cast_column(
            "content_label",
            Value("int64")
        )
    )

    return tokenized_dataset