from transformers import AutoTokenizer

MODEL_NAME = "allenai/scibert_scivocab_uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# -------------------------
# LABEL ENCODING
# -------------------------

label2id = {
    "background": 0,
    "method": 1,
    "result": 2
}


# -------------------------
# PREPROCESS FUNCTION
# -------------------------

def preprocess_function(example):

    encoding = tokenizer(
        example["string"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

    # Convert string label -> integer
    encoding["label"] = label2id[example["label"]]

    return encoding


# -------------------------
# TOKENIZE DATASET
# -------------------------

def tokenize_dataset(dataset):

    tokenized_dataset = dataset.map(
        preprocess_function
    )

    return tokenized_dataset