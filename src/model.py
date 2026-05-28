from transformers import AutoModelForSequenceClassification

MODEL_NAME = "allenai/scibert_scivocab_uncased"


def build_model(num_labels):

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )

    return model