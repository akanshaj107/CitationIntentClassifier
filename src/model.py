from transformers import AutoModelForSequenceClassification

from configs.scicite_config import MODEL_NAME


def build_model(num_labels):

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )

    return model