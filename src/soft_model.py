import torch
import torch.nn as nn

from transformers import (
    AutoModel
)

from configs.current_config import (
    MODEL_NAME,
    INTENT_LABELS,
    CONTENT_LABELS
)


class SoftMultiTaskModel(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.encoder = (
            AutoModel.from_pretrained(
                MODEL_NAME
            )
        )

        hidden_size = (
            self.encoder.config.hidden_size
        )

        self.dropout = nn.Dropout(
            0.1
        )

        self.intent_classifier = (
            nn.Linear(
                hidden_size,
                len(INTENT_LABELS)
            )
        )

        self.content_classifier = (
            nn.Linear(
                hidden_size,
                len(CONTENT_LABELS)
            )
        )

    def forward(

        self,

        input_ids,

        attention_mask,

        token_type_ids=None,

        intent_label=None,

        content_label=None
    ):

        outputs = self.encoder(

            input_ids=input_ids,

            attention_mask=attention_mask,

            token_type_ids=token_type_ids
        )

        cls_output = (
            outputs.last_hidden_state[
                :, 0, :
            ]
        )

        cls_output = self.dropout(
            cls_output
        )

        intent_logits = (
            self.intent_classifier(
                cls_output
            )
        )

        content_logits = (
            self.content_classifier(
                cls_output
            )
        )

        return {

            "intent_logits":
            intent_logits,

            "content_logits":
            content_logits
        }


def build_soft_model():

    model = SoftMultiTaskModel()

    return model