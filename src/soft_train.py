import os

import torch

import torch.nn as nn

from transformers import (
    Trainer,
    TrainingArguments
)

from src.soft_preprocess import (
    tokenizer
)

from configs.current_config import (

    EPOCHS,

    LEARNING_RATE,

    BATCH_SIZE,

    WEIGHT_DECAY,

    SEED
)
from src.soft_metrics import (
    compute_soft_metrics
)


class SoftTrainer(
    Trainer
):

    def compute_loss(

        self,

        model,

        inputs,

        return_outputs=False,
        
        num_items_in_batch=None
    ):

        intent_labels = inputs.pop(
            "intent_label"
        )

        content_labels = inputs.pop(
            "content_label"
        )

        outputs = model(
            **inputs
        )

        intent_logits = outputs[
            "intent_logits"
        ]

        content_logits = outputs[
            "content_logits"
        ]

        loss_fn = (
            nn.CrossEntropyLoss()
        )

        intent_loss = loss_fn(

            intent_logits,

            intent_labels
        )

        content_loss = loss_fn(

            content_logits,

            content_labels
        )

        loss = (
            intent_loss
            +
            content_loss
        )

        if return_outputs:

            return (
                loss,
                outputs
            )

        return loss


def train_soft_model(

    model,

    tokenized_dataset,

    output_dir
):

    training_args = (
        TrainingArguments(

            output_dir=output_dir,

            learning_rate=
            LEARNING_RATE,

            per_device_train_batch_size=
            BATCH_SIZE,

            per_device_eval_batch_size=
            BATCH_SIZE,

            num_train_epochs=
            EPOCHS,

            weight_decay=
            WEIGHT_DECAY,
            eval_strategy="epoch",

            save_strategy="no",
            logging_strategy="epoch",

            report_to="tensorboard",

            seed=SEED,
            #to be removed after testing
            #max_steps=10,

            logging_dir=
            f"{output_dir}/logs"
        )
    )

    trainer = SoftTrainer(

        model=model,

        args=training_args,

        train_dataset=
        tokenized_dataset["train"],

        eval_dataset=
        tokenized_dataset[
            "validation"
        ],
        compute_metrics=
    compute_soft_metrics
    )

    trainer.train()
    #to be removed after testing
    #print("Training completed")

    model_dir = (
        f"{output_dir}/model"
    )

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    torch.save(
    model.state_dict(),
    f"{model_dir}/pytorch_model.bin"
)

    tokenizer.save_pretrained(
        model_dir
    )

    return trainer