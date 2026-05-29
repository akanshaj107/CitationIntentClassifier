from transformers import Trainer
from transformers import TrainingArguments
import os
from src.metrics import compute_metrics
from src.preprocess import tokenizer


def train_model(
    model,
    tokenized_dataset,
    output_dir
):

    training_args = TrainingArguments(
        output_dir=output_dir,

        learning_rate=2e-5,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        num_train_epochs=4,

        weight_decay=0.01,
        save_strategy = "no",

        logging_dir=f"{output_dir}/logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],

        compute_metrics=compute_metrics
    )

    trainer.train()
    model_dir = f"{output_dir}/model"

    os.makedirs(model_dir, exist_ok=True)

    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    return trainer