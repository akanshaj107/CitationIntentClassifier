from transformers import Trainer
from transformers import TrainingArguments

from src.metrics import compute_metrics


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

    return trainer