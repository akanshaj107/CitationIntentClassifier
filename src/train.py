from transformers import Trainer
from transformers import TrainingArguments
import os
from src.metrics import compute_metrics
from src.preprocess import tokenizer
from configs.current_config import (
    EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    WEIGHT_DECAY,
    OUTPUT_DIR,
    SEED
)

def train_model(
    model,
    tokenized_dataset,
    output_dir
):

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        logging_strategy="epoch",
        report_to="tensorboard",
        #to be removed after testing
        #max_steps=10,
        #For testing purposes.
        weight_decay=WEIGHT_DECAY,
        seed=SEED,
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