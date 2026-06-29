from src.dataloader import load_scicite
from src.preprocess import tokenize_dataset
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.benchmark_loader import load_benchmark_dataset
from configs.current_config import (
    LABELS,
    DATA_DIR,
    OUTPUT_DIR,
    RUN_BENCHMARK,
    BENCHMARK_FILE
)


# ------------------------
# LABELS
# ------------------------

LABELS = [
    "background",
    "method",
    "result"
]


# ------------------------
# LOAD DATA
# ------------------------

dataset = load_scicite(DATA_DIR)


# ------------------------
# TOKENIZE
# ------------------------

tokenized_dataset = tokenize_dataset(
    dataset, LABELS
)

#----To be removed after testing----#
tokenized_dataset["train"] = tokenized_dataset["train"].select(range(50))
tokenized_dataset["validation"] = tokenized_dataset["validation"].select(range(50))
tokenized_dataset["test"] = tokenized_dataset["test"].select(range(50))
# For testing purposes, we are selecting a subset of the dataset.

# ------------------------
# BUILD MODEL
# ------------------------

model = build_model(
    num_labels=len(LABELS)
)


# ------------------------
# TRAIN
# ------------------------

trainer = train_model(
    model=model,
    tokenized_dataset=tokenized_dataset,
    output_dir=OUTPUT_DIR
)


# ------------------------
# EVALUATE
# ------------------------

evaluate_model(
    trainer,
    tokenized_dataset["test"],
    LABELS,
    output_dir=OUTPUT_DIR,
    dataset_name="test"
)

#Benchmark evaluation
if RUN_BENCHMARK:

    print("\nRunning Benchmark Evaluation...")

    benchmark_dataset = load_benchmark_dataset(
        BENCHMARK_FILE,
        taxonomy="scicite"
    )

    benchmark_dataset = tokenize_dataset(
        benchmark_dataset,
        LABELS
    )

    evaluate_model(
        trainer,
        benchmark_dataset,
        LABELS,
        f"{OUTPUT_DIR}/benchmark",
        dataset_name="benchmark"
    )
