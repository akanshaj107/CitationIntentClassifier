from src.dataloader import load_scicite
from src.predict import predict_dataset
from src.external_loader import load_external_dataset
from src.predict import predict_dataset
from src.preprocess import tokenize_dataset
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict_external import predict_external
from src.external_preprocess import tokenize_external_dataset
from src.benchmark_loader import load_benchmark_dataset
from configs.current_config import (
    LABELS,
    DATA_DIR,
    OUTPUT_DIR,
    RUN_BENCHMARK,
    BENCHMARK_FILE,
    EXTERNAL_DATASET,
    APE_FILE,
    UNARXIV_FILE,
    UNARXIV_CS_FILE
)


# ------------------------
# LABELS
# ------------------------

LABELS = [
    "background",
    "method",
    "result"
]

def run_external_prediction(input_file, output_name):

    dataset = load_external_dataset(input_file)

    tokenized_dataset = tokenize_external_dataset(dataset)

    predict_external(
        trainer,
        tokenized_dataset,
        LABELS,
        output_file=f"{OUTPUT_DIR}/predictions/{output_name}_predictions.jsonl"
    )

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
#tokenized_dataset["train"] = tokenized_dataset["train"].select(range(50))
#tokenized_dataset["validation"] = tokenized_dataset["validation"].select(range(50))
#tokenized_dataset["test"] = tokenized_dataset["test"].select(range(50))
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
    

#Ape Dataset evaluation
if EXTERNAL_DATASET == "ape":

    print("\nRunning APE prediction...")

    run_external_prediction(
        APE_FILE,
        "ape"
    )
    
#Unarchive Dataset evaluation
elif EXTERNAL_DATASET == "unarxiv":

    print("\nRunning unArXiv prediction...")
    run_external_prediction(
        UNARXIV_FILE,
        "unarxiv"
    )
    
elif EXTERNAL_DATASET == "unarxiv_cs":

    run_external_prediction(
        UNARXIV_CS_FILE,
        "unarxiv_cs"
    )
    
elif EXTERNAL_DATASET == "external":

    run_external_prediction(
        APE_FILE,
        "ape"
    )

    run_external_prediction(
        UNARXIV_FILE,
        "unarxiv"
    )
    
    run_external_prediction(
        UNARXIV_CS_FILE,
        "unarxiv_cs"
    )