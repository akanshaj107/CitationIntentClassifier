from src.dataloader import load_soft, load_soft_cross_domain
from src.soft_model import (
    build_soft_model
)
from src.soft_external_loader import load_soft_external_dataset
from configs.current_config import (
    DATA_DIR, OUTPUT_DIR, INTENT_LABELS, CONTENT_LABELS, BENCHMARK_FILE, RUN_BENCHMARK,
    EXTERNAL_DATASET, APE_FILE, UNARXIV_FILE, UNARXIV_CS_FILE
)
from src.soft_preprocess import tokenize_soft_dataset
from src.soft_train import (
    train_soft_model    )
from src.soft_evaluate import (
    evaluate_soft_model
)
from src.external_loader import load_external_dataset
from src.soft_external_preprocess import (
    tokenize_soft_external_dataset
)

from src.soft_predict_external import (
    predict_soft_external
)
from src.benchmark_loader import load_soft_benchmark_dataset
dataset = load_soft(
    DATA_DIR
)

def run_external_prediction(
        input_file,
        output_name
):

    dataset = load_soft_external_dataset(
        input_file
    )

    tokenized_dataset = tokenize_soft_external_dataset(
        dataset
    )

    predict_soft_external(

        trainer,

        tokenized_dataset,

        INTENT_LABELS,

        CONTENT_LABELS,

        output_file=f"{OUTPUT_DIR}/predictions/{output_name}_predictions.jsonl"

    )

print(dataset["test"].features)


tokenized_dataset = tokenize_soft_dataset(
    dataset)

#to be removed after testing
#tokenized_dataset["train"] = (
#    tokenized_dataset["train"]
#    .select(range(100))
#)

#tokenized_dataset["validation"] = (
#    tokenized_dataset["validation"]
#    .select(range(20))
#)
#to be removed after testing

model = build_soft_model()


trainer = train_soft_model(
    model,
    tokenized_dataset,
    OUTPUT_DIR
)
#print(model)

evaluate_soft_model(

    trainer,

    tokenized_dataset["test"],

    INTENT_LABELS,

    CONTENT_LABELS,

    OUTPUT_DIR,
    dataset_name="test"
)

#Load Cross-Encoder model and evaluate on test set
cross_domain_dataset = load_soft_cross_domain(
    DATA_DIR
)

tokenized_cross_domain = (
    tokenize_soft_dataset(
        cross_domain_dataset
    )
)

evaluate_soft_model(
    trainer,
    tokenized_cross_domain,
    INTENT_LABELS,
    CONTENT_LABELS,
    f"{OUTPUT_DIR}/cross_domain",
    dataset_name="cross_domain"

)

if RUN_BENCHMARK:

    print("\nRunning Benchmark Evaluation...")

    benchmark_dataset = load_soft_benchmark_dataset(
        BENCHMARK_FILE
    )

    tokenized_benchmark = tokenize_soft_dataset(
        benchmark_dataset
    )

    evaluate_soft_model(
        trainer,
        tokenized_benchmark,
        INTENT_LABELS,
        CONTENT_LABELS,
        f"{OUTPUT_DIR}/benchmark",
        dataset_name="benchmark"
    )
    
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
    
