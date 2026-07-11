# Dataset
DATASET_NAME = "SOFT"
DATA_DIR = "data/soft"
#benchmark
BENCHMARK_FILE = "data/benchmark/benchmark.json"

RUN_BENCHMARK = False
EXTERNAL_DATASET = "external"

# Options: None, "benchmark", "ape", "unarxiv", "external", "unarxiv_cs"

APE_FILE = "data/ape/ape_citations.jsonl"

UNARXIV_FILE = "data/external/econ_citations.jsonl"

UNARXIV_CS_FILE = "data/external/cs_citations.jsonl"
# Model
MODEL_NAME = "bert-base-uncased"

# Intent Labels
INTENT_LABELS = [
    "Contextualize",
    "Signal Gap",
    "Highlight Limitation",
    "Justify Design Choice",
    "Use",
    "Modify",
    "Evaluate Against"
]

# Content Labels
CONTENT_LABELS = [
    "Performed Work",
    "Discovery",
    "Produced Resource"
]

# Training
EPOCHS = 4
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
MAX_LENGTH = 256
WEIGHT_DECAY = 0.01

# Output
OUTPUT_DIR = "outputs/soft_multitask_bert"

# Reproducibility
SEED = 42

# Experiment
EXPERIMENT_NAME = "soft_multitask_bert"