
# Dataset
DATASET_NAME = "SciCite"
DATA_DIR = "data/scicite"

EXTERNAL_DATASET = "None"  # Options: None, "benchmark", "ape", "unarxiv", "external"
# Options: None, "benchmark", "ape", "unarxiv", "external", "unarxiv_cs"

#Benchmark
BENCHMARK_FILE = "data/benchmark/benchmark.json"
RUN_BENCHMARK = False
APE_FILE = "data/ape/ape_citations.jsonl"
UNARXIV_FILE = "data/external/econ_citations.jsonl"
UNARXIV_CS_FILE = "data/external/cs_citations.jsonl"



# Model
MODEL_NAME = "allenai/scibert_scivocab_uncased"

# Labels
LABELS = [
    "background",
    "method",
    "result"
]

# Training
EPOCHS = 4
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
MAX_LENGTH = 256
WEIGHT_DECAY = 0.01

# Output
OUTPUT_DIR = "outputs/scicite_scibert"

# Reproducibility
SEED = 42

#Experiment
EXPERIMENT_NAME = "scicite_scibert"