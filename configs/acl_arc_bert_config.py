DATASET_NAME = "ACL-ARC"

DATA_DIR = "data/acl_arc"

#benchmark
BENCHMARK_FILE = "data/benchmark/benchmark.json"

RUN_BENCHMARK = False

EXTERNAL_DATASET = None

# Options: None, "benchmark", "ape", "unarxiv", "external", "unarxiv_cs"

APE_FILE = "data/ape/ape_citations.jsonl"

UNARXIV_FILE = "data/external/econ_citations.jsonl"

UNARXIV_CS_FILE = "data/external/cs_citations.jsonl"

MODEL_NAME = "bert-base-uncased"

LABELS = [
    "background",
    "uses",
    "compareorcontrast",
    "motivation",
    "extends",
    "future"
]

EPOCHS = 4
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
MAX_LENGTH = 256
WEIGHT_DECAY = 0.01

OUTPUT_DIR = "outputs/acl_arc_bert"

SEED = 42

EXPERIMENT_NAME = "acl_arc_bert"