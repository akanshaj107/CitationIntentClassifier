DATASET_NAME = "ACL-ARC"

DATA_DIR = "data/acl_arc"

MODEL_NAME = "bert-base-uncased"

LABELS = [
    "background",
    "uses",
    "compareorcontrast",
    "motivation",
    "extends",
    "future"
]

EPOCHS = 1
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
MAX_LENGTH = 256
WEIGHT_DECAY = 0.01

OUTPUT_DIR = "outputs/acl_arc_bert"

SEED = 42

EXPERIMENT_NAME = "acl_arc_bert"