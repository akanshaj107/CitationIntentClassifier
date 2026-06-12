
# Dataset
DATASET_NAME = "ACL-ARC"
DATA_DIR = "data/acl_arc"

# Model
MODEL_NAME = "allenai/scibert_scivocab_uncased"

# Labels
LABELS = [
    "background",
    "uses",
    "compareorcontrast",
    "motivation",
    "extends",
    "future"
]


# Training
EPOCHS = 4
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
MAX_LENGTH = 256
WEIGHT_DECAY = 0.01

# Output
OUTPUT_DIR = "outputs/acl_arc_scibert"

# Reproducibility
SEED = 42

#Experiment
EXPERIMENT_NAME = "acl_arc_scibert"