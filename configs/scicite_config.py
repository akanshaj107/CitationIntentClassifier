
# Dataset
DATASET_NAME = "SciCite"
DATA_DIR = "data/scicite"

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
OUTPUT_DIR = "outputs/scicite"

# Reproducibility
SEED = 42