from src.dataloader import load_acl_arc
from src.preprocess import tokenize_dataset
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model
from configs.current_config import (
    LABELS,
    DATA_DIR,
    OUTPUT_DIR
)


# ------------------------
# LABELS
# ------------------------

LABELS = [
    "background",
    "uses",
    "compareorcontrast",
    "motivation",
    "extends",
    "future"
]


# ------------------------
# LOAD DATA
# ------------------------

dataset = load_acl_arc(DATA_DIR)
#print(sorted(set(dataset["train"]["label"])))

# ------------------------
# TOKENIZE
# ------------------------

tokenized_dataset = tokenize_dataset(
    dataset, LABELS
)

#----To be removed after testing----#
#tokenized_dataset["train"] = tokenized_dataset["train"].select(range(100))
#tokenized_dataset["validation"] = tokenized_dataset["validation"].select(range(20))
#tokenized_dataset["test"] = tokenized_dataset["test"].select(range(20))
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
    output_dir=OUTPUT_DIR
)