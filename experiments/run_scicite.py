from src.dataloader import load_scicite
from src.preprocess import tokenize_dataset
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model


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

dataset = load_scicite(
    "data/scicite"
)


# ------------------------
# TOKENIZE
# ------------------------

tokenized_dataset = tokenize_dataset(
    dataset
)


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
    output_dir="outputs/scicite"
)


# ------------------------
# EVALUATE
# ------------------------

evaluate_model(
    trainer,
    tokenized_dataset["test"],
    LABELS,
    output_dir="outputs/scicite"
)