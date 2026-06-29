from configs.benchmark_scicite_config import *

from src.benchmark_loader import load_benchmark_dataset
from src.preprocess import tokenize_dataset
from src.benchmark_predict import (
    load_trained_model,
    predict_dataset
)
import os


import numpy as np


# ----------------------------
# Load Benchmark Dataset
# ----------------------------

dataset = load_benchmark_dataset(
    BENCHMARK_FILE,
    TAXONOMY
)

print(dataset)

print(dataset[0]["string"])

# ----------------------------
# Tokenize
# ----------------------------

tokenized_dataset = tokenize_dataset(
    dataset,
    LABELS
)

print(tokenized_dataset)

# ----------------------------
# Load trained model
# ----------------------------

model = load_trained_model(
    MODEL_PATH
)

print("Model loaded successfully.")
print(model.config.label2id)
print(model.config.id2label)

# ----------------------------
# Predict
# ----------------------------

predictions = predict_dataset(
    model,
    tokenized_dataset
)


print(predictions.predictions[:10])

print("Prediction completed.")

# ----------------------------
# Convert logits to labels
# ----------------------------

predicted_ids = np.argmax(
    predictions.predictions,
    axis=1
)

predicted_labels = [
    ID2LABEL[idx]
    for idx in predicted_ids
]

from collections import Counter

print("\nPrediction Distribution")
print(Counter(predicted_labels))

print("\nGround Truth Distribution")
print(Counter(dataset["label"]))

print("\nFirst 10 Predictions:\n")

for i in range(10):

    print(
        f"True : {dataset[i]['label']}"
    )

    print(
        f"Pred : {predicted_labels[i]}"
    )

    print("----------------------------")
    
    print(MODEL_PATH)


print(MODEL_PATH)
print(os.listdir(MODEL_PATH))
