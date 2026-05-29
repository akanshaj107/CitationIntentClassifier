import os
import json
import numpy as np
import pandas as pd
from configs.scicite_config import (
    DATASET_NAME,
    MODEL_NAME,
    EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    MAX_LENGTH
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)


def evaluate_model(
    trainer,
    test_dataset,
    label_names,
    output_dir
):

    # -------------------------
    # PREDICTIONS
    # -------------------------

    predictions = trainer.predict(test_dataset)

    preds = np.argmax(
        predictions.predictions,
        axis=-1
    )

    labels = predictions.label_ids
    id2label = {
    idx: label
    for idx, label in enumerate(label_names)
  }
    predictions_dir = f"{output_dir}/predictions"

    os.makedirs(
        predictions_dir,
        exist_ok=True
    )
    prediction_df = pd.DataFrame({
      "text": test_dataset["string"],
      "true_label": [
        id2label[label]
        for label in labels
      ],
    "predicted_label": [
        id2label[pred]
        for pred in preds
      ]
  })

    prediction_df.to_csv(
      f"{predictions_dir}/test_predictions.csv",
      index=False
  )

    # -------------------------
    # METRICS
    # -------------------------

    accuracy = accuracy_score(
        labels,
        preds
    )

    macro_f1 = f1_score(
        labels,
        preds,
        average="macro"
    )

    report = classification_report(
        labels,
        preds,
        target_names=label_names
    )

    matrix = confusion_matrix(
        labels,
        preds
    )

    # -------------------------
    # PRINT RESULTS
    # -------------------------

    print("\nClassification Report:\n")
    print(report)

    print("\nConfusion Matrix:\n")
    print(matrix)

    # -------------------------
    # CREATE OUTPUT FOLDER
    # -------------------------

    metrics_dir = f"{output_dir}/metrics"

    os.makedirs(
        metrics_dir,
        exist_ok=True
    )

    # -------------------------
    # SAVE METRICS JSON
    # -------------------------

    metrics = {
    "dataset": DATASET_NAME,
    "model": MODEL_NAME,

    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH,

    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1)
  }

    with open(
        f"{metrics_dir}/metrics.json",
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    # -------------------------
    # SAVE CLASSIFICATION REPORT
    # -------------------------

    with open(
        f"{metrics_dir}/classification_report.txt",
        "w"
    ) as f:

        f.write(report)

    # -------------------------
    # SAVE CONFUSION MATRIX
    # -------------------------

    np.savetxt(
        f"{metrics_dir}/confusion_matrix.txt",
        matrix,
        fmt="%d"
    )

    print("\nMetrics saved successfully.")