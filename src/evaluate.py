import os
import json
import numpy as np

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
      "dataset": "SciCite",
      "model": "allenai/scibert_scivocab_uncased",
      "epochs": 4,
      "learning_rate": 2e-5,
      "batch_size": 16,
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