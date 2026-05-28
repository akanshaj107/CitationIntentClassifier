from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import numpy as np


def evaluate_model(
    trainer,
    test_dataset,
    label_names
):

    predictions = trainer.predict(test_dataset)

    preds = np.argmax(
        predictions.predictions,
        axis=-1
    )

    labels = predictions.label_ids

    print("\nClassification Report:\n")

    print(
        classification_report(
            labels,
            preds,
            target_names=label_names
        )
    )

    print("\nConfusion Matrix:\n")

    print(
        confusion_matrix(labels, preds)
    )