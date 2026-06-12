import os
import json
import numpy as np
import pandas as pd
from configs.current_config import (
    DATASET_NAME,
    MODEL_NAME,
    EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    MAX_LENGTH
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate_soft_model(
    trainer,
    test_dataset,
    intent_labels,
    content_labels,
    output_dir
):

    predictions = trainer.predict(
        test_dataset
    )

    intent_logits = (
        predictions.predictions[0]
    )

    content_logits = (
        predictions.predictions[1]
    )

    intent_preds = np.argmax(
        intent_logits,
        axis=1
    )

    content_preds = np.argmax(
        content_logits,
        axis=1
    )

    intent_true = np.array(
        test_dataset[
            "intent_label"
        ]
    )

    content_true = np.array(
        test_dataset[
            "content_label"
        ]
    )
    
    intent_id2label = {
    idx: label
    for idx, label in enumerate(
        intent_labels
    )
    }

    content_id2label = {
        idx: label
        for idx, label in enumerate(
            content_labels
        )
    }

    predictions_dir = (
        f"{output_dir}/predictions"
    )

    os.makedirs(
        predictions_dir,
        exist_ok=True
    )
    
    prediction_df = pd.DataFrame({

    "citation_context":
    test_dataset[
        "citation_context"
    ],

    "true_intent": [

        intent_id2label[x]

        for x in intent_true
    ],

    "predicted_intent": [

        intent_id2label[x]

        for x in intent_preds
    ],

    "true_content": [

        content_id2label[x]

        for x in content_true
    ],

    "predicted_content": [

        content_id2label[x]

        for x in content_preds
    ]
    })
    prediction_df.to_csv(
        f"{predictions_dir}/test_predictions.csv",
        index=False
    )
    
    print(
    "\nPredictions saved successfully."
    )

    intent_accuracy = accuracy_score(
        intent_true,
        intent_preds
    )

    content_accuracy = accuracy_score(
        content_true,
        content_preds
    )

    intent_f1 = f1_score(
        intent_true,
        intent_preds,
        average="macro"
    )

    content_f1 = f1_score(
        content_true,
        content_preds,
        average="macro"
    )

    joint_accuracy = np.mean(

        (intent_preds == intent_true)

        &

        (content_preds == content_true)
    )

    print("\nIntent Accuracy")
    print(intent_accuracy)

    print("\nContent Accuracy")
    print(content_accuracy)

    print("\nJoint Accuracy")
    print(joint_accuracy)

    metrics_dir = (
        f"{output_dir}/metrics"
    )

    os.makedirs(
        metrics_dir,
        exist_ok=True
    )

    intent_metrics = {

    "dataset":
    DATASET_NAME,

    "task":
    "intent",

    "model":
    MODEL_NAME,

    "epochs":
    EPOCHS,

    "learning_rate":
    LEARNING_RATE,

    "batch_size":
    BATCH_SIZE,

    "max_length":
    MAX_LENGTH,

    "accuracy":
    float(intent_accuracy),

    "macro_f1":
    float(intent_f1),

    "labels":
    intent_labels
    }
    
    content_metrics = {

    "dataset":
    DATASET_NAME,

    "task":
    "content",

    "model":
    MODEL_NAME,

    "epochs":
    EPOCHS,

    "learning_rate":
    LEARNING_RATE,

    "batch_size":
    BATCH_SIZE,

    "max_length":
    MAX_LENGTH,

    "accuracy":
    float(content_accuracy),

    "macro_f1":
    float(content_f1),

    "labels":
    content_labels
    }
    
    metrics = {

    "dataset":
    DATASET_NAME,

    "task":
    "joint",

    "model":
    MODEL_NAME,

    "epochs":
    EPOCHS,

    "learning_rate":
    LEARNING_RATE,

    "batch_size":
    BATCH_SIZE,

    "max_length":
    MAX_LENGTH,
    
    "intent_accuracy":
    float(intent_accuracy),

    "intent_macro_f1":
    float(intent_f1),

    "content_accuracy":
    float(content_accuracy),

    "content_macro_f1":
    float(content_f1),

    "joint_accuracy":
    float(joint_accuracy)
    }
    
    with open(
    f"{metrics_dir}/intent_metrics.json",
    "w"
    ) as f:

        json.dump(
            intent_metrics,
            f,
            indent=4
        )
    with open(
    f"{metrics_dir}/content_metrics.json",
    "w"
    ) as f:

        json.dump(
            content_metrics,
            f,
            indent=4
        )
    
    with open(
    f"{metrics_dir}/metrics.json",
    "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    intent_report = (
        classification_report(

            intent_true,

            intent_preds,
            labels=list(range(len(intent_labels))),
            target_names=
            intent_labels,
            zero_division=0
        )
    )

    content_report = (
        classification_report(

            content_true,

            content_preds,
            labels=list(range(len(content_labels))),
            target_names=
            content_labels,
            zero_division=0
        )
    )

    with open(

        f"{metrics_dir}/intent_report.txt",

        "w"
    ) as f:

        f.write(
            intent_report
        )

    with open(

        f"{metrics_dir}/content_report.txt",

        "w"
    ) as f:

        f.write(
            content_report
        )

    np.savetxt(

        f"{metrics_dir}/intent_confusion.txt",

        confusion_matrix(

            intent_true,

            intent_preds,
            labels=list(range(len(intent_labels))),
        ),

        fmt="%d"
    )

    np.savetxt(

        f"{metrics_dir}/content_confusion.txt",

        confusion_matrix(

            content_true,

            content_preds,
            labels=list(range(len(content_labels))),
        ),

        fmt="%d"
    )

    print(
        "\nMetrics saved successfully."
    )