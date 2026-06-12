import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score
)


def compute_soft_metrics(eval_pred):

    predictions, labels = eval_pred

    intent_logits = predictions[0]
    content_logits = predictions[1]

    intent_labels = labels[0]
    content_labels = labels[1]

    intent_preds = np.argmax(
        intent_logits,
        axis=1
    )

    content_preds = np.argmax(
        content_logits,
        axis=1
    )

    intent_accuracy = accuracy_score(
        intent_labels,
        intent_preds
    )

    content_accuracy = accuracy_score(
        content_labels,
        content_preds
    )

    intent_macro_f1 = f1_score(
        intent_labels,
        intent_preds,
        average="macro"
    )

    content_macro_f1 = f1_score(
        content_labels,
        content_preds,
        average="macro"
    )

    multitask_macro_f1 = (
        intent_macro_f1
        +
        content_macro_f1
    ) / 2

    return {

        "intent_accuracy":
        intent_accuracy,

        "content_accuracy":
        content_accuracy,

        "intent_macro_f1":
        intent_macro_f1,

        "content_macro_f1":
        content_macro_f1,

        "multitask_macro_f1":
        multitask_macro_f1
    }