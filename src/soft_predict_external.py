import json
import numpy as np
import os


def predict_soft_external(

        trainer,

        dataset,

        intent_labels,

        content_labels,

        output_file
):

    predictions = trainer.predict(dataset)

    intent_logits = predictions.predictions[0]
    content_logits = predictions.predictions[1]

    intent_predictions = np.argmax(intent_logits, axis=1)
    content_predictions = np.argmax(content_logits, axis=1)

    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        for i in range(len(dataset)):

            record = {

                "paper_id":
                    dataset[i]["paper_id"],

                "citation_context":
                    dataset[i]["citation_context"],

                "predicted_intent":
                    intent_labels[
                        intent_predictions[i]
                    ],

                "predicted_content":
                    content_labels[
                        content_predictions[i]
                    ]

            }

            f.write(
                json.dumps(record)
                + "\n"
            )

    print(
        f"Predictions saved to {output_file}"
    )