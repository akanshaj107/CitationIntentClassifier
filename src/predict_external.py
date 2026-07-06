import json
import numpy as np
import os


def predict_external(
    trainer,
    dataset,
    labels,
    output_file
):

    predictions = trainer.predict(dataset)

    pred_ids = np.argmax(
        predictions.predictions,
        axis=1
    )

    id2label = {
        i: label
        for i, label in enumerate(labels)
    }

    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        for i, pred in enumerate(pred_ids):

            record = {

                "paper_id":
                    dataset[i]["paper_id"],

                "citation_context":
                    dataset[i]["string"],

                "predicted_label":
                    id2label[pred]

            }

            f.write(
                json.dumps(record)
                + "\n"
            )

    print(
        f"Saved predictions to {output_file}"
    )