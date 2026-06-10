import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DIR = "data/acl_arc/raw"
OUTPUT_DIR = "data/acl_arc"

SEED = 42


def load_acl_arc_examples(raw_dir):

    examples = []

    json_files = sorted(
        Path(raw_dir).glob("*.json")
    )

    print(f"Found {len(json_files)} JSON files")

    total_citations = 0
    labeled_citations = 0
    unlabeled_citations = 0

    for file_path in json_files:

        with open(file_path, "r", encoding="utf-8") as f:
            paper = json.load(f)

        paper_id = paper.get("paper_id", "")

        citation_contexts = paper.get(
            "citation_contexts",
            []
        )

        for citation in citation_contexts:

            total_citations += 1

            if "citation_function" not in citation:

                unlabeled_citations += 1
                continue

            labeled_citations += 1

            text = str(
                citation.get(
                    "cite_context",
                    ""
                )
            ).strip()

            label = str(
                citation.get(
                    "citation_function",
                    ""
                )
            ).strip()

            citation_id = citation.get(
                "citation_id",
                ""
            )

            if len(text) == 0:
                continue

            examples.append(
                {
                    "paper_id": paper_id,
                    "citation_id": citation_id,
                    "string": text,
                    "label": label
                }
            )

    print("\nDataset Statistics")
    print("-" * 40)
    print(f"Total citations     : {total_citations}")
    print(f"Labeled citations   : {labeled_citations}")
    print(f"Unlabeled citations : {unlabeled_citations}")

    return pd.DataFrame(examples)


def print_distribution(df, name):

    print(f"\n{name} Distribution")

    counts = df["label"].value_counts()

    for label, count in counts.items():

        pct = count / len(df)

        print(
            f"{label:<20} "
            f"{count:>4} "
            f"({pct:.1%})"
        )


def build_acl_arc_dataset():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    df = load_acl_arc_examples(
        RAW_DIR
    )

    print("\nTotal labeled examples:")
    print(len(df))

    print_distribution(
        df,
        "Full Dataset"
    )

    # -------------------------
    # 85% train
    # 7.5% validation
    # 7.5% test
    # -------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=0.15,
        stratify=df["label"],
        random_state=SEED
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=SEED
    )

    print("\nSplit Sizes")
    print("-" * 40)

    print(
        f"Train      : {len(train_df)}"
    )

    print(
        f"Validation : {len(val_df)}"
    )

    print(
        f"Test       : {len(test_df)}"
    )

    print_distribution(
        train_df,
        "Train"
    )

    print_distribution(
        val_df,
        "Validation"
    )

    print_distribution(
        test_df,
        "Test"
    )

    train_df.to_json(
       f"{OUTPUT_DIR}/train.jsonl",
        orient="records",
        lines=True
    )

    val_df.to_json(
        f"{OUTPUT_DIR}/dev.jsonl",
        orient="records",
        lines=True
    )

    test_df.to_json(
        f"{OUTPUT_DIR}/test.jsonl",
        orient="records",
        lines=True
    )

    print("\nSaved Files")
    print("-" * 40)

    print(
        f"{OUTPUT_DIR}/train.jsonl"
    )

    print(
        f"{OUTPUT_DIR}/dev.jsonl"
    )

    print(
        f"{OUTPUT_DIR}/test.jsonl"
    )


if __name__ == "__main__":
    build_acl_arc_dataset()