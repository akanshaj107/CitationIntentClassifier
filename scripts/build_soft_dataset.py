from json.tool import main
import os

import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

RAW_DIR = "data/soft/raw"

OUTPUT_DIR = "data/soft"

SEED = 42


def build_soft_dataset():

    df = pd.read_csv(
        f"{RAW_DIR}/ACLARC_SOFT.tsv",
        sep="\t"
    )

    df = df[
        [
            "citation_context",
            "citation_function",
            "citation_object"
        ]
    ]

    df = df.dropna()

    train_df, temp_df = train_test_split(
        df,
        test_size=0.15,
        random_state=SEED,
        stratify=df[
            "citation_function"
        ]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df[
            "citation_function"
        ]
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
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

    cross_domain_df = pd.read_csv(
        f"{RAW_DIR}/ACT2_test_SOFT.tsv",
        sep="\t"
    )
    
    cross_domain_df = cross_domain_df.rename(
        columns={
            "cite_context": "citation_context"
        }
    )
    
    # Normalize labels
    cross_domain_df["citation_function"] = (
        cross_domain_df["citation_function"]
        .str.strip()
        .str.title()
    )

    cross_domain_df["citation_object"] = (
        cross_domain_df["citation_object"]
        .str.strip()
        .str.title()
    )
    
    print(
        sorted(
            cross_domain_df["citation_function"].unique()
        )
    )

    print(
        sorted(
            cross_domain_df["citation_object"].unique()
        )
    )

    cross_domain_df = cross_domain_df[
        [
            "citation_context",
            "citation_function",
            "citation_object"
        ]
    ]
    
    cross_domain_df.to_json(
        f"{OUTPUT_DIR}/cross_domain.jsonl",
        orient="records",
        lines=True
    )
    
    

    print("SOFT dataset built.")
    
if __name__ == "__main__":
    build_soft_dataset()