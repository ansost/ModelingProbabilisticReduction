"""Fix eventfile formatting by chaning the separator to tsv, removing the index column and merging into one file.

Usage:
    python correctingV1eventfiles.py
"""

import os
import gzip
import shutil

import pandas as pd

if __name__ == "__main__":
    path = "../data/updated_eventfiles/"
    all_ = []

    for file in os.listdir(path):
        df = pd.read_csv(
            path + file,
            low_memory=True,
            dtype=({"outcomes": "category"}),
            engine="c",
            sep="\t",
        )
        df = df.drop(columns="Unnamed: 0")
        df.to_csv(
            "../data/updated_eventfiles/" + file,
            sep="\t",
            index=False,
        )

    # Merge the individual files again.
    path = "../data/updated_eventfiles/"
    all_ = []

    for file in os.listdir(path):
        df = pd.read_csv(
            path + file,
            low_memory=True,
            dtype=({"outcomes": "category"}),
            engine="c",
            sep="\t",
        )
        all_.append(df)

    big_df = pd.concat(all_)
    concat_w = big_df["outcomes"].tolist()

    # Test if any words are missing.
    regression_words = pd.read_csv(
        "../data//regression_data.csv",
        usecols=["wordID"],
        dtype=({"wordID": "category"}),
        engine="c",
        low_memory=True,
    )
    regr_w = regression_words["wordID"].tolist()
    missing = [w for w in regr_w if w not in concat_w]
    assert len(missing) == 0

    # Write final event file to file and create a gzipped version needed for running the NDL model.
    big_df.to_csv(
        "../data/final_eventfile_buckeye.tsv",
        sep="\t",
        index=False,
    )

    with open("../data/final_eventfile_buckeye.tsv", "rb") as f:
        with gzip.open(
            "../data/final_eventfile_buckeye.gz",
            "wb",
        ) as output:
            shutil.copyfileobj(f, output)
