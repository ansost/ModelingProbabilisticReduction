"""Simplify POS tags provided by the Buckeye corpus.

Usage:
    python simplifyPOS.py
"""

import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv(
        "../data/regression_data.csv",
        dtype={
            "speakerID": "category",
            "speakerAge": "category",
            "speakerGender": "category",
            "interviewerGender": "category",
            "wordID": "category",
            "wordDur": "float",
            "n_segments": "category",
            "n_syllables": "category",
        },
        engine="c",
        low_memory=True,
    )
    df.info(verbose=False, memory_usage="deep")

    new_tags = {
        "CC": "CC",
        "PP": "PP",
        "CD": "CD",
        "RB": "RB",
        "DT": "DT",
        "EX": "EX",
        "RBS": "RB",
        "FW": "FW",
        "RP": "RP",
        "IN": "IN",
        "SYM": "SYM",
        "JJ": "JJ",
        "TO": "to",
        "JJR": "JJ",
        "UH": "UH",
        "JJS": "JJ",
        "VB": "V",
        "LS": "LS",
        "VBD": "V",
        "MD": "MD",
        "VBG": "V",
        "NN": "NN",
        "VBN": "V",
        "NNS": "NN",
        "VBP": "V",
        "NNP": "NN",
        "VBZ": "V",
        "NNPS": "NN",
        "WDT": "WH",
        "PDT": "DT",
        "WP": "WP",
        "POS": "POS",
        "WP$": "WP",
        "PRP": "PRP",
        "WRB": "RB",
        "PP$": "PP",
        "PRP_VBP": "PRP",
        "V": "V",
        "to": "TO",
        "PRP$": "PRP",
        "WH": "WH",
        "RBR": "RB",
    }

    old_tags = df["wordPOS"].tolist()
    missing = []

    for index, tag in enumerate(old_tags):

        # Take care of Hybrid tags
        tag = tag.split("_")
        tag = tag[0]

        # Replace with new tag
        if tag in new_tags.keys():
            df.at[index, "wordPOS"] = new_tags[tag]
        else:
            missing.append((tag, index))

    assert len(missing) == 0

    df.drop(labels=["Unnamed: 0"], inplace=True, axis=1)
    df.to_csv("../data/regression_data.csv", index=False)
