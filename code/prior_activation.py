"""Compute prior and activation for all words.

Usage:
    python prior_activation.py
"""

import json

import pandas as pd
from tqdm.notebook import tqdm

from variablesOtherPrior import *

if __name__ == "__main__":
    # Load the event file.
    event_file = pd.read_csv(
        "../data/final_eventfile_buckeye.gz", sep="\t", low_memory=True, engine="c"
    )

    # Load the word column from the regression dataframe.
    speaker_word = pd.read_csv(
        "../data/regression_data.csv",
        usecols=["wordID"],
        dtype={"wordID": "category"},
        low_memory=True,
        engine="c",
    )

    # Load weights.
    df = xr.open_dataarray("../output/weights/weights_buckeye.nc")
    weight_matrix = df.to_pandas()
    weight_matrix = weight_matrix.transpose()
    weight_matrix.info(verbose=False, memory_usage="deep")

    words = speaker_word["wordID"].tolist()

    prior_dict = {}

    # Make a prior dictionary.
    for index, word in tqdm(enumerate(words)):
        if word not in prior_dict.keys():

            prior_all = get_prior(
                weight_matrix=weight_matrix, word_outcome=word, domain_specific=False
            )
            priors = get_prior(
                weight_matrix=weight_matrix, word_outcome=word, domain_specific=True
            )

            prior_dict[word] = {
                "prior_all": prior_all,
                "prior_segments": priors["Segment"],
                "prior_syllables": priors["Syllable"],
                "prior_context": priors["Context"],
            }
    out_file = open("../data/otherPrior_dictionary.json", "w")
    json.dump(prior_dict, out_file, indent=6)
    out_file.close()

    df = pd.DataFrame(
        {
            "prior_all": [],
            "prior_segments": [],
            "prior_syllables": [],
            "prior_context": [],
        }
    )

    for index, word in tqdm(enumerate(words)):
        df.at[index, "prior_all"] = prior_dict[word]["prior_all"]
        df.at[index, "prior_segments"] = prior_dict[word]["prior_segments"]
        df.at[index, "prior_syllables"] = prior_dict[word]["prior_syllables"]
        df.at[index, "prior_context"] = prior_dict[word]["prior_context"]

    # Loading whole regression dataset.
    regression_data = pd.read_csv(
        "../data/regression_data.csv",
        dtype={
            "speakerID": "category",
            "speakerAge": "category",
            "speakerGender": "category",
            "interviewerGender": "category",
            "wordID": "category",
            "wordDur": "float",
            "wordPOS": "category",
            "n_segments": "category",
            "n_syllables": "category",
        },
        engine="c",
        low_memory=True,
    )
    print("Concating...")
    result = pd.concat(objs=[regression_data, df], axis=1)
    result.to_csv("../data/regression_data_Prior.csv", index=False)

    words = speaker_word["wordID"].tolist()

    df = pd.DataFrame(
        {
            "activation_all": [],
            "activation_segments": [],
            "activation_syllables": [],
            "activation_context": [],
        }
    )

    for index, word in tqdm(enumerate(words)):
        if index == 0:
            c1 = "c." + words[index + 1]
            c2 = None
        elif index == len(words) - 1:
            c1 = "c." + words[index - 1]
            c2 = None
        else:
            c1 = "c." + words[index - 1]
            c2 = "c." + words[index + 1]

        act = activation(
            word_outcome=word,
            c1=c1,
            c2=c2,
            event_files=[event_file],
            weight_matrix=weight_matrix,
            domain_specific=False,
        )
        act_domain = activation(
            word_outcome=word,
            c1=c1,
            c2=c2,
            event_files=[event_file],
            weight_matrix=weight_matrix,
            domain_specific=True,
        )

        df.at[index, "activation_all"] = act
        df.at[index, "activation_segments"] = act_domain["Segment"]
        df.at[index, "activation_syllables"] = act_domain["Syllable"]
        df.at[index, "activation_context"] = act_domain["Context"]

    regression_data = pd.read_csv(
        "../data/regression_data_prior.csv",
        dtype={
            "speakerID": "category",
            "speakerAge": "category",
            "speakerGender": "category",
            "interviewerGender": "category",
            "wordID": "category",
            "wordDur": "float",
            "wordPOS": "category",
            "n_segments": "category",
            "n_syllables": "category",
        },
        engine="c",
        low_memory=True,
    )

    ende = pd.concat(obs=[regression_data, df], axis=1)
    ende.to_csv("../data/regression_data.csv", index=False)
