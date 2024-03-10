"""Formats the trianing data into eventfile format.
The eventfile format is a tab-separated file with two columns: cues and outcomes.
The cues column contains the context, syllables, and segments of a word.
The outcomes column contains the word itself.  

NOTE: This script requires download of the en_us_cmudict_forward.pt file. THe syllabifier code is an early version of the syllabifier by Kyle Gorman.

Usage:
    python eventfilesV1.py
"""

import os
import re
from typing import LiteralString

import pandas as pd
from tqdm import tqdm
from dp.phonemizer import Phonemizer


def get_context(index) -> list[str]:
    """Returns the context of a word in a list of words."""
    if index == 0:
        context = "c." + words[index + 1]
    elif index == len(words) - 1:
        context = "c." + words[index - 1]
    else:
        previous_word = "c." + words[index - 1]
        following_word = "c." + words[index + 1]
        context = previous_word + "_" + following_word
    return context


def get_segments(word, upper=False) -> str | list[str]:
    """Returns the segments of a word."""
    raw_segment_string = phonemizer(word, lang="en_us")

    if upper:
        segments_string = re.sub(r"[\[\]-]", " ", raw_segment_string)
        # segments = segments_string.split()
        return segments_string
    else:
        segments_string = re.sub(r"[\[\]-]", " ", raw_segment_string.lower())
        segments = segments_string.split()
        return segments


def join_segments(word) -> LiteralString:
    """Returns the segments of a word in a cue formatted string."""
    segments = get_segments(word)
    segments_y = []
    for segment in segments:
        segment = "s." + segment
        segments_y.append(segment)
    segments_joined = "_".join(segments_y)
    return segments_joined


def join_syllables(syllables) -> LiteralString:
    """Returns the syllables of a word in a cue formatted string."""
    syll_list = syllables.split()
    syllable_cuestring = []
    for entry in syll_list:
        syllable_cue = "y." + entry
        syllable_cuestring.append(syllable_cue)
    syllables_joined = "_".join(syllable_cuestring)
    return syllables_joined


def syllabify(language, word):
    """Syllabifies the word, given a language configuration loaded with
    loadLanguage. word is either a string of phonemes from the CMU
    pronouncing dictionary set (with optional stress numbers after vowels),
    or a Python list of phonemes, e.g. "B AE1 T" or ["B", "AE1", "T"]
    """

    if type(word) == str:
        word = word.split()
    # This is the returned data structure.
    syllables = []

    # This maintains a list of phonemes between nuclei.
    internuclei = []

    for phoneme in word:

        phoneme = phoneme.strip()
        if phoneme == "":
            continue
        stress = None
        if phoneme[-1].isdigit():
            stress = int(phoneme[-1])
            phoneme = phoneme[0:-1]

        # Split the consonants seen since the last nucleus into coda and
        # onset.
        if phoneme in language["vowels"]:

            coda = None
            onset = None

            # If there is a period in the input, split there.
            if "." in internuclei:
                period = internuclei.index(".")
                coda = internuclei[:period]
                onset = internuclei[period + 1 :]

            else:
                # Make the largest onset we can. The 'split' variable marks
                # the break point.
                for split in range(0, len(internuclei) + 1):
                    coda = internuclei[:split]
                    onset = internuclei[split:]

                    # If we are looking at a valid onset, or if we're at the
                    # start of the word (in which case an invalid onset is
                    # better than a coda that doesn't follow a nucleus), or
                    # if we've gone through all of the onsets and we didn't
                    # find any that are valid, then split the nonvowels
                    # we've seen at this location.
                    if (
                        " ".join(onset) in language["onsets"]
                        or len(syllables) == 0
                        or len(onset) == 0
                    ):
                        break

            # Tack the coda onto the coda of the last syllable. Can't do it
            # if this is the first syllable.
            if len(syllables) > 0:
                syllables[-1][3].extend(coda)

            # Make a new syllable out of the onset and nucleus.
            syllables.append((stress, onset, [phoneme], []))

            # At this point we've processed the internuclei list.
            internuclei = []

        elif not phoneme in language["consonants"] and phoneme != ".":
            raise ValueError("Invalid phoneme: " + phoneme)

        else:  # a consonant
            internuclei.append(phoneme)

    # Done looping through phonemes. We may have consonants left at the end.
    # We may have even not found a nucleus.
    if len(internuclei) > 0:
        if len(syllables) == 0:
            syllables.append((None, internuclei, [], []))
        else:
            syllables[-1][3].extend(internuclei)

    return syllables


def stringify(syllables) -> LiteralString:
    """This function takes a syllabification returned by syllabify and
    turns it into a string, with phonemes spearated by spaces and
    syllables spearated by periods."""
    ret = []
    for syl in syllables:
        stress, onset, nucleus, coda = syl
        if stress != None and len(nucleus) != 0:
            nucleus[0] += str(stress)
        ret.append("".join(onset + nucleus + coda))
    return " ".join(ret)


if __name__ == "__main__":
    phonemizer = Phonemizer.from_checkpoint("../data/en_us_cmudict_forward.pt")

    # English language settings for the language parameter in the syllabifier.
    English = {
        "consonants": [
            "B",
            "CH",
            "D",
            "DH",
            "F",
            "G",
            "HH",
            "JH",
            "K",
            "L",
            "M",
            "N",
            "NG",
            "P",
            "R",
            "S",
            "SH",
            "T",
            "TH",
            "V",
            "W",
            "Y",
            "Z",
            "ZH",
        ],
        "vowels": [
            "AA",
            "AE",
            "AH",
            "AO",
            "AW",
            "AY",
            "EH",
            "ER",
            "EY",
            "IH",
            "IY",
            "OW",
            "OY",
            "UH",
            "UW",
        ],
        "onsets": [
            "P",
            "T",
            "K",
            "B",
            "D",
            "G",
            "F",
            "V",
            "TH",
            "DH",
            "S",
            "Z",
            "SH",
            "CH",
            "JH",
            "M",
            "N",
            "R",
            "L",
            "HH",
            "W",
            "Y",
            "P R",
            "T R",
            "K R",
            "B R",
            "D R",
            "G R",
            "F R",
            "TH R",
            "SH R",
            "P L",
            "K L",
            "B L",
            "G L",
            "F L",
            "S L",
            "T W",
            "K W",
            "D W",
            "S W",
            "S P",
            "S T",
            "S K",
            "S F",
            "S M",
            "S N",
            "G W",
            "SH W",
            "S P R",
            "S P L",
            "S T R",
            "S K R",
            "S K W",
            "S K L",
            "TH W",
            "ZH",
            "P Y",
            "K Y",
            "B Y",
            "F Y",
            "HH Y",
            "V Y",
            "TH Y",
            "M Y",
            "S P Y",
            "S K Y",
            "G Y",
            "HH W",
            "",
        ],
    }
    language = English

    path = "../data/allwords_perspeaker_csv/"
    files = os.listdir(path)

    speakers = []
    transcriptions = {}

    for file in tqdm(files):
        # List of words for the speaker
        df = pd.read_csv(path + file)
        words = df["token"].tolist()

        # Create new dataframe for speaker.
        df_name = file.replace(".csv", "")
        df = pd.DataFrame({"cues": [], "outcomes": []})

        for index, word in enumerate(words):

            # Get context.
            context = get_context(index, word)

            # Get Segments.
            if word not in transcriptions.keys():
                segments = join_segments(word)
                raw_segments = get_segments(word, upper=True)
                transcriptions[word] = {
                    "cue_segments": str(segments),
                    "segments": str(raw_segments),
                }
            else:
                segments = transcriptions[word]["cue_segments"]

            # Get syllables.
            raw_syllables = stringify(
                syllabify(English, transcriptions[word]["segments"])
            )
            syllables = join_syllables(raw_syllables)

            # Append all cue strings and clean track boundaries
            cues = context + "_" + syllables.lower() + "_" + segments
            if "NA_" in cues:
                cues = cues.replace("NA", "")

            # Append all information to the dataframe as a new row.
            df.loc[len(df)] = {"cues": str(cues), "outcomes": str(word)}

            # Save individual speaker dataframe.
            df.to_csv(
                "/gpfs/project/anste145/input_files/buckeye_data/event_files/"
                + df_name
                + ".tsv"
            )
            speakers.append(df)

    # Concat all individual speaker dataframes into one dataframe.
    buckeye_event_file = pd.concat(speakers)
    buckeye_event_file.to_csv("../data/buckeye_event_file.tsv", index=False)
