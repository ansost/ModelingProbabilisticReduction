"""Compute speech rate for each speaker utterance.
Defined as number of syllables per utterance divided by the total duration of the utterance.

Presupposes a dataframe with pauses, words and two empty columns: 'utteranceID' and 'global_sr'.

Usage: 
    python speech_rate.py
"""

import os
from typing import LiteralString

import buckeye
import pandas as pd
import regex as re
from tqdm import tqdm
from dp.phonemizer import Phonemizer


def get_segments(word, upper=False) -> list[str]:
    """Returns the segments of a word."""
    raw_segment_string = phonemizer(str(word), lang="en_us")

    if upper:
        segments_string = re.sub(r"[\[\]-]", " ", raw_segment_string)
        segments = segments_string.split()
        return segments
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
        syllable_cuestring.append(syllable_cue.lower())
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


def stringify(syllables):
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

    corpus = buckeye.corpus("../data/buckeye_corpus/")
    df = pd.DataFrame({"items": [], "trackID": []})
    num = 0

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

    for speaker in tqdm(corpus):
        num = num + 1
        tracks = {}
        trackNum = 0
        for track in speaker:
            trackWords = []

            for word in track.words:
                if isinstance(word, buckeye.containers.Word):
                    trackWords.append(word)
                elif isinstance(word, buckeye.containers.Pause):
                    pauseType = str(word).split(" ")
                    trackWords.append(pauseType[1])

            tracks[trackNum] = trackWords

            trackNum = trackNum + 1

        for entry in tracks.items():
            for word in entry[1]:
                df.loc[len(df)] = {"items": word, "trackID": entry[0]}

    num = 0
    pattern = r"\{(\w*)\}|\<(\w*)\>"
    utteranceID = 1
    inUtterance = bool
    forbidden_words = [
        "oh",
        "uh",
        "ah",
        "um",
        "mm",
        "hm",
        "huh",
        "uh-huh",
        "um-hum",
        "huh-uh",
        "hum-hum",
        "hmm",
        "hmmm",
        "mh",
        "mmh",
    ]
    # Appends all speaker dfs so there are no speaker overlaps while the df is being constructed.
    all_dfs = []

    for speaker in tqdm(corpus):
        num = num + 1
        df_name = str(num) + "df"
        df_name = pd.DataFrame({"items": [], "utteranceID": [], "global_sr": []})
        speaker_words = []

        for track in speaker:
            for word in track.words:
                if isinstance(word, buckeye.containers.Word):
                    speaker_words.append(word)
                elif isinstance(word, buckeye.containers.Pause):
                    pauseType = str(word).split(" ")
                    speaker_words.append(pauseType[1])

        # Collects words per utterance
        wordsUtterance = []

        for index, word in enumerate(speaker_words):

            if (
                isinstance(word, buckeye.containers.Word)
                and word.orthography not in forbidden_words
            ):
                inUtterance = True
            else:
                inUtterance = False

            if inUtterance == True:
                raw_syllables = stringify(
                    syllabify(English, get_segments(word.orthography, upper=True))
                )
                wordsUtterance.append(
                    (index, word.dur, len(raw_syllables.split(" ")), word.orthography)
                )

            elif inUtterance == False and len(wordsUtterance) > 0:
                totalDur = sum([item[1] for item in wordsUtterance])
                totalSyl = sum([item[2] for item in wordsUtterance])
                globalSr = totalSyl / totalDur

                for entry in wordsUtterance:
                    df_name.at[entry[0], "utteranceID"] = utteranceID
                    df_name.at[entry[0], "global_sr"] = globalSr
                    df_name.at[entry[0], "items"] = entry[3]

                utteranceID = utteranceID + 1
                wordsUtterance = []

        df_name.reset_index(drop=True, inplace=True)
        all_dfs.append(df_name)

    assert len(all_dfs) == 40
    regression = pd.read_csv(
        "../data/regression_data.csv",
        low_memory=True,
        engine="c",
    )
    alle_dfs = pd.concat(all_dfs, axis=0)
    alle_dfs.reset_index(drop=True, inplace=True)
    finalDF = pd.concat([regression, alle_dfs], axis=1)
    finalDF = finalDF.drop("items", axis=1)
    finalDF.to_csv(
        "../data/regression_data_final.csv",
        index=False,
    )
