"""Extract word lists per speaker from Buckeye corpus.
There are 40 speakers, each has up to 6 tracks (interviews).
The word class has two attributes: word and pause. Pauses are filtered out since they do not have an orthography attribute.

Usage: 
    python buckeye_text.py
"""

import buckeye

if __name__ == "__main__":
    corpus = buckeye.corpus("../data/buckeye_corpus/")

    for speaker in corpus:
        word_list = []

        for track in speaker:
            for word in track.words:
                if hasattr(word, "orthography"):
                    word_list.append(word.orthography)

        new_file = open("../data/" + speaker.name + ".txt", "w")
        for line in word_list:
            new_file.write(line + "\n")
        new_file.close()
