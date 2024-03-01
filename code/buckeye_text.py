"""Extract word lists per speaker from Buckeye corpus."""
import buckeye

if __name__ == "__main__":
    corpus = buckeye.corpus("../data/buckeye_corpus/")

    # Iterating over all 40 speakers form corpus.
    for speaker in corpus:
        # Init word_list.
        word_list = []

        # Each speaker has up to 6 tracks (interviews).
        # Iterating over each interview of a speaker.
        for track in speaker:
            # Iterating over each word for wach interview to append it to the dataframe.
            for word in track.words:
                # The word attribute has two attributes: word and pause.
                # We dont want the pauses and they dont have an orthography attribute.
                if hasattr(word, "orthography"):
                    word_list.append(word.orthography)

        new_file = open("../data/" + speaker.name + ".txt", "w")
        for line in word_list:
            new_file.write(line + "\n")
        new_file.close()
