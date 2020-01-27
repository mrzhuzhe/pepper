def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1

    # ------------------
    # Write your implementation here.
    flattened_corpus = [y for x in corpus for y in x]
    num_corpus_words += 1
    for word in flattened_corpus:
        if word not in corpus_words:
            corpus_words.append(word)
            num_corpus_words += 1
    corpus_words.sort()
    # ------------------

    print("corpus_words",  "\n",  corpus_words, "\n", "num_corpus_words",  "\n",  num_corpus_words)

    return corpus_words, num_corpus_words
