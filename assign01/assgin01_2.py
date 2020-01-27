import numpy as np
from assign01_1 import distinct_words

def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).

        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.

              For example, if we take the document "START All that glitters is not gold END" with window size of 4,
              "All" will co-occur with "START", "that", "glitters", "is", and "not".

        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of corpus words, number of corpus words)):
                Co-occurence matrix of word counts.
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = None
    word2Ind = {}
    # ------------------
    # Write your implementation here.
    flattened_corpus = [y for x in corpus for y in x]
    for index, word in enumerate(words):
        word2Ind[word] = index
    M = np.zeros((len(words),len(words)))
    for index, word in enumerate(flattened_corpus):
        left = max(0,index-window_size)
        right = min(len(flattened_corpus),index+window_size)
        for i in range(left,right):
            if i != index:
                co_word = flattened_corpus[i]
                if word in ["START","END"] and co_word in ["STRAT","END"]:
                    pass
                else:
                    M[word2Ind[word]][word2Ind[co_word]] += 1.
                    M[word2Ind[co_word]][word2Ind[word]] += 1.
    # ------------------
    return M, word2Ind
    
print(compute_co_occurrence_matrix("I am a pig , I am a dog"))
