import matplotlib.pyplot as plt
from assign01_1 import distinct_words
from assign01_2 import compute_co_occurrence_matrix
from assign01_3 import reduce_to_k_dim

def plot_embeddings(M_reduced, word2Ind, words):
    """ Plot in a scatterplot the embeddings of the words specified in the list "words".
        NOTE: do not plot all the words listed in M_reduced / word2Ind.
        Include a label next to each point.

        Params:
            M_reduced (numpy matrix of shape (number of unique words in the corpus , k)): matrix of k-dimensioal word embeddings
            word2Ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to visualize
    """
    # ------------------
    # Write your implementation here.
    for i,type in enumerate(words):
        x = M_reduced[i][0]
        y = M_reduced[i][1]
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x, y, type, fontsize=9)
    plt.show()
    # ------------------

input = ["START I am a pig , I am a dog , he is a rat, she is a cat, oldder is a snake END".split(' ')]
words, wordsNum = distinct_words(input)
M , word2Ind = compute_co_occurrence_matrix(input)
M_reduced = reduce_to_k_dim(M)
print("res", plot_embeddings(M_reduced, word2Ind, words))
