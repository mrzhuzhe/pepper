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
