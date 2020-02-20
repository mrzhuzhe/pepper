######################################################################
# Define the model
# ----------------
#
# The model is composed of the
# `EmbeddingBag <https://pytorch.org/docs/stable/nn.html?highlight=embeddingbag#torch.nn.EmbeddingBag>`__
# layer and the linear layer (see the figure below). ``nn.EmbeddingBag``
# computes the mean value of a “bag” of embeddings. The text entries here
# have different lengths. ``nn.EmbeddingBag`` requires no padding here
# since the text lengths are saved in offsets.
#
# Additionally, since ``nn.EmbeddingBag`` accumulates the average across
# the embeddings on the fly, ``nn.EmbeddingBag`` can enhance the
# performance and memory efficiency to process a sequence of tensors.
#
# .. image:: ../_static/img/text_sentiment_ngrams_model.png
#

import torch.nn as nn
import torch.nn.functional as F
class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
