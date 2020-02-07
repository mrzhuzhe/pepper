import torch

from rnn import RNN
from loadData import all_letters, n_letters, category_lines, all_categories, n_categories
from line2tensor import letterToIndex, letterToTensor, lineToTensor

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

input = letterToTensor('A')
hidden =torch.zeros(1, n_hidden)

output, next_hidden = rnn(input, hidden)

input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)

# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

output, next_hidden = rnn(input[0], hidden)
print(output)
