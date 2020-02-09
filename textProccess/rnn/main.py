# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
# https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification
import torch
import time
import math
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from rnn import RNN
from loadData import all_letters, n_letters, category_lines, all_categories, n_categories
from line2tensor import letterToIndex, letterToTensor, lineToTensor
from helper import categoryFromOutput, randomChoice, randomTrainingExample

n_iters = 100000
print_every = 5000
plot_every = 1000

learning_rate = 0.005
PATH = './model/lstmTestModel.pth'

n_hidden = 128

#
"""
print(findFiles('data/names/*.txt'))
print(unicodeToAscii('Ślusàrski'))
print(category_lines['Italian'][:5])
"""

# 现在rnn就是lstm
rnn = RNN(n_letters, n_hidden, n_categories)

"""
#   one-hot encoding
print(letterToTensor('J'))
print(lineToTensor('Jones').size())

# 输入转稀疏向量
input = letterToTensor('A')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input, hidden)

print(output)

input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)
output, next_hidden = rnn(input[0], hidden)
print(output)

print(categoryFromOutput(output))

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)
"""

# negative log likehood fn
criterion = nn.NLLLoss()

def train(category_tensor, line_tensor):
    # 还真是重新初始化隐藏层
    hidden = rnn.initHidden()

    # 梯度先还原
    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # 一次更新所有参数
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

def runTainfn():
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()

    # 训练
    for iter in range(1, n_iters + 1):
        # minibatch
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    # 保存训练结果
    torch.save(rnn.state_dict(), PATH)
    plt.figure()
    plt.plot(all_losses)

def showReslt():

    # 弱类型引用形语言 直接load就行
    rnn.load_state_dict(torch.load(PATH))

    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = evaluate(line_tensor)
        guess, guess_i = categoryFromOutput(output)
        category_i = all_categories.index(category)
        # 实际语言 对应语言
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()

    predict('Dovesky')
    predict('Jackson')
    predict('Satoshi')
    predict('Hazaki')


def init():

    #   runTainfn()

    showReslt()

if __name__ == "__main__":
    init()
