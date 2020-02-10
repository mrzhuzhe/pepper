# https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
import torch
from loadData import all_letters, n_letters, category_lines, all_categories, n_categories, unicodeToAscii
from rnn import RNN
from helper import randomChoice, randomTrainingPair, categoryTensor, inputTensor, targetTensor, randomTrainingExample, timeSince
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time

if n_categories == 0:
    raise RuntimeError('Data not found. Make sure that you downloaded data '
        'from https://download.pytorch.org/tutorial/data.zip and extract it to '
        'the current directory.')

#       print('# categories:', n_categories, all_categories)
#   print(unicodeToAscii("O'Néàl"))

learning_rate = 0.0005
n_iters = 100000
print_every = 5000
plot_every = 500
PATH = './model/generateModel.pth'

# neg log likehood
criterion = nn.NLLLoss()
rnn = RNN(n_letters, 128, n_letters)

def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / input_line_tensor.size(0)

def runTrain():

    all_losses = []
    total_loss = 0 # Reset every plot_every iters

    start = time.time()

    for iter in range(1, n_iters + 1):
        output, loss = train(*randomTrainingExample())
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

    # 保存训练结果
    torch.save(rnn.state_dict(), PATH)
    plt.figure()
    plt.plot(all_losses)

max_length = 20

# Sample from a category and starting letter
def sample(category, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name

# Get multiple samples from one category and multiple starting letters
def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))

def loadModel():
    # 弱类型引用形语言 直接load就行
    rnn.load_state_dict(torch.load(PATH))

def init():
    #runTrain()
    loadModel()
    #samples('Russian', 'RUS')
    #samples('German', 'GER')
    #samples('Spanish', 'SPA')
    samples('Chinese', 'zzzzzz')

if __name__ == "__main__":
    init()
