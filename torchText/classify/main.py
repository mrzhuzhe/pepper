"""
 https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
 data loader 改造参考： https://pytorch.org/text/_modules/torchtext/datasets/text_classification.html
 代码结构待拆分
 加载数据集的时候用了n-gram
 label 是数据集的第一col
 目标：output 和 实际label的差
 95 分训练和验证集
 model：只有一个 1308844 x 32 的 embbedingbag 词袋层  和一个 32 x 4 的全联通层
"""

import torch
import torchtext
#from torchtext.datasets import text_classification
from dataloader import _setup_datasets
from model import TextSentiment

savePATH = './model/torchText/simpleModel.pth'
from torch.utils.data import DataLoader

import time
from torch.utils.data.dataset import random_split

import re
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer

NGRAMS = 2
#import os

#if not os.path.isdir('./data/AG/ag_news_csv'):
#	os.mkdir('./data/AG/ag_news_csv')

# train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
#     root='./data/AG/', ngrams=NGRAMS, vocab=None, download=False)
train_dataset, test_dataset = _setup_datasets(root='./data/AG/ag_news_csv', ngrams=NGRAMS, vocab=None)

BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
NUN_CLASS = len(train_dataset.get_labels())

# embbed 层 32维 num class 对应label 4层 vocab
print("VOCAB_SIZE", "NUN_CLASS", VOCAB_SIZE, NUN_CLASS)
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)

def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


def train_func(sub_train_):

    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)


N_EPOCHS = 5
min_valid_loss = float('inf')

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

def runTrain():
    train_len = int(len(train_dataset) * 0.95)
    sub_train_, sub_valid_ = \
        random_split(train_dataset, [train_len, len(train_dataset) - train_len])

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_loss, train_acc = train_func(sub_train_)
        valid_loss, valid_acc = test(sub_valid_)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

    torch.save(model.state_dict(), savePATH)

def runTest():
    print('Checking the results of test dataset...')
    test_loss, test_acc = test(test_dataset)
    print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')

ag_news_label = {1 : "World",
                 2 : "Sports",
                 3 : "Business",
                 4 : "Sci/Tec"}

def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

def runPredict(model):
    """
    ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
        enduring the season’s worst weather conditions on Sunday at The \
        Open on his way to a closing 75 at Royal Portrush, which \
        considering the wind and the rain was a respectable showing. \
        Thursday’s first round at the WGC-FedEx St. Jude Invitational \
        was another story. With temperatures in the mid-80s and hardly any \
        wind, the Spaniard was 13 strokes better in a flawless round. \
        Thanks to his best putting performance on the PGA Tour, Rahm \
        finished with an 8-under 62 for a three-stroke lead, which \
        was even more impressive considering he’d never played the \
        front nine at TPC Southwind."
    """
    ex_text_str = "alien kill a man with a laser gun"
    vocab = train_dataset.get_vocab()
    model = model.to("cpu")
    print("This is a %s news" %ag_news_label[predict(ex_text_str, model, vocab, 2)])

def loadModel():
    # 弱类型引用形语言 直接load就行
    model.load_state_dict(torch.load(savePATH))
    return model

def init():
    print("init")
    #   runTrain()
    model = loadModel()
    runTest()
    runPredict(model)


if __name__ == "__main__":
    init()
