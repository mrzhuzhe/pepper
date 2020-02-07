import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from vision_nn import Net
import torch.nn as nn
import torch.optim as optim

# gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
print("Let's use", torch.cuda.device_count(), "GPUs!")

#  全局参数
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
PATH = './model/visionModel.pth'

def init():
    # loadData
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # cifar10 可以从这下 贼好 哈哈哈 http://ai-atest.bj.bcebos.com/cifar-10-python.tar.gz
    #   download=True
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=1)

    # functions to show an image
    showExample(trainloader)

    # broken pipline
    #   images, labels = images.to(device), labels.to(device)

    # 训练
    #   trainNN(trainloader)

    # 交叉验证
    #   validNN(testloader)

    # 重新加载验证
    modelTest(testloader)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def showExample(trainloader):
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

def trainNN(trainloader):

    net = Net()

    #   net.to(device)

    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')

    torch.save(net.state_dict(), PATH)

def validNN(testloader):
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

def modelTest(testloader):
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    net = Net()
    net.load_state_dict(torch.load(PATH))
    # 使用 gpu 还是 cpu 似乎需要用to明确的指定 使用多个gpu 似乎需要 DataParalize来明确指定给多gpu
    # 多核 cpu gpu 如何指定使用多核呢
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))
    showReslt(net, testloader)

def showReslt(net, testloader):
    correct = 0
    total = 0

    net.to(device)

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

if __name__ == "__main__":
    init()
