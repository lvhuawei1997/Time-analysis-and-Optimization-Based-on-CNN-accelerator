import torch
from torch import nn, optim
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tensorboardX import SummaryWriter
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import csv
import numpy as np
import matplotlib.pyplot as plt
import re


sparsity = []
batch_size = 128
data_dir = '/homework/lv/time_performance_optimization_for_CNN/dataSet/tiny-imagenet-200'
writer = SummaryWriter('/homework/lv/time_performance_optimization_for_CNN/result/Result_prune')
num_label = 200
normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
transform_train = transforms.Compose(
    [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        normalize, ])
transform_test = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize, ])
train_set = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
test_set = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)

# 因为batch_size的存在，因此输入的x的size实际为([60,1,224,224])
# 网络开始搭建，自定义类均要继承 nn.Module 类
# 只需要定义 __init__ 和 forward 即可
# class_number表明最后分类的数量，200意味着一共有200个类型的图片，编号为0到199


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x), True), 3, 2)
        x = F.max_pool2d(F.relu(self.conv2(x), True), 3, 2)
        x = F.relu(self.conv3(x), True)
        x = F.relu(self.conv4(x), True)
        x = F.max_pool2d(F.relu(self.conv5(x), True), 3, 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(F.dropout(self.fc6(x), 0.5), True)
        x = F.relu(F.dropout(self.fc7(x), 0.5), True)
        x = self.fc8(x)
        return x


# e_epoch为迭代次数，8代表迭代8次，但实际e_epoch是从0到7
# lr为初始学习率
# step用于学习率的更新
# checkpoint是指输出loss时的节点
e_epoch = 30
lr = 0.01
step = 20
checkpoint = 100
# device用于判断是不是满足cuda运行的条件，如果不满足则使用cpu
# 需要注意的是，用了device定义之后，往后的代码中，涉及到网络内容、参数、及数据集的情况都要加上to(device)
# 用net = Net().to(device)来调用上面创建的网络
# nn.CrossEntropyLoss()用来计算损失loss
# optim.SGD用来更新参数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AlexNet().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
loss_show = []
acc_show = []


# 制定学习率更新的条件
def re_learningrate(epoch):
    re_lr = lr *(0.1 ** (epoch // step))
    return re_lr


def train(epoch):
    model.train()
    # 更新梯度下降中的学习率
    for p in optimizer.param_groups:
        p['lr'] = re_learningrate(epoch)
    total_loss = 0.0
    # enumerate可以将数据集排序，因此i提取到每一个数据集的序号
    # 每一个数据集中有60张图片，意味着每一个i运行结束后有60张图片进行了训练
    # 调用的data_read函数中已经将图片读取到内存中
    # 因此traindataloader中，imgs可以直接读取到图片，labels则是对应的标签，是用0-199的数值代表的标签
    for i, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        # 运行前需要清除原有的grad
        optimizer.zero_grad()
        output = model(imgs)
        # 计算损失，这个loss代表的是一个batch_size大小的数据计算出的损失，不是指每一张图片的损失
        loss = criterion(output, labels)
        # tensorboardX
        loss_show.append(loss)
        # 反向传播
        loss.backward()
        # 梯度下降方法更新参数
        optimizer.step()
        # 计算累加的损失
        total_loss += loss.item()
        # 以checkpoint为检查点，计算平均的损失loss
        # 其中imgs.size()应该为([60,1,224,224])
        # 因为batch_size=60，所以一个epoch中，当i=99时，代表已经循环了训练了100次
        # 因此已经训练的图片量为(i+1)*imgs.size()[0]
        # 总共要训练的图片量为len(traindataloader.dataset)
        # 因为已经训练了100次（checkpoint设置的数量），因此平均的loss为total_loss/float(checkpoint)
        if (i+1) % checkpoint == 0:
            print('Epoch: {} [{}/{}] loss: {}'.format(epoch+1, (i+1)*imgs.size()[0], len(train_loader.dataset),
                                                      total_loss/float(checkpoint)))
            # 最后需要将总的loss归0，防止循环叠加出现误判
            total_loss = 0.0


def val(epoch):
    model.eval()
    correct = 0.0
    # 仅对网络训练结果进行验证，不需要反向传播，也不需要计算梯度，所以要使用with torch.no_grad()
    with torch.no_grad():
        # 此处的循环次数与train函数中的一样
        # 因为不用每个循环都输出correct，只需要最后整个测试集的correct，因此不用enumerate进行编号
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            # 以下注释没有考虑总数据集的大小，是将总的图片看为了60张，其实60只是一个batch_size的大小
            # output中其实为60行200列的数组,size大小为([60,200])
            # 假如一个epoch中的60张图片分别属于第1类、第2类、第3类...第200类
            # 逻辑判断的结果，1则代表判断结果是属于这一类
            # 则里面的内容为([[1,0,0,...,0],[0,1,0,...,0],...,[0,0,0,...,1]])
            # dim=1代表在列方向进行处理
            # argmax(dim=1)可以提取出列方向上最大值所在的序号值
            # 结果为([0,1,2,...,199])
            # predict.shape输入预测结果的维数，结果为([60])
            # labels.view()可以将标签文件中的数据按照predict.shape()的维度重组
            # 然后用predict.eq()判断两者是不是相等，相等则为1，不相等为0
            # 然后对判断结果进行累加，累加之后需要用item()将其转换为纯数字
            predict = output.argmax(dim=1)
            correct += predict.eq(labels.view(predict.shape)).sum().item()
        #然后计算出识别的正确率，此处为计算每一个epoch的正确率，因此要跳出循环
        acc = correct/float(len(test_loader.dataset))*100
        acc_show.append(acc)
        print('Epoch: {} Accuracy: {}'.format(epoch+1, acc))


# 每次完成一个epoch保存一次
def save(epoch):
    root = '/homework/lv/time_performance_optimization_for_CNN/log/log_prune/'
    stats = {
        'epoch': epoch,
        'model': model.state_dict()
    }
    if not os.path.exists(root):
        os.makedirs(root)
    savepath = os.path.join(root, 'model_{}.pth'.format(epoch + 1))
    torch.save(stats, savepath)
    print('saving checkpoint in {}'.format(savepath))


def pruning():
    parameters_to_prune = (
        (model.conv1, 'weight'),
        (model.conv2, 'weight'),
        (model.conv3, 'weight'),
        (model.conv4, 'weight'),
        (model.conv5, 'weight'),
        (model.fc6, 'weight'),
        (model.fc7, 'weight'),
        (model.fc8, 'weight'),
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.2,
    )
    sparsity.append(float(torch.sum(model.conv1.weight == 0))
            / float(model.conv1.weight.nelement()))
    print(
        "Sparsity in conv1.weight: {:.2f}%".format(
            100. * float(torch.sum(model.conv1.weight == 0))
            / float(model.conv1.weight.nelement())
        )
    )
    sparsity.append(float(torch.sum(model.conv2.weight == 0))
                    / float(model.conv2.weight.nelement()))
    print(
        "Sparsity in conv2.weight: {:.2f}%".format(
            100. * float(torch.sum(model.conv2.weight == 0))
            / float(model.conv2.weight.nelement())
        )
    )
    sparsity.append(float(torch.sum(model.conv3.weight == 0))
                    / float(model.conv3.weight.nelement()))
    print(
        "Sparsity in conv3.weight: {:.2f}%".format(
            100. * float(torch.sum(model.conv3.weight == 0))
            / float(model.conv3.weight.nelement())
        )
    )
    sparsity.append(float(torch.sum(model.conv4.weight == 0))
                    / float(model.conv4.weight.nelement()))
    print(
        "Sparsity in conv4.weight: {:.2f}%".format(
            100. * float(torch.sum(model.conv4.weight == 0))
            / float(model.conv4.weight.nelement())
        )
    )
    sparsity.append(float(torch.sum(model.conv5.weight == 0))
                    / float(model.conv5.weight.nelement()))
    print(
        "Sparsity in conv5.weight: {:.2f}%".format(
            100. * float(torch.sum(model.conv5.weight == 0))
            / float(model.conv5.weight.nelement())
        )
    )
    sparsity.append(float(torch.sum(model.fc6.weight == 0))
            / float(model.fc6.weight.nelement()))
    print(
        "Sparsity in fc6.weight: {:.2f}%".format(
            100. * float(torch.sum(model.fc6.weight == 0))
            / float(model.fc6.weight.nelement())
        )
    )
    sparsity.append(float(torch.sum(model.fc7.weight == 0))
                    / float(model.fc7.weight.nelement()))
    print(
        "Sparsity in fc7.weight: {:.2f}%".format(
            100. * float(torch.sum(model.fc7.weight == 0))
            / float(model.fc7.weight.nelement())
        )
    )
    sparsity.append(float(torch.sum(model.fc8.weight == 0))
                    / float(model.fc8.weight.nelement()))
    print(
        "Sparsity in fc8.weight: {:.2f}%".format(
            100. * float(torch.sum(model.fc8.weight == 0))
            / float(model.fc8.weight.nelement())
        )
    )
    print(
        "Global sparsity: {:.2f}%".format(
            100. * float(
                torch.sum(model.conv1.weight == 0)
                + torch.sum(model.conv2.weight == 0)
                + torch.sum(model.conv3.weight == 0)
                + torch.sum(model.conv4.weight == 0)
                + torch.sum(model.conv5.weight == 0)
                + torch.sum(model.fc6.weight == 0)
                + torch.sum(model.fc7.weight == 0)
                + torch.sum(model.fc8.weight == 0)
            )
            / float(
                model.conv1.weight.nelement()
                + model.conv2.weight.nelement()
                + model.conv3.weight.nelement()
                + model.conv4.weight.nelement()
                + model.conv5.weight.nelement()
                + model.fc6.weight.nelement()
                + model.fc7.weight.nelement()
                + model.fc8.weight.nelement()
            )
        )
    )


if __name__ == '__main__':
    print(model)
    summary(model, input_size=(3, 224, 224))
    print(model.state_dict().keys())
    for epoch in range(e_epoch):
        train(epoch)
        val(epoch)
        save(epoch)
    pruning()
    print(model)
    summary(model, input_size=(3, 224, 224))
    print(model.state_dict().keys())
    val(e_epoch)
    with open('/homework/lv/time_performance_optimization_for_CNN/result/AlexNet/pruneData/sparisty.txt', 'w') as f:
        for i in range(len(sparsity)):
            f.write(str(sparsity[i]))
            f.write('\n')
