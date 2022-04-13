import time

import numpy
import torch
from torch import nn, optim
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tensorboardX import SummaryWriter
import net.alexnet.alexnet as alexnet


layer_names = []
batch_size = 128
data_dir = '/homework/lv/Time analysis and Optimization Based on CNN accelerator/dataSet/tiny-imagenet-200'
writer = SummaryWriter('/homework/lv/Time analysis and Optimization Based on CNN accelerator/result/train_writer')
num_label = 200
normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
transform_train = transforms.Compose(
    [transforms.RandomResizedCrop((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        normalize, ])
transform_test = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize, ])
train_set = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
test_set = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)

# 因为batch_size的存在，因此输入的x的size实际为([60,1,224,224])
# 网络开始搭建，自定义类均要继承 nn.Module 类
# 只需要定义 __init__ 和 forward 即可
# class_number表明最后分类的数量，200意味着一共有200个类型的图片，编号为0到199
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
model = alexnet.AlexNet().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
loss_show = []
acc_show = []
t = []
tt = []
ttt = []

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
    # print(len(train_loader))
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
        # print(i)
        if (i+1) % checkpoint == 0:
            print('Epoch: {} [{}/{}] loss: {}'.format(epoch+1, (i+1)*imgs.size()[0], len(train_loader.dataset),
                                                      total_loss/float(checkpoint)))
            # 最后需要将总的loss归0，防止循环叠加出现误判
            total_loss = 0.0


def val(epoch):
    model.eval()
    correct = 0.0
    f1 = True
    # 仅对网络训练结果进行验证，不需要反向传播，也不需要计算梯度，所以要使用with torch.no_grad()
    with torch.no_grad():
        # 此处的循环次数与train函数中的一样
        # 因为不用每个循环都输出correct，只需要最后整个测试集的correct，因此不用enumerate进行编号
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if f1 == True:
                print(imgs)
                f1 = False
            t1 = time.time()
            output = model(imgs)
            t2 = time.time()
            t.append(1000 * (t2 - t1))
            predict = output.argmax(dim=1)
            correct += predict.eq(labels.view(predict.shape)).sum().item()
        # 然后计算出识别的正确率，此处为计算每一个epoch的正确率，因此要跳出循环
        acc = correct/float(len(test_loader.dataset))*100
        acc_show.append(acc)
        print('Epoch: {} Accuracy: {}'.format(epoch+1, acc))


def val2(epoch):
    f2 = True
    model.eval()
    correct = 0.0
    # 仅对网络训练结果进行验证，不需要反向传播，也不需要计算梯度，所以要使用with torch.no_grad()
    with torch.no_grad():
        # 此处的循环次数与train函数中的一样
        # 因为不用每个循环都输出correct，只需要最后整个测试集的correct，因此不用enumerate进行编号
        for imgs, labels in test_loader:
            x, labels = imgs.to(device), labels.to(device)
            mask = (x == x.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)
            imgs = torch.mul(mask, x)
            if f2 == True:
                print(imgs)
                f2 = False
            t1 = time.time()
            output = model(imgs)
            t2 = time.time()
            tt.append(1000 * (t2 - t1))
            predict = output.argmax(dim=1)
            correct += predict.eq(labels.view(predict.shape)).sum().item()
        # 然后计算出识别的正确率，此处为计算每一个epoch的正确率，因此要跳出循环
        acc = correct/float(len(test_loader.dataset))*100
        acc_show.append(acc)
        print('Epoch: {} Accuracy: {}'.format(epoch+1, acc))


def val3(epoch):
    f3 = True
    model.eval()
    correct = 0.0
    # 仅对网络训练结果进行验证，不需要反向传播，也不需要计算梯度，所以要使用with torch.no_grad()
    with torch.no_grad():
        # 此处的循环次数与train函数中的一样
        # 因为不用每个循环都输出correct，只需要最后整个测试集的correct，因此不用enumerate进行编号
        for i in range(79):
            imgs, labels = torch.tensor(numpy.zeros((128, 3, 224, 224))).type(torch.FloatTensor).to(device), torch.tensor(numpy.zeros(128)).to(device)
            if f3 == True:
                print(imgs)
                f3 = False
            t1 = time.time()
            output = model(imgs)
            t2 = time.time()
            ttt.append(1000 * (t2 - t1))
            predict = output.argmax(dim=1)
            correct += predict.eq(labels.view(predict.shape)).sum().item()
        # 然后计算出识别的正确率，此处为计算每一个epoch的正确率，因此要跳出循环
        acc = correct/float(len(test_loader.dataset))*100
        acc_show.append(acc)
        print('Epoch: {} Accuracy: {}'.format(epoch+1, acc))


# 每次完成一个epoch保存一次
def save(epoch):
    root = '/homework/lv/Time analysis and Optimization Based on CNN accelerator/log/train_pth'
    stats = {
        'epoch': epoch,
        'model': model.state_dict()
    }
    if not os.path.exists(root):
        os.makedirs(root)
    savepath = os.path.join(root, 'model_{}.pth'.format(epoch + 1))
    torch.save(stats, savepath)
    print('saving checkpoint in {}'.format(savepath))


if __name__ == '__main__':
    # summary(model, input_size=(3, 224, 224))
    # for epoch in range(e_epoch):
    #     train(epoch)
    #     val(epoch)
    #     save(epoch)
    val(e_epoch)
    val2(e_epoch)
    val3(e_epoch)
    sum = -t[0]
    count = 0
    for v in t:
        sum += v
        count += 1
    print(sum / count)

    sum2 = -tt[0]
    count2 = 0
    for v in tt:
        sum2 += v
        count2 += 1
    print(sum2 / count2)

    sum3 = -ttt[0]
    count3 = 0
    for v in ttt:
        sum3 += v
        count3 += 1
    print(sum3 / count3)

    print(count, count2, count3)





