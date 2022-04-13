import argparse
import datetime
import time
import random
import copy
import torch.utils.model_zoo as model_zoo
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Any
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import models, utils, datasets, transforms
import os
import math
from torch.nn import Parameter
from tensorboardX import SummaryWriter
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import csv
import math
import numpy as np
from numpy import dot
import numpy as np
import matplotlib.pyplot as plt
import re
import sys


layers_name = []
predict_time = []
sparsity = []


class myConv2d(nn.Conv2d):
    def __init__(self, h_in, w_in, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(myConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)
        self.h_in = h_in
        self.w_in = w_in
        self.xi = Parameter(torch.LongTensor(1), requires_grad=False)
        self.xi.data[0] = stride
        self.g = Parameter(torch.LongTensor(1), requires_grad=False)
        self.g.data[0] = groups
        self.p = Parameter(torch.LongTensor(1), requires_grad=False)
        self.p.data[0] = padding

    def __repr__(self):
        s = ('{name}({h_in}, {w_in}, {in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


def conv2d_out_dim(dim, kernel_size, padding=0, stride=1, dilation=1, ceil_mode=False):
    if ceil_mode:
        return int(math.ceil((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
    else:
        return int(math.floor((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))


# 不使用输入掩码情况下的卷积操作
class FixHWConv2d(myConv2d):
    def __init__(self, h_in, w_in, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(FixHWConv2d, self).__init__(h_in, w_in, in_channels, out_channels, kernel_size, stride,
                                          padding, dilation, groups, bias)

        self.hw = Parameter(torch.LongTensor(2), requires_grad=False)
        self.hw.data[0] = h_in
        self.hw.data[1] = w_in

    def forward(self, input):
        # Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        assert input.size(2) == self.hw.data[0] and input.size(3) == self.hw.data[1], 'input_size={}, but hw={}'.format(
            input.size(), self.hw.data)
        return super(FixHWConv2d, self).forward(input)


# 使用输入掩码情况下的卷积操作
class SparseConv2d(myConv2d):
    def __init__(self, h_in, w_in, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(SparseConv2d, self).__init__(h_in, w_in, in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias)

        self.input_mask = Parameter(torch.Tensor(in_channels, h_in, w_in))
        self.input_mask.data.fill_(1.0)

    def forward(self, input):
        print("###{}, {}".format(input.size(), self.input_mask.size()))
        return super(SparseConv2d, self).forward(input * self.input_mask)


# 添加输入掩码后的网络
class MyAlexNet(nn.Module):
    def __init__(self, h=224, w=224, conv_class=FixHWConv2d, linear_class=nn.Linear, num_classes=1000, dropout=True):
        super(MyAlexNet, self).__init__()
        feature_layers = []

        # conv
        feature_layers.append(conv_class(h, w, 3, 64, kernel_size=11, stride=4, padding=2))
        h = conv2d_out_dim(h, kernel_size=11, stride=4, padding=2)
        w = conv2d_out_dim(w, kernel_size=11, stride=4, padding=2)
        feature_layers.append(nn.ReLU(inplace=True))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        h = conv2d_out_dim(h, kernel_size=3, stride=2)
        w = conv2d_out_dim(w, kernel_size=3, stride=2)

        # conv
        feature_layers.append(conv_class(h, w, 64, 192, kernel_size=5, padding=2))
        h = conv2d_out_dim(h, kernel_size=5, padding=2)
        w = conv2d_out_dim(w, kernel_size=5, padding=2)
        feature_layers.append(nn.ReLU(inplace=True))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        h = conv2d_out_dim(h, kernel_size=3, stride=2)
        w = conv2d_out_dim(w, kernel_size=3, stride=2)

        # conv
        feature_layers.append(conv_class(h, w, 192, 384, kernel_size=3, padding=1))
        h = conv2d_out_dim(h, kernel_size=3, padding=1)
        w = conv2d_out_dim(w, kernel_size=3, padding=1)
        feature_layers.append(nn.ReLU(inplace=True))

        # conv
        feature_layers.append(conv_class(h, w, 384, 256, kernel_size=3, padding=1))
        h = conv2d_out_dim(h, kernel_size=3, padding=1)
        w = conv2d_out_dim(w, kernel_size=3, padding=1)
        feature_layers.append(nn.ReLU(inplace=True))

        # conv
        feature_layers.append(conv_class(h, w, 256, 256, kernel_size=3, padding=1))
        h = conv2d_out_dim(h, kernel_size=3, padding=1)
        w = conv2d_out_dim(w, kernel_size=3, padding=1)
        feature_layers.append(nn.ReLU(inplace=True))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        h = conv2d_out_dim(h, kernel_size=3, stride=2)
        w = conv2d_out_dim(w, kernel_size=3, stride=2)

        self.features = nn.Sequential(*feature_layers)

        fc_layers = []
        fc_layers.append(nn.Dropout(p=0.5 if dropout else 0.0))
        fc_layers.append(linear_class(256 * 6 * 6, 4096))
        fc_layers.append(nn.ReLU(inplace=True))

        fc_layers.append(nn.Dropout(p=0.5 if dropout else 0.0))
        fc_layers.append(linear_class(4096, 4096))
        fc_layers.append(nn.ReLU(inplace=True))

        fc_layers.append(linear_class(4096, num_classes))
        # fc_layers = [nn.Dropout(p=0.5 if dropout else 0.0),
        #              nn.Linear(256 * 6 * 6, 4096),
        #              nn.ReLU(inplace=True),
        #              nn.Dropout(p=0.5 if dropout else 0.0),
        #              nn.Linear(4096, 4096),
        #              nn.ReLU(inplace=True),
        #              nn.Linear(4096, num_classes)]

        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def get_inhw(self, x):
        res = []
        for module in self.features._modules.values():
            if isinstance(module, nn.Conv2d):
                res.append((x.size(2), x.size(3)))
            x = module(x)
        for module in self.classifier._modules.values():
            if isinstance(module, nn.Linear):
                res.append((1, 1))
        return res


# 不添加输入掩码的网络
class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 获取模型
def get_net_model(net='alexnet', dropout=False, input_mask=False):
    if input_mask:
        conv_class = SparseConv2d
    else:
        conv_class = FixHWConv2d
    model = MyAlexNet(dropout=dropout, conv_class=conv_class)
    teacher_model = AlexNet()

    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.eval()

    return model, teacher_model


def parse_results(input_file, coeffi):
    predict_time.clear()
    input_f = open(input_file, 'r')
    conv_signs = ['conv', 'res', 'cccp']
    fc_signs = ['ip', 'fc', 'innerproduct']
    for line in input_f:
        if len(line.strip()) == 0: continue
        if 'json' in line and 'Network' in line.split()[0]:
            print("\n%s" % line.split('/')[-1])
        layer_name = line.split()[0].lower()
        if any(conv in layer_name for conv in conv_signs):
            res = []
            items = line.split(':')
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[0])[-4:]  # Output
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[1])  # Filters
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[2])  # Padding
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[3])  # strides
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[5])  # Inputs
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[6])  # Input_nonzero
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[7])  # filter_nonzero
            # print(res)
            input = map(float, res)
            pre_runtime = predict_runtime('conv', input, coeffi)
            print("%s\t%.3f" % (layer_name, pre_runtime))
            layers_name.append(layer_name)
            predict_time.append(pre_runtime)
        if any(fc in layer_name for fc in fc_signs):
            res = []
            items = line.split(':')
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[0])[-4::3]  # Output
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[5])[-3:]  # Inputs
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[6])  # Input_nonzero
            res += re.findall(r"[-+]?\d*\.\d+|\d+", items[7])  # filter_nonzero
            input = map(float, res)
            pre_runtime = predict_runtime('fc', input, coeffi)
            print("%s\t%.3f" % (layer_name, pre_runtime))
            layers_name.append(layer_name)
            predict_time.append(pre_runtime)
    input_f.close()


def predict_runtime(type, input, coeffi):
    input = list(input)
    if type == 'conv':
        input_1 = input[0:3] + input[5:8] + input[13:14]
        input_2 = []
        for i in range(len(input_1)):
            for j in range(i, len(input_1)):
                input_2.append(input_1[i] * input_1[j])
        input_3 = []
        for i in range(len(input_1)):
            for j in range(i, len(input_1)):
                for k in range(j, len(input_1)):
                    input_3.append(input_1[i] * input_1[j] * input_1[k])
        p = input
        input_others = [p[1] * p[2] * p[3] * p[4] * p[5] * p[6],  # output pixels
                        p[12] * p[1] * p[2] * p[3] * p[4] * p[5] * p[6],
                        p[12] * p[1] * p[2] * p[4] * p[5] * p[6],
                        p[0] * p[1] * p[2] * p[3],  # output data
                        p[4] * p[5] * p[6] * p[7],  # filter data
                        p[12] * p[13] * p[14] * p[15],  # input data
                        p[12] * p[14] * p[15],  # input data
                        p[12] * p[13] * p[15],
                        p[16],
                        p[17],
                        p[16] * p[17]]  # input data
        input_runtime = input_1 + input_2 + input_3 + input_others + [1]  # 1 is the intercept
        runtime = sum(np.array(input_runtime) * np.array(coeffi[type, 'runtime']))
        return runtime
    if type == 'fc':
        input_1 = input
        input_2 = []
        for i in range(len(input_1)):
            for j in range(i, len(input_1)):
                input_2.append(input_1[i] * input_1[j])
        input_3 = []
        for i in range(len(input_1)):
            for j in range(i, len(input_1)):
                for k in range(j, len(input_1)):
                    input_3.append(input_1[i] * input_1[j] * input_1[k])
        p = input
        input_others = [p[0] * p[1] * p[2] * p[3] * p[4], p[5], p[6], p[5] * p[6]]  # operations pixels
        input_runtime = input_1 + input_2 + input_3 + input_others + [1]  # 1 is the intercept
        runtime = sum(np.array(input_runtime) * np.array(coeffi[type, 'runtime']))
        return runtime


def parse_coeff(coeffi):
    with open('/homework/lv/time_performance_optimization_for_CNN/result/AlexNet/coeff_conv.txt', 'r') as f:
        res = csv.reader(f)
        coeffi[('conv', 'runtime')] = list(map(float, next(res)))

    with open('/homework/lv/time_performance_optimization_for_CNN/result/AlexNet/coeff_fc.txt', 'r') as f:
        res = csv.reader(f)
        coeffi[('fc', 'runtime')] = list(map(float, next(res)))
    return coeffi


def cross_entropy(input, target, label_smoothing=0.0, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets (long tensor)
         size_average: if false, sum is returned instead of mean
    Examples::
        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)
        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    if label_smoothing <= 0.0:
        return F.cross_entropy(input, target)
    assert input.dim() == 2 and target.dim() == 1
    target_ = torch.unsqueeze(target, 1)
    one_hot = torch.zeros_like(input)
    one_hot.scatter_(1, target_, 1)
    one_hot = torch.clamp(one_hot, max=1.0-label_smoothing, min=label_smoothing/(one_hot.size(1) - 1.0))

    if size_average:
        return torch.mean(torch.sum(-one_hot * F.log_softmax(input, dim=1), dim=1))
    else:
        return torch.sum(torch.sum(-one_hot * F.log_softmax(input, dim=1), dim=1))


def joint_loss(model, data, target, teacher_model, distill, label_smoothing=0.0):
    criterion = lambda pred, y: cross_entropy(pred, y, label_smoothing=label_smoothing)
    output = model(data)
    if distill <= 0.0:
        return criterion(output, target)
    else:
        with torch.no_grad():
            teacher_output = teacher_model(data).data
        distill_loss = torch.mean((output - teacher_output) ** 2)
        if distill >= 1.0:
            return distill_loss
        else:
            class_loss = criterion(output, target)
            # print("distill loss={:.4e}, class loss={:.4e}".format(distill_loss, class_loss))
            return distill * distill_loss + (1.0 - distill) * class_loss


# 加载数据
def get_data_loaders(data_dir, dataset='imagenet', batch_size=32, val_batch_size=512, num_workers=0, nsubset=-1,
                     normalize=None):
    if dataset == 'imagenet':
        traindir = os.path.join(data_dir, 'train')
        valdir = os.path.join(data_dir, 'val')
        if normalize is None:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 如果是imagenet数据集，那么ImageNet的数据在加载的时候就已经转换成了[0, 1].[0.485, 0.456, 0.406]这一组平均值是从imagenet训练集中抽样算出来的。

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),  # 将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小；
                transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5；
                transforms.ToTensor(),  # 将给定图像转为Tensor
                normalize,  # 归一化处理
            ]))

        if nsubset > 0:  # 使用随机数据子集
            rand_idx = torch.randperm(len(train_dataset))[:nsubset]
            print('use a random subset of data:')
            print(rand_idx)
            train_sampler = SubsetRandomSampler(rand_idx)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=True, sampler=train_sampler)  # 训练数据集的加载器，自动将数据分割成batch，顺序随机打乱

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=val_batch_size, shuffle=False,  # shuffle() 方法将序列的所有元素随机排序。
            num_workers=num_workers, pin_memory=True)  # pin_memory就是锁页内存，创建DataLoader时，设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。

        # use 10K training data to see the training performance
        train_loader4eval = torch.utils.data.DataLoader(
            datasets.ImageFolder(traindir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=val_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            sampler=SubsetRandomSampler(torch.randperm(len(train_dataset))[:10000]))

        return train_loader, val_loader, train_loader4eval
    else:
        raise NotImplementedError


def ncorrect(output, target, topk=(1,)):
    """Computes the numebr of correct@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum().item()
        res.append(correct_k)
    return res


def eval_loss_acc1_acc5(model, data_loader, loss_func=None, cuda=True, class_offset=0):
    val_loss = 0.0
    val_acc1 = 0.0
    val_acc5 = 0.0
    num_data = 0
    with torch.no_grad():
        model.eval()
        for data, target in data_loader:
            num_data += target.size(0)
            target.data += class_offset
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            if loss_func is not None:
                val_loss += loss_func(model, data, target).item()
            # val_loss += F.cross_entropy(output, target).item()
            nc1, nc5 = ncorrect(output.data, target.data, topk=(1, 5))
            val_acc1 += nc1
            val_acc5 += nc5
            # print('acc:{}, {}'.format(nc1 / target.size(0), nc5 / target.size(0)))

    val_loss /= len(data_loader)
    val_acc1 /= num_data
    val_acc5 /= num_data

    return val_loss, val_acc1, val_acc5


def filtered_parameters(model, param_name, inverse=False):
    for name, param in model.named_parameters():
        if inverse != (name.endswith(param_name)):
            yield param


def model_sparsity(model, normalized=True, param_name='weight'):
    nnz = 0
    numel = 0
    for name, W in model.named_parameters():
        if name.endswith(param_name):
            W_nz = torch.nonzero(W.data)
            if W_nz.dim() > 0:
                nnz += W_nz.shape[0]
            numel += torch.numel(W.data)

    return float(nnz) / float(numel) if normalized else float(nnz)


def layers_stat(model, param_names=('weight',), param_filter=lambda p: True):
    input_nonzero = []
    count = 0
    if isinstance(param_names, str):
        param_names = (param_names,)
    def match_endswith(name):
        for param_name in param_names:
            if name.endswith(param_name):
                return param_name
        return None
    res = "########### layer stat ###########\n"
    for name, W in model.named_parameters():
        param_name = match_endswith(name)
        if param_name is not None:
            if param_filter(W):
                layer_name = name[:-len(param_name) - 1]
                W_nz = torch.nonzero(W.data)
                nnz = W_nz.shape[0] / W.data.numel() if W_nz.dim() > 0 else 0.0
                input_nonzero.append(nnz)
                W_data_abs = W.data.abs()
                res += "{:>20}".format(layer_name) + 'abs(W): mean={:.4e}, max={:.4e}, nnz={:.4f}\n'\
                    .format(W_data_abs.mean().item(), W_data_abs.max().item(), nnz)
    with open('/homework/lv/time_performance_optimization_for_CNN/result/AlexNet/alexnetData.txt', 'r') as f1:
        lines = f1.readlines()
        f2 = open('/homework/lv/time_performance_optimization_for_CNN/result/input_mask/inputmaskData.txt', 'w')
        for line in lines:
            if re.match("conv", line) or re.match("fc", line):
                f2.write(re.sub(r"input_nonzero:  +\d+(\.\d+)?", r"input_nonzero:  " + str(input_nonzero[count]), line))
                count += 1
        f2.close()
    res += "########### layer stat ###########"
    return res


def layers_grad_stat(model, param_name='weight'):
    res = "########### layer grad stat ###########\n"
    for name, W in model.named_parameters():
        if name.endswith(param_name):
            layer_name = name[:-len(param_name) - 1]
            W_nz = torch.nonzero(W.grad.data)
            nnz = W_nz.shape[0] / W.grad.data.numel() if W_nz.dim() > 0 else 0.0
            W_data_abs = W.grad.data.abs()
            res += "{:>20}".format(layer_name) + 'abs(W.grad): min={:.4e}, mean={:.4e}, max={:.4e}, nnz={:.4f}\n'.format(W_data_abs.min().item(), W_data_abs.mean().item(), W_data_abs.max().item(), nnz)

    res += "########### layer grad stat ###########"
    return res


def expand_user(path):
    return os.path.abspath(os.path.expanduser(path))


def model_snapshot(model, new_file, old_file=None, verbose=False):
    from collections import OrderedDict
    import torch
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if old_file and os.path.exists(expand_user(old_file)):
        if verbose:
            print("Removing old model {}".format(expand_user(old_file)))
        os.remove(expand_user(old_file))
    if verbose:
        print("Saving model to {}".format(expand_user(new_file)))

    state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if v.is_cuda:
            v = v.cpu()
        state_dict[k] = v
    torch.save(state_dict, expand_user(new_file))


def clamp_model_weights(model, min=0.0, max=1.0, param_name='input_mask'):
    for name, W in model.named_parameters():
        if name.endswith(param_name):
            W.data.clamp_(min=min, max=max)

    return model


def copy_model_weights(model, W_flat, W_shapes, param_name='weight'):
    offset = 0
    if isinstance(W_shapes, list):  # 如果对象的类型与参数二的类型（classinfo）相同则返回 True，否则返回 False。
        W_shapes = iter(W_shapes)
    for name, W in model.named_parameters():  # 返回对模块参数的迭代器，生成参数名称和参数本身。
        if name.endswith(param_name):
            name_, shape = next(W_shapes)
            if shape is None:
                continue
            assert name_ == name
            numel = W.numel()  # 返回数组中元素的个数
            W.data.copy_(W_flat[offset: offset + numel].view(shape))
            offset += numel


def l0proj(model, k, normalized=True, param_name='weight'):
    # get all the weights
    W_shapes = []
    res = []
    for name, W in model.named_parameters():
        if name.endswith(param_name):
            if W.dim() == 1:
                W_shapes.append((name, None))
            else:
                W_shapes.append((name, W.data.shape))
                res.append(W.data.view(-1))  # X.view(-1)中的-1本意是根据另外一个数来自动调整维度，但是这里只有一个维度，因此就会将X里面的所有维度数据转化成一维的，并且按先后顺序排列。

    res = torch.cat(res, dim=0)
    # print(res.shape[0])
    if normalized:
        assert 0.0 <= k <= 1.0
        nnz = math.floor(res.shape[0] * k)
    else:
        assert k >= 1 and round(k) == k
        nnz = k
    if nnz == res.shape[0]:
        z_idx = []
    else:
        _, z_idx = torch.topk(torch.abs(res), int(res.shape[0] - nnz), largest=False, sorted=False)
        res[z_idx] = 0.0
        copy_model_weights(model, res, W_shapes, param_name)
    return z_idx, W_shapes


def round_model_weights(model, param_name='input_mask'):
    for name, W in model.named_parameters():
        if name.endswith(param_name):
            W.data.round_()

    return model

if __name__ == '__main__':
    # 添加超参数
    parser = argparse.ArgumentParser(description='Model-Based Reasoning Time Constrained Training')
    parser.add_argument('--net', default='alexnet', help='network arch')  # 网络结构
    parser.add_argument('--dataset', default='imagenet', help='dataset used in the experiment')  # 数据集
    parser.add_argument('--datadir', default='/homework/lv/time_performance_optimization_for_CNN/dataSet/tiny-imagenet-200', help='dataset dir in this machine')  # 数据集路径
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')  # 训练批大小 raw_data:128
    parser.add_argument('--val_batch_size', type=int, default=512, help='batch size for evaluation')  # 评价批大小
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for training loader')  # 训练加载器线程数
    parser.add_argument('--epochs', type=int, default=8, help='number of epochs to train')  # 所有样本训练次数 raw_data:30
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # 学习率 raw_data:0.001
    parser.add_argument('--xlr', type=float, default=1e-4, help='learning rate for input mask')  # 输入掩码学习率 raw_data:1e-4
    parser.add_argument('--l2wd', type=float, default=1e-4, help='l2 weight decay')  # l2权重衰减 raw_data:1e-4
    parser.add_argument('--xl2wd', type=float, default=1e-5, help='l2 weight decay (for input mask)')  # l2权重衰减（输入掩码）raw_data:1e-5
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')  # 动量 动量的作用是为了在梯度下降中，加快下降的速度
    parser.add_argument('--proj_int', type=int, default=10, help='how many batches for each projection')  # 每个投影有多少批
    parser.add_argument('--nodp', default=True, help='turn off dropout')  # 在执行梯度检验时，请记住关闭网络中的任何不确定性影响，如丢失（dropout）、随机数据增强（random data augmentations）等。否则，在估计数值梯度时，这些明显会引入巨大的错误。关闭这些部分的缺点是不会对它们进行梯度检查（例如，可能是dropout没有正确地反向传播）。action='store_true'
    parser.add_argument('--input_mask', default=True, help='enable input mask')  # 启用输入掩码 action='store_true'，只要运行时该变量有传参就将该变量设为True
    parser.add_argument('--randinit', default=True, help='use random init')  # 启用随机初始化 action='store_true'
    parser.add_argument('--pretrain', default=False, help='file to load pretrained model')  # 加载预训练模型的文件
    parser.add_argument('--eval', action='store_true', help='evaluate testset in the begining')  # 开始时评估测试集
    parser.add_argument('--seed', type=int, default=117, help='random seed')  # 随机数种子
    parser.add_argument('--log_interval', type=int, default=100,
                        help='how many batches to wait before logging training status')  # 记录训练状态前要等待多少批 raw_data:100
    parser.add_argument('--test_interval', type=int, default=1, help='how many epochs to wait before another test')  # 在另一个测试之前要等待多少个epochs
    parser.add_argument('--save_interval', type=int, default=-1, help='how many epochs to wait before save a model')  # 保存一个模型要等待多少个epochs
    parser.add_argument('--logdir', default='/homework/lv/Time_performance_optimization_base_on_input_mask/log', help='folder to save to the log')  # 要保存到日志的文件夹
    parser.add_argument('--distill', type=float, default=0.5, help='distill loss weight')  # 知识蒸馏损失
    parser.add_argument('--budget', type=float, default=0.8, help='energy budget (relative)')  # 能源预算（相对）
    parser.add_argument('--exp_bdecay', action='store_true', help='exponential budget decay')  # 指数预算衰减
    parser.add_argument('--skip1', action='store_true', help='skip the first W update')  # 跳过第一次W更新 action='store_true'

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    # 添加日志
    if args.logdir is None:
        args.logdir = 'log/' + sys.argv[0] + str(datetime.datetime.now().strftime("_%Y_%m_%d_AT_%H_%M_%S"))

    args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    # 显示操作与超参数
    print('command:\npython {}'.format(' '.join(sys.argv)))
    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")

    # set up random seeds 在需要生成随机数据的实验中，每次实验都需要生成数据。设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了，有利于实验的比较和改进。
    torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    if args.cuda:
        torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子；
    np.random.seed(args.seed)
    random.seed(args.seed)

    # get training and validation data loaders 获取训练和验证数据加载程序
    normalize = None
    tr_loader, val_loader, train_loader4eval = get_data_loaders(data_dir=args.datadir,
                                                                dataset=args.dataset,
                                                                batch_size=args.batch_size,
                                                                val_batch_size=args.val_batch_size,
                                                                num_workers=args.num_workers,
                                                                normalize=normalize)
    # get network model 获取网络模型
    model, teacher_model = get_net_model(net=args.net,  dropout=(not args.nodp), input_mask=args.input_mask)

    # 当前时间
    print('================Model Reasoning Time Summary================')
    coeffi = {}
    parse_coeff(coeffi)
    cur_runtime = parse_results("/homework/lv/time_performance_optimization_for_CNN/result/input_mask/inputmaskData.txt", coeffi)
    print(cur_runtime)


    netl2wd = args.l2wd

    if args.cuda:
        if args.distill > 0.0:
            teacher_model.cuda()
        model.cuda()

    # 损失函数
    loss_func = lambda m, x, y: joint_loss(model=m, data=x, target=y, teacher_model=teacher_model, distill=args.distill)

    if args.eval or args.dataset != 'imagenet':
        val_loss, val_acc1, val_acc5 = eval_loss_acc1_acc5(model, val_loader, loss_func, args.cuda)
        print('**Validation loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}'.format(val_loss, val_acc1,
                                                                                              val_acc5))
        # also evaluate training data
        tr_loss, tr_acc1, tr_acc5 = eval_loss_acc1_acc5(model, train_loader4eval, loss_func, args.cuda)
        print('###Training loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}'.format(tr_loss, tr_acc1, tr_acc5))
    else:
        val_acc1 = 0.0
        print('For imagenet, skip the first validation evaluation.')

    old_file = None


    optimizer = torch.optim.SGD(filtered_parameters(model, param_name='input_mask', inverse=True), lr=args.lr, momentum=args.momentum, weight_decay=netl2wd)
    if args.input_mask:
        Xoptimizer = torch.optim.Adam(filtered_parameters(model, param_name='input_mask', inverse=False), lr=args.xlr, weight_decay=args.xl2wd)

    lr = args.lr
    xlr = args.xlr
    cur_sparsity = model_sparsity(model)

    best_acc_pruned = None
    Xbudget = 0.9
    iter_idx = 0

    W_proj_time = 0.0
    W_proj_time_cnt = 1e-15

    # 更新部分 算法
    while True:
        # update W 更新权重
        if not (args.skip1 and iter_idx == 0):  # 判断是不是跳过第一次W更新
            t_begin = time.time()
            log_tic = t_begin
            for epoch in range(args.epochs):
                for batch_idx, (data, target) in enumerate(tr_loader):
                    model.train()
                    if args.cuda:
                        data, target = data.cuda(), target.cuda()

                    loss = loss_func(model, data, target)
                    # update network weights
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if args.proj_int == 1 or (batch_idx > 0 and batch_idx % args.proj_int == 0) or batch_idx == len(tr_loader) - 1:
                        temp_tic = time.time()
                        W_proj_time += time.time() - temp_tic
                        W_proj_time_cnt += 1

                    if batch_idx % args.log_interval == 0:
                        print('======================================================')
                        print('+-------------- epoch {}, batch {}/{} ----------------+'.format(epoch, batch_idx,
                                                                                               len(tr_loader)))
                        log_toc = time.time()
                        print('primal update: net loss={:.4e}, lr={:.4e},  time_elapsed={:.3f}s, averaged projection_time {}'.format(
                                     loss.item(), optimizer.param_groups[0]['lr'], log_toc - log_tic, W_proj_time / W_proj_time_cnt))
                        log_tic = time.time()
                        if batch_idx % args.proj_int == 0:
                            cur_sparsity = model_sparsity(model)
                        print('sparsity:{}'.format(cur_sparsity))
                        print(layers_stat(model, param_names='weight', param_filter=lambda p: p.dim() > 1))
                        print('+-----------------------------------------------------+')

                cur_runtime = parse_results( "/homework/lv/time_performance_optimization_for_CNN/result/input_mask/inputmaskData.txt", coeffi)
                cur_sparsity = model_sparsity(model)
                print(cur_runtime)
                if epoch % args.test_interval == 0:
                    val_loss, val_acc1, val_acc5 = eval_loss_acc1_acc5(model, val_loader, loss_func, args.cuda)

                    # also evaluate training data
                    tr_loss, tr_acc1, tr_acc5 = eval_loss_acc1_acc5(model, train_loader4eval, loss_func, args.cuda)
                    print('###Training loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}'.format(tr_loss, tr_acc1,
                                                                                                         tr_acc5))

                    print(
                        '***Validation loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}, sparsity: {:.4e}'.format(
                            val_loss, val_acc1,
                            val_acc5, cur_sparsity))
                    # save current model
                    model_snapshot(model, os.path.join(args.logdir, 'primal_model_latest.pkl'))

                if args.save_interval > 0 and epoch % args.save_interval == 0:
                    model_snapshot(model, os.path.join(args.logdir, 'Wprimal_model_epoch{}_{}.pkl'.format(iter_idx, epoch)))

                elapse_time = time.time() - t_begin
                speed_epoch = elapse_time / (1 + epoch)
                eta = speed_epoch * (args.epochs - epoch)
                print("Updating Weights, Elapsed {:.2f}s, ets {:.2f}s".format(elapse_time, eta))

        if not args.input_mask:
            print("Complete weights training.")
            break
        else:
            print("Continue to train input mask.")

        if best_acc_pruned is not None and val_acc1 <= best_acc_pruned:
            print("Pruned accuracy does not improve, stop here!")
            break
        best_acc_pruned = val_acc1

        # update X
        t_begin = time.time()
        log_tic = t_begin
        for epoch in range(args.epochs):
            for batch_idx, (data, target) in enumerate(tr_loader):
                model.train()
                Xoptimizer.param_groups[0]['lr'] = xlr
                if args.cuda:
                    data, target = data.cuda(), target.cuda()

                loss = loss_func(model, data, target)
                # update network weights
                Xoptimizer.zero_grad()
                loss.backward()
                Xoptimizer.step()
                clamp_model_weights(model, min=0.0, max=1.0, param_name='input_mask')

                if (batch_idx > 0 and batch_idx % args.proj_int == 0) or batch_idx == len(tr_loader) - 1:
                    l0proj(model, Xbudget, param_name='input_mask')

                if batch_idx % args.log_interval == 0:
                    print('======================================================')
                    print('+-------------- epoch {}, batch {}/{} ----------------+'.format(epoch, batch_idx,
                                                                                           len(tr_loader)))
                    log_toc = time.time()
                    print('primal update: net loss={:.4e}, xlr={:.4e}, time_elapsed={:.3f}s'.format(
                            loss.item(), Xoptimizer.param_groups[0]['lr'], log_toc - log_tic))
                    log_tic = time.time()
                    if batch_idx % args.proj_int == 0:
                        cur_sparsity = model_sparsity(model, param_name='input_mask')
                    print('sparsity:{}'.format(cur_sparsity))
                    print(layers_stat(model, param_names='input_mask'))
                    print('+-----------------------------------------------------+')

            cur_runtime = parse_results(
                "/homework/lv/time_performance_optimization_for_CNN/result/input_mask/inputmaskData.txt", coeffi)
            print(cur_runtime)
            cur_sparsity = model_sparsity(model, param_name='input_mask')
            if epoch % args.test_interval == 0:

                val_loss, val_acc1, val_acc5 = eval_loss_acc1_acc5(model, val_loader, loss_func, args.cuda)

                # also evaluate training data
                tr_loss, tr_acc1, tr_acc5 = eval_loss_acc1_acc5(model, train_loader4eval, loss_func, args.cuda)
                print(
                    '###Training loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}'.format(tr_loss, tr_acc1,
                                                                                                   tr_acc5))

                print(
                    '***Validation loss:{:.4e}, top-1 accuracy:{:.5f}, top-5 accuracy:{:.5f}, sparsity: {:.4e}'.format(
                        val_loss, val_acc1,
                        val_acc5, cur_sparsity))
                # save current model
                model_snapshot(model, os.path.join(args.logdir, 'primal_model_latest.pkl'))

            if args.save_interval > 0 and epoch % args.save_interval == 0:
                model_snapshot(model, os.path.join(args.logdir, 'Xprimal_model_epoch{}_{}.pkl'.format(iter_idx, epoch)))

            elapse_time = time.time() - t_begin
            speed_epoch = elapse_time / (1 + epoch)
            eta = speed_epoch * (args.epochs - epoch)
            print("Updating input mask, Elapsed {:.2f}s, ets {:.2f}s".format(elapse_time, eta))

        round_model_weights(model, param_name='input_mask')
        # refresh X_energy_cache
        cur_runtime = parse_results(
            "/homework/lv/time_performance_optimization_for_CNN/result/input_mask/inputmaskData.txt", coeffi)
        print(cur_runtime)
        iter_idx += 1
        Xbudget -= 0.1
