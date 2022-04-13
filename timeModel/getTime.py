from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn import Parameter
from torch.hub import tqdm, load_state_dict_from_url as load_url

T_link = 60000000  # Link Time 200MHz
T_core = 200000000  # Core Time  60MHz
Cache_input = int(108 * 1024 * 8/ 2)  # input cache
Cache_filter = int(108 * 1024 * 8/ 2)  # Filter cache

default_t_mem = 1 / T_link * 10  # ms
default_t_cache = 1 / T_core * 10
default_t_rf = 1.0 / T_core * 10
default_t_mac = default_t_rf  # including both read and write RF

PE_h = 14  # PE array height
PE_w = 12  # PE array weight


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class myConv2d(nn.Conv2d):
    def __init__(self, h_in, w_in, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(myConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)
        self.h_in = h_in
        self.w_in = w_in
        self.s = Parameter(torch.LongTensor(1), requires_grad=False)
        self.s.data[0] = stride
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


class SparseConv2d(myConv2d):
    def __init__(self, h_in, w_in, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(SparseConv2d, self).__init__(h_in, w_in, in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias)

        self.input_mask = Parameter(torch.Tensor(in_channels, h_in, w_in))
        self.input_mask.data.fill_(1.0)

    def forward(self, input):
        # print("###{}, {}".format(input.size(), self.input_mask.size()))
        return super(SparseConv2d, self).forward(input * self.input_mask)


class MyAlexNet(nn.Module):
    def __init__(self, h=227, w=227, conv_class=FixHWConv2d, num_classes=1000, dropout=True):
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

        fc_layers = [nn.Dropout(p=0.5 if dropout else 0.0),
                     nn.Linear(256 * 6 * 6, 4096),
                     nn.ReLU(inplace=True),
                     nn.Dropout(p=0.5 if dropout else 0.0),
                     nn.Linear(4096, 4096),
                     nn.ReLU(inplace=True),
                     nn.Linear(4096, num_classes)]

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


def myalexnet(pretrained=False, model_root=None, **kwargs):
    model = MyAlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['alexnet'], model_root), strict=False)
    return model


def conv2d_out_dim(dim, kernel_size, padding=0, stride=1, dilation=1, ceil_mode=False):
    if ceil_mode:
        return int(math.ceil((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
    else:
        return int(math.floor((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))


def get_net_model(net='alexnet', pretrained_dataset='imagenet', dropout=False, pretrained=True, input_mask=False):
    if input_mask:
        conv_class = SparseConv2d
    else:
        conv_class = FixHWConv2d
    if net == 'alexnet':
        model = myalexnet(pretrained=(pretrained_dataset == 'imagenet') and pretrained, dropout=dropout, conv_class=conv_class)
    else:
        raise NotImplementedError

    return model


class Layer(object):
    def __init__(self, **kwargs):
        super(Layer, self).__init__()
        self.h = kwargs['h'] if 'h' in kwargs else None  # input height
        self.w = kwargs['w'] if 'w' in kwargs else None  # input width
        self.c = kwargs['c'] if 'c' in kwargs else None  # channel number
        self.d = kwargs['d'] if 'd' in kwargs else None  # filter number
        self.s = kwargs['s'] if 's' in kwargs else None  # stride
        self.p = kwargs['p'] if 'p' in kwargs else None  # padding
        self.g = kwargs['g'] if 'g' in kwargs else None  # group
        self.peh = kwargs['peh'] if 'peh' in kwargs else None  # PE array height
        self.pew = kwargs['pew'] if 'pew' in kwargs else None  # PE array width
        self.ci = kwargs['ci'] if 'ci' in kwargs else None  # Input Cache
        self.cf = kwargs['cf'] if 'cf' in kwargs else None  # Filter Cache
        self.r = kwargs['r'] if 'r' in kwargs else None  # filter height and width
        self.is_conv = True if self.r is not None else False  # is conv layer

        if self.h is not None:
            self.h_ = max(0.0, math.floor((self.h + 2.0 * self.p - self.r) / float(self.s)) + 1)  # output height
        if self.w is not None:
            self.w_ = max(0.0, math.floor((self.w + 2.0 * self.p - self.r) / float(self.s)) + 1)  # output width


def layers_nnz(model, normalized=True, param_name='weight'):
    res = {}
    nnz = []
    for name, W in model.named_parameters():
        if name.endswith(param_name) and name.startswith("features"):
            layer_name = name[:-len(param_name)-1]
            W_nz = torch.nonzero(W.data)
            nnz.append(float(W_nz.shape[0]) / torch.numel(W))
            # print("{} layer nnz:{}".format(name, W_nz.shape[0]/torch.numel(W)))
            if W_nz.dim() > 0:
                if not normalized:
                    res[layer_name] = W_nz.shape[0]
                else:
                    print("{} layer nnz:{}".format(name, torch.nonzero(W.data)))
                    res[layer_name] = float(W_nz.shape[0]) / torch.numel(W)
            else:
                res[layer_name] = 0
    # print(nnz)
    return res, nnz


def conv_cache_overlap(X_supp, padding, kernel_size, stride, k_X):
    rs = X_supp.transpose(0, 1).contiguous().view(X_supp.size(1), -1).sum(dim=1).cpu()
    rs = torch.cat([torch.zeros(padding, dtype=rs.dtype, device=rs.device),
                   rs,
                    torch.zeros(padding, dtype=rs.dtype, device=rs.device)])
    res = 0
    beg = 0
    end = None
    while beg + kernel_size - 1 < rs.size(0):
        if end is not None:
            if beg < end:
                res += rs[beg:end].sum().item()
        n_elements = 0
        for i in range(rs.size(0) - beg):
            if n_elements + rs[beg+i] <= k_X:
                n_elements += rs[beg+i]
                if beg + i == rs.size(0) - 1:
                    end = rs.size(0)
            else:
                end = beg + i
                break
        assert end - beg >= kernel_size, 'can only hold {} rows with {} elements < {} rows in {}, cache size={}'.format(end - beg, n_elements, kernel_size, X_supp.size(), k_X)
        # print('map size={}. begin={}, end={}'.format(X_supp.size(), beg, end))
        beg += (math.floor((end - beg - kernel_size) / stride) + 1) * stride
    return res


def build_layer_info(model, peh=PE_h, pew=PE_w, ci=Cache_input, cf=Cache_filter):
    res = {}
    for name, p in model.named_parameters():
        if name.endswith('input_mask'):
            layer_name = name[:-len('input_mask') - 1]
            if layer_name in res:
                res[layer_name]['h'] = p.size()[1]
                res[layer_name]['w'] = p.size()[2]
            else:
                res[layer_name] = {'h': p.size()[1], 'w': p.size()[2]}
        elif name.endswith('.hw'):
            layer_name = name[:-len('hw') - 1]
            if layer_name in res:
                res[layer_name]['h'] = float(p.data[0])
                res[layer_name]['w'] = float(p.data[1])
            else:
                res[layer_name] = {'h': float(p.data[0]), 'w': float(p.data[1])}
        elif name.endswith('.s'):
            layer_name = name[:-len('s') - 1]
            if layer_name in res:
                res[layer_name]['s'] = float(p.data[0])
            else:
                res[layer_name] = {'s': float(p.data[0])}
        elif name.endswith('.g'):
            layer_name = name[:-len('g') - 1]
            if layer_name in res:
                res[layer_name]['g'] = float(p.data[0])
            else:
                res[layer_name] = {'g': float(p.data[0])}
        elif name.endswith('.p'):
            layer_name = name[:-len('p') - 1]
            if layer_name in res:
                res[layer_name]['p'] = float(p.data[0])
            else:
                res[layer_name] = {'p': float(p.data[0])}
        elif name.endswith('weight'):
            if len(p.size()) == 2 or len(p.size()) == 4:
                layer_name = name[:-len('weight') - 1]
                if layer_name in res:
                    res[layer_name]['d'] = p.size()[0]
                    res[layer_name]['c'] = p.size()[1]
                else:
                    res[layer_name] = {'d': p.size()[0], 'c': p.size()[1]}
                if p.dim() > 2:
                    # (out_channels, in_channels, kernel_size[0], kernel_size[1])
                    assert p.dim() == 4
                    res[layer_name]['r'] = p.size()[2]
        else:
            continue

        res[layer_name]['peh'] = peh
        res[layer_name]['pew'] = pew
        res[layer_name]['ci'] = ci
        res[layer_name]['cf'] = cf

    for layer_name in res:
        res[layer_name] = Layer(**(res[layer_name]))
        if res[layer_name].g is not None and res[layer_name].g > 1:
            res[layer_name].c *= res[layer_name].g
    return res


def get_time(model, layer_info, t_mac=default_t_mac, t_DRAM=default_t_mem, t_GLB=default_t_cache,
                t_PE=default_t_rf, verbose=False, crelax=True):
    res = {}
    X_nnz_dict, X_nnz_ratio = layers_nnz(model, normalized=False, param_name='input_mask')
    W_nnz_dict, W_nnz_ratio = layers_nnz(model, normalized=False, param_name='weight')
    print(X_nnz_ratio, W_nnz_ratio)
    X_supp_dict = {}
    T_comm = []
    T_comp = []

    for name, p in model.named_parameters():
        if name.endswith('input_mask') and name.startswith("features"):
            layer_name = name[:-len('input_mask') - 1]
            X_supp_dict[layer_name] = (p.data != 0.0).float()
    for name, p in model.named_parameters():
        if name.endswith('weight') and name.startswith("features"):
            if p is None or p.dim() == 1:
                continue
            layer_name = name[:-len('weight') - 1]
            layer = layer_info[layer_name]

            W_nnz = W_nnz_dict[layer_name]
            if len(X_nnz_dict) == 0:
                X_nnz = layer.h * layer.w * layer.c
            else:
                X_nnz = X_nnz_dict[layer_name]

            if layer_name in X_supp_dict:
                X_supp = X_supp_dict[layer_name].unsqueeze(0)
            else:
                if layer.is_conv:
                    X_supp = torch.ones(1, int(layer.c), int(layer.h), int(layer.w), dtype=p.dtype, device=p.device)
                else:
                    X_supp = None


            # communcation time
            if layer.is_conv:
                h_, w_ = max(0.0, math.floor((layer.h + 2 * layer.p - layer.r) / layer.s) + 1), max(0.0, math.floor(
                    (layer.w + 2 * layer.p - layer.r) / layer.s) + 1)
                X_time_DRAM = h_ * w_ * layer.c * t_DRAM
                W_time_DRAM = layer.r * layer.r * layer.c * layer.d * t_DRAM
                X_time_PE = X_nnz * t_PE
                W_time_PE = W_nnz * t_PE
                X_time_GLB = 0
                W_time_GLB = 0
                T_comm.append(X_time_DRAM + W_time_DRAM + X_time_PE + W_time_PE + X_time_GLB + W_time_GLB)

            # computation time
            if layer.is_conv:
                if crelax:
                    N_mac = layer_info[layer_name].h_ * float(layer_info[layer_name].w_) * W_nnz_dict[layer_name]
                else:
                    N_mac = torch.sum(
                        F.conv2d(X_supp, (p.data != 0.0).float(), None, int(layer_info[layer_name].s),
                                 int(layer_info[layer_name].p), 1, int(layer_info[layer_name].g))).item()
                T_comp.append(N_mac * t_mac)
            if layer.is_conv:
                # print("Layer: {}, W_time={:.2e}, C_time={:.2e}, X_time={:.2e}, SUM={:.2e}".format(layer_name,
                # W_time[-1], C_time[-1],X_time[-1],W_time[-1]+C_time[-1]+X_time[-1]),)
                print("Layer:{}, Runtime: {} ".format(layer_name, T_comp[-1] + T_comm[-1]))
                res[layer_name] =round(T_comp[-1] + T_comm[-1], 2)
    # print(res)
    return res



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model-Based Energy Constrained Training')
    parser.add_argument('--net', default='alexnet', help='network arch')  # 网络结构

    parser.add_argument('--dataset', default='imagenet', help='dataset used in the experiment')  # 数据集
    parser.add_argument('--datadir', default='/homework/lv/Time analysis and Optimization Based on CNN accelerator/dataSet/imageNet', help='dataset dir in this machine')  # 数据集路径

    parser.add_argument('--nodp', action='store_true',
                        help='turn off dropout')  # 在执行梯度检验时，请记住关闭网络中的任何不确定性影响，如丢失（dropout）、随机数据增强（random data augmentations）等。否则，在估计数值梯度时，这些明显会引入巨大的错误。关闭这些部分的缺点是不会对它们进行梯度检查（例如，可能是dropout没有正确地反向传播）。
    parser.add_argument('--input_mask', action='store_true',
                        help='enable input mask')  # 启用输入掩码 action='store_true'，只要运行时该变量有传参就将该变量设为True

    parser.add_argument('--randinit', action='store_true', help='use random init')  # 启用随机初始化
    args = parser.parse_args()
    model = get_net_model(net=args.net, pretrained_dataset=args.dataset, dropout=(not args.nodp),
                                         pretrained=not args.randinit, input_mask=args.input_mask)
    # print(model)
    layer = build_layer_info(model)
    cur_time = sum(get_time(model, layer, verbose=True).values())
    print("Total_Time:{:.4}".format(cur_time))
    with open('/homework/lv/Time analysis and Optimization Based on CNN accelerator/result/AlexNet/AlexNetData.txt', 'w') as f:
        f.write(str(cur_time))
