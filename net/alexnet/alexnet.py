import time
import torch.nn as nn
import torch.nn.functional as F


layer_time = []
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
        begin_time = time.time()
        x = self.conv1(x)
        layer_time.append(round(1000 * (time.time() - begin_time), 4))
        x = F.relu(x, True)
        layer_time.append(round(1000 * (time.time() - begin_time), 4))
        x = F.max_pool2d(x, 3, 2)
        layer_time.append(round(1000 * (time.time() - begin_time), 4))
        x = self.conv2(x)
        layer_time.append(round(1000 * (time.time() - begin_time), 4))
        x = F.relu(x, True)
        layer_time.append(round(1000 * (time.time() - begin_time), 4))
        x = F.max_pool2d(x, 3, 2)
        layer_time.append(round(1000 * (time.time() - begin_time), 4))
        x = self.conv3(x)
        layer_time.append(round(1000 * (time.time() - begin_time), 4))
        x = F.relu(x, True)
        layer_time.append(round(1000 * (time.time() - begin_time), 4))
        x = self.conv4(x)
        layer_time.append(round(1000 * (time.time() - begin_time), 4))
        x = F.relu(x, True)
        layer_time.append(round(1000 * (time.time() - begin_time), 4))
        x = self.conv5(x)
        layer_time.append(round(1000 * (time.time() - begin_time), 4))
        x = F.relu(x, True)
        layer_time.append(round(1000 * (time.time() - begin_time), 4))
        x = F.max_pool2d(x, 3, 2)
        layer_time.append(round(1000 * (time.time() - begin_time), 4))
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = self.fc6(x)
        layer_time.append(round(1000 * (time.time() - begin_time), 4))
        x = F.dropout(x, 0.5)
        layer_time.append(round(1000 * (time.time() - begin_time), 4))
        x = F.relu(x, True)
        layer_time.append(round(1000 * (time.time() - begin_time), 4))
        x = self.fc7(x)
        layer_time.append(round(1000 * (time.time() - begin_time), 4))
        x = F.dropout(x, 0.5)
        layer_time.append(round(1000 * (time.time() - begin_time), 4))
        x = F.relu(x, True)
        layer_time.append(round(1000 * (time.time() - begin_time), 4))
        x = self.fc8(x)
        layer_time.append(round(1000 * (time.time() - begin_time), 4))
        return x
