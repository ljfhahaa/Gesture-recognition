import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
class basicblock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, downsample=None):
        super(basicblock, self).__init__()
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=outchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
        x = self.relu(x)

        return x


class Myresnet18(nn.Module):
    def __init__(self,num_class=10):
        super(Myresnet18, self).__init__()
        block_num = [2, 2, 2, 2]
        self.num_class=num_class
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, block_num[0], stride=1)
        self.layer2 = self._make_layer(64, 128, block_num[1], stride=2)
        self.layer3 = self._make_layer(128, 256, block_num[2], stride=2)
        self.layer4 = self._make_layer(256, 512, block_num[3], stride=2)
        self.dyop1=nn.Dropout(0.5)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
        self.fc1 = nn.Linear(512, self.num_class)  

    def _make_layer(self, inchannel, outchannel, blocks, stride):
        downsample = None
        if stride != 1 or inchannel != outchannel:
            downsample = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

        layers = []
        layers.append(basicblock(inchannel, outchannel, stride, downsample))
        for _ in range(1, blocks):
            layers.append(basicblock(outchannel, outchannel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  
        x = torch.flatten(x, 1)  
        x=self.dyop1(x)
        x = self.fc1(x)  

        return x

# 测试
if __name__ == '__main__':
    model = Myresnet18()
    print(model)
    writer = SummaryWriter(log_dir="runs/MyResNet18")
    dummy_input = torch.randn(1, 3, 224, 224)
    writer.add_graph(model, dummy_input)
    writer.close()
    print("模型结构已写入 TensorBoard,使用 `tensorboard --logdir=runs` 查看")