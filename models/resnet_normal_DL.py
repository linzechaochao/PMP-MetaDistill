import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class SepConv(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        #   depthwise and pointwise convolution, downsample by 2
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.auxiliary1 = nn.Sequential(
            SepConv(
                channel_in=16 * block.expansion,
                channel_out=32 * block.expansion
            ),
            SepConv(
                channel_in=32 * block.expansion,
                channel_out=64 * block.expansion
            ),
            
            nn.AvgPool2d(4, 4)
        )

        self.auxiliary2 = nn.Sequential(
            SepConv(
                channel_in=32 * block.expansion,
                channel_out=64 * block.expansion,
            ),
            
            nn.AvgPool2d(4, 4)
        )
       
        self.auxiliary3 = nn.AvgPool2d(4, 4)

        # self.fc1 = nn.Linear(64 * block.expansion, num_classes)
        # self.fc2 = nn.Linear(64 * block.expansion, num_classes)
        # self.fc3 = nn.Linear(64 * block.expansion, num_classes)

        self.fc1 = nn.Linear(64 * 4, num_classes)
        self.fc2 = nn.Linear(64 * 4, num_classes)
        self.fc3 = nn.Linear(64 * 4, num_classes)
        
        # self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        self.feature_list = []
        self.out_feature_list = []
        self.out_list = []

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        self.feature_list.append(out)

        out = self.layer2(out)
        self.feature_list.append(out)

        out = self.layer3(out)
        self.feature_list.append(out)

        out1_feature = self.auxiliary1(self.feature_list[0]).view(x.size(0), -1)
        out2_feature = self.auxiliary2(self.feature_list[1]).view(x.size(0), -1)
        out3_feature = self.auxiliary3(self.feature_list[2]).view(x.size(0), -1)

        self.out_feature_list.append(out3_feature)
        self.out_feature_list.append(out2_feature)
        self.out_feature_list.append(out1_feature)

        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)
        
        self.out_list.append(out3)
        self.out_list.append(out2)
        self.out_list.append(out1)

        # out = F.avg_pool2d(out, out.size()[3])
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)

        return out3


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])