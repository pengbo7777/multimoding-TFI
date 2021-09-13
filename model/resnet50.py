import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torchvision.models as resnet


class Net(nn.Module):
    def __init__(self, nclass, backbone='resnet50'):
        super(Net, self).__init__()
        self.backbone = backbone
        # copying modules from pretrained models
        self.pretrained = resnet.resnet50(pretrained=True)
        self.fc1 = nn.Linear(2048, nclass)


    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        print(x)
        np.save('../show/conv1.npy',x.detach().numpy())
        x = self.pretrained.maxpool(x)

        x = self.pretrained.layer1(x)
        print(x)
        print(x.shape)
        np.save('../show/conv2.npy', x.detach().numpy())
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x)

        x = self.pretrained.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        return x



def test():
    net = Net(nclass=3)
    # print(net)
    # x = torch.randn(1, 3, 224, 224)
    x = np.load("../show/tensor.npy")
    x = torch.from_numpy(x)
    x = x.unsqueeze(0)
    # print(x.shape)
    y = net(x)
    print(y)
    # params = net.parameters()
    # sum = 0
    # for param in params:
    #     sum += param.nelement()
    # print('Total params:', sum)


if __name__ == "__main__":
    test()
