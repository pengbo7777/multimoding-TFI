import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import model.resnest.torch.resnest as ResNest
# import encoding
# import torchvision.models as resnet
import model.resnest.torch.resnest as resnest


class Net(nn.Module):
    def __init__(self, nclass, backbone='resnet50'):
        super(Net, self).__init__()
        self.backbone = backbone

        if backbone == 'resnet50':
            self.resnest50 = resnest.resnest50(pretrained=False)
            self.feature_dim=2048
        n_codes = 8

        # 添加vlad模块
        self.num_clusters = 8
        self.dim = 128
        self.alpha = 1.0
        self.normalize_input = True
        self.conv = nn.Conv2d(self.dim, self.num_clusters, kernel_size=(1, 1), bias=True)
        self.conv_1_1 = nn.Conv2d(self.feature_dim, self.dim, kernel_size=(1, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(self.feature_dim)
        self.bn2 = nn.BatchNorm2d(self.dim)
        self.relu = nn.ReLU()
        self.centroids = nn.Parameter(torch.rand(self.num_clusters, self.dim))
        self._init_params()
        self.vlad_liner = nn.Linear(self.dim * self.num_clusters, 64)
        # self.resNest = ResNest.resnest50(pretrained=True)
        # self.head = nn.Sequential(
        #     # nn.Conv2d(512, 512, 1),
        #     nn.BatchNorm2d(512),
        #     # nn.ReLU(inplace=True),
        #     encoding.nn.Encoding(D=512, K=n_codes),
        #     encoding.nn.View(-1, 512 * n_codes),
        #     encoding.nn.Normalize(),
        #     nn.Linear(512 * n_codes, 64),
        #     # nn.BatchNorm1d(64),
        # )
        # self.vlad = nn.Sequential(
        #     nn.BatchNorm2d(512),
        #     nn.Conv2d(512, 128, kernel_size=(1, 1), bias=True)
        # )
        # self.pool = nn.Sequential(
        #     nn.AvgPool2d(7),
        #     encoding.nn.View(-1, 512),
        #     nn.Linear(512, 64),
        #     nn.BatchNorm1d(64),
        # )

        self.avgPool = nn.AvgPool2d(7)
        # self.pool_view = encoding.nn.View(-1, 512)
        self.poolLiner = nn.Linear(self.feature_dim, 64)
        self.poolBatchNorm1d = nn.BatchNorm1d(64)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(64 * 64),
            # nn.Linear(64 * 64, 128),
            # nn.BatchNorm1d(128),
            nn.Linear(64*64, nclass))

        # self.fc_norm = nn.BatchNorm1d()
        # self.fc_liner1 = nn.Linear(64 * 64, 128)
        # self.fc_liner2 = nn.Linear(128,nclass)
        # self.head = nn.Sequential(
        #     nn.Conv2d(512, 128, 1),
        #     encoding.nn.GramMatrix(),
        #     encoding.nn.View(-1, 128*128),
        #     nn.Linear(128*128, nclass)
        #     )
        self.se = SELayer(512)

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            var_input = x
            while not isinstance(var_input, Variable):
                var_input = var_input[0]
            _, _, h, w = var_input.size()
        else:
            raise RuntimeError('unknown input type: ', type(x))

        if self.backbone == 'resnet50' or self.backbone == 'resnet101' \
                or self.backbone == 'resnet152':
            # pre-trained ResNet feature
            # x = self.pretrained.conv1(x)
            # x = self.pretrained.bn1(x)
            # x = self.pretrained.relu(x)
            # x = self.pretrained.maxpool(x)
            # x = self.pretrained.layer1(x)
            # x = self.pretrained.layer2(x)
            # x = self.pretrained.layer3(x)
            # x_source = self.pretrained.layer4(x)

            # x_source = self.resnest50(x)
            x = self.resnest50.conv1(x)
            x = self.resnest50.bn1(x)
            x = self.resnest50.relu(x)
            x = self.resnest50.maxpool(x)

            x = self.resnest50.layer1(x)
            x = self.resnest50.layer2(x)
            x = self.resnest50.layer3(x)
            x_source = self.resnest50.layer4(x)
            print(x_source.shape)
            # x_source = self.se(x_source)
            # x1 = self.head(x)
            print(x_source[0].shape)
            print(x_source[1].shape)
            print(x_source[:,:,:2,:2].shape)
            print(x_source[:,:,2:4,:2].shape)
            print(x_source[:,:,4:7,:3].shape)

            print(x_source[:,:,:2,2:4].shape)
            print(x_source[:,:,2:4,2:4].shape)


            print(x_source[:,:,:3,4:7].shape)
            print(x_source[:,:,4:7,4:7].shape)

            # print(x_source[:,:,:3,4:7].shape)
            # x_source = x_source[:,:,:2,:2]

            # print(x_source[:,:,:2,:2].shape)
            # print(x_source[:,:,:,0].shape)
            # print(x_source[2].shape)
            # vlad
            if self.normalize_input:
                x = F.normalize(x_source, p=2, dim=1)  # across descriptor dim
            print("hahah")
            print(x.shape)
            x = self.conv_1_1(x)
            x = self.bn2(x)
            x = self.relu(x)
            print(x.shape)
            N, C = x.shape[:2]
            print(N,C)

            # soft-assignment
            # [20, 2048, 7, 7]
            # print("x初始shape：", x.shape)
            soft_assign = self.conv(x)
            # print(soft_assign.shape)
            soft_assign = soft_assign.view(N, self.num_clusters, -1)
            print("soft_assign.shape")
            print(soft_assign.shape)
            soft_assign = F.softmax(soft_assign, dim=1)
            print(soft_assign.shape)
            x_flatten = x.view(N, C, -1)  # C=2048
            print("x_flatten.shape")
            print(x_flatten.shape)
            # calculate residuals to each clusters
            residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                       self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            # print("residual.shape")
            # print(residual.shape)
            residual *= soft_assign.unsqueeze(2)
            vlad = residual.sum(dim=-1)
            # print(vlad.shape)
            vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
            vlad = vlad.view(x.size(0), -1)  # flatten
            # print(vlad.shape)
            x1 = F.normalize(vlad, p=2, dim=1)  # L2 normalize

            # ---end
            # print("x1：")
            # print(x1.shape)
            # x2 = self.pool(x)
            x1 = self.vlad_liner(x1)
            print(x_source.shape)
            x2 = self.avgPool(x_source)
            print(x2.shape)
            x2 = x2.view(-1, 2048)
            print(x2.shape)
            x2 = self.poolLiner(x2)
            x2 = self.poolBatchNorm1d(x2)

            print(x2.shape)
            print(x1.shape)
            x1 = x1.unsqueeze(1).expand(x1.size(0), x2.size(1), x1.size(-1))

            print(x1.shape)
            x = x1 * x2.unsqueeze(-1)
            x = x.view(-1, x1.size(-1) * x2.size(1))
            print(x.shape)
            x = self.fc(x)
        else:
            x = self.pretrained(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def test():
    net = Net(nclass=3)
    # print(net)
    x = Variable(torch.randn(2, 3, 224, 224))
    # x = Variable(torch.randn(64, 128))
    # x2 = Variable(torch.randn(64, 128))

    # x3 = x + 0.2* x2

    # print(x3.shape)

    y = net(x)
    print(y.shape)
    # x1 = Variable(torch.randn(2, 3,3))
    # x2 = Variable(torch.randn(2, 2))
    # print(x1)
    # x1 = F.softmax(x1, dim=1)
    # print(x1)
    # print(x1)
    # print(x2)
    # print(x2.unsqueeze(-1))
    # print(x1)
    # print(x1.unsqueeze(1))
    # print(x1.unsqueeze(1).expand(x1.size(0), x2.size(1), x1.size(-1)))
    # print(y.shape)
    # params = net.parameters()
    # sum = 0
    # for param in params:
    #     sum += param.nelement()
    # print('Total params:', sum)


if __name__ == "__main__":
    test()
