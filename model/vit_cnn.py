import timm
from torch import nn, einsum
import torch.nn.functional as F

import torch

pretrained_v = timm.create_model('vit_base_patch16_224', pretrained=True)
pretrained_v.head = nn.Linear(768, 2)
print(pretrained_v)

if __name__ == '__main__':
    # v = ViT(
    #     image_size=224,
    #     patch_size=4,
    #     num_classes=6,
    #     dim=48,
    #     depth=6,
    #     heads=8,
    #     mlp_dim=2048,
    #     dropout=0.1,
    #     emb_dropout=0.1
    # )
    # net = TFIResNet(3)

    img = torch.randn(2, 3, 224, 224)
    # mask = torch.ones(1, 8, 8).bool()  # optional mask, designating which patch to attend to

    preds = pretrained_v(img)  # (1, 1000)
    print(preds.shape)
    print(preds)
