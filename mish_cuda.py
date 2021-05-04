# https://github.com/WongKinYiu/ScaledYOLOv4/issues/94#issuecomment-745233661

import torch
from torch import nn


class MishCuda(nn.Module):
    def forward(self, x):
        return x * torch.nn.functional.softplus(x).tanh()
