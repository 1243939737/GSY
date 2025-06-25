import torch
import torch.nn as nn

class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        B, C, H, W = x.shape
        x1 = x[..., ::2, ::2]
        x2 = x[..., 1::2, ::2]
        x3 = x[..., ::2, 1::2]
        x4 = x[..., 1::2, 1::2]
        # 使用上采样恢复到原始大小
        x1 = torch.nn.functional.interpolate(x1, size=(H, W), mode='nearest')
        x2 = torch.nn.functional.interpolate(x2, size=(H, W), mode='nearest')
        x3 = torch.nn.functional.interpolate(x3, size=(H, W), mode='nearest')
        x4 = torch.nn.functional.interpolate(x4, size=(H, W), mode='nearest')
        return torch.cat([x1, x2, x3, x4], dim=1)
