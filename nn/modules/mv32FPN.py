import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# 1x1卷积层带BN
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


# nxn卷积层带BN（扩张卷积）
def conv_nxn_bn(inp, oup, kernel_size=5, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


# PreNorm模块
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# Attention模块
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


# FeedForward模块
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# Transformer模块
class MBTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# FPN模块（特征金字塔网络）
class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_channels[i], out_channels, kernel_size=1) for i in range(len(in_channels))])
        self.smooth_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in range(len(in_channels) - 1)]
        )

    def forward(self, features):
        laterals = [conv(feature) for conv, feature in zip(self.lateral_convs, features)]
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] += F.interpolate(laterals[i + 1], size=laterals[i].shape[2:], mode="nearest")

        out_features = [conv(laterals[i]) for i, conv in enumerate(self.smooth_convs)]
        return out_features


# MV2Block模块（MobileNetV2模块）
class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expansion == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# MobileViTBv3模型
class MobileViTBv3(nn.Module):
    def __init__(self, channel, dim, depth=2, kernel_size=3, patch_size=(2, 2), mlp_dim=int(64 * 2), dropout=0.1):
        super().__init__()
        self.ph, self.pw = patch_size
        self.mv01 = MV2Block(channel, channel)
        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv2 = conv_1x1_bn(channel, dim)
        self.transformer = MBTransformer(dim, depth, 4, 8, mlp_dim, dropout)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
        self.fpn = FPN([channel, dim], channel)  # 加入FPN模块

    def forward(self, x):
        y = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        z = x.clone()
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)
        x = self.conv3(x)

        # FPN输出
        features = self.fpn([x])
        x = features[-1]  # 使用FPN最后的特征图

        x = self.mv01(x)
        return x
