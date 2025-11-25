import torch
import torch.nn as nn

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化获取通道信息
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化获取通道信息
        x = torch.cat([avg_out, max_out], dim=1)  # 拼接两种池化结果
        return self.sigmoid(self.conv1(x))  # 卷积+激活得到空间注意力图

# CBAM模块（通道+空间注意力）
class CBAM(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channel, ratio)  # 通道注意力
        self.spatial_att = SpatialAttention(kernel_size)  # 空间注意力

    def forward(self, x):
        x = x * self.channel_att(x)  # 先应用通道注意力
        x = x * self.spatial_att(x)  # 再应用空间注意力
        return x