import torch
import torch.nn as nn
from .attention import CBAM  # 导入CBAM模块

class Conv(nn.Module):
    """基础卷积块：卷积+BN+激活"""
    def __init__(self, c1, c2, k=1, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()  # YOLOv8默认激活函数

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class PANBlock(nn.Module):
    """PAN-FPN融合块，添加CBAM注意力"""
    def __init__(self, c1, c2):
        super().__init__()
        self.conv1 = Conv(c1, c2, 1)  # 降维卷积
        self.conv2 = Conv(c2, c2, 3, 1, 1)  # 特征提取卷积
        self.cbam = CBAM(c2)  # 添加CBAM模块（关键）

    def forward(self, x, y):
        """
        x: 上层特征（小尺寸，高语义）
        y: 当前层特征（大尺寸，低语义）
        """
        x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x)  # 上采样
        x = torch.cat([x, self.conv1(y)], dim=1)  # 特征拼接
        x = self.conv2(x)  # 融合特征
        x = self.cbam(x)  # 应用CBAM增强特征
        return x

class PAN(nn.Module):
    """PAN-FPN整体结构"""
    def __init__(self, in_channels=[256, 512, 1024], out_channels=256):
        super().__init__()
        # 定义3个PAN融合块（对应3个尺度特征）
        self.pan_block1 = PANBlock(in_channels[2] + in_channels[1], out_channels)
        self.pan_block2 = PANBlock(out_channels + in_channels[0], out_channels)

    def forward(self, features):
        """
        features: 主干网络输出的3个尺度特征 [P3, P4, P5]
        """
        P3, P4, P5 = features  # P3:52x52, P4:26x26, P5:13x13
        # 自顶向下融合
        P5_up = self.pan_block1(P5, P4)  # 融合P5和P4，输出26x26
        P4_up = self.pan_block2(P5_up, P3)  # 融合P5_up和P3，输出52x52
        return [P3, P4_up, P5_up]  # 返回增强后的多尺度特征