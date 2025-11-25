import torch
import torch.nn as nn
from .CSPdarknet import CSPDarknet  # 导入主干网络
from .pan import PAN  # 导入带CBAM的PAN-FPN

class YOLOv8(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = CSPDarknet()  # CSPDarknet主干
        self.neck = PAN()  # 带CBAM的PAN-FPN颈部
        # 头部（保持原检测头结构，此处简化）
        self.head = nn.ModuleList([
            nn.Conv2d(256, 3*(num_classes+5), 1)  # 每个尺度输出：3锚框x(类别数+5)
            for _ in range(3)
        ])

    def forward(self, x):
        # 主干网络提取特征
        features = self.backbone(x)  # [P3, P4, P5]
        # PAN-FPN融合（带CBAM增强）
        neck_outs = self.neck(features)  # [P3, P4_up, P5_up]
        # 检测头输出
        outputs = [self.head[i](neck_outs[i]) for i in range(3)]
        return outputs

    def save(self, path):
        """保存模型"""
        torch.save(self.state_dict(), path)

