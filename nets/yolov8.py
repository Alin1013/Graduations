import torch
import torch.nn as nn
from .CSPdarknet import CSPDarkNet  # 导入CSPDarknet主干网络
from .pan import PAN  # 导入带CBAM的PAN-FPN颈部
from .yolo_training import YOLOLoss  # 导入损失函数
from utils.utils_bbox import DecodeBox  # 导入解码模块


class YOLOv8(nn.Module):
    def __init__(self, num_classes, anchors, input_shape=[640, 640], cuda=True,
                 anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]], label_smoothing=0,
                 focal_loss=False, alpha=0.25, gamma=2):
        super().__init__()
        # 基础参数配置
        self.num_classes = num_classes
        self.anchors = anchors
        self.input_shape = input_shape
        self.cuda = cuda
        self.anchors_mask = anchors_mask
        self.focal_loss = focal_loss

        # 网络结构
        self.backbone = CSPDarkNet([1, 2, 8, 8, 4])  # CSPDarknet53主干网络
        self.neck = PAN()  # 带CBAM的PAN-FPN颈部

        # 检测头（3个尺度）- 与PAN输出特征维度匹配
        self.head = nn.ModuleList([
            nn.Conv2d(256, 3 * (num_classes + 5), 1),  # 小尺度特征输出 (52x52)
            nn.Conv2d(512, 3 * (num_classes + 5), 1),  # 中尺度特征输出 (26x26)
            nn.Conv2d(1024, 3 * (num_classes + 5), 1)  # 大尺度特征输出 (13x13)
        ])

        # 解码和损失函数
        self.decode_box = DecodeBox(anchors, num_classes, input_shape, anchors_mask)
        self.loss_fun = YOLOLoss(
            anchors, num_classes, input_shape, cuda,
            anchors_mask, label_smoothing,
            focal_loss=focal_loss, alpha=alpha, gamma=gamma
        )

    def forward(self, x, targets=None):
        # 主干网络提取特征 [P3, P4, P5]
        # P3: 52x52x256, P4:26x26x512, P5:13x13x1024
        features = self.backbone(x)

        # PAN-FPN特征融合 [P3_out, P4_out, P5_out]
        neck_outs = self.neck(features)

        # 检测头输出
        outputs = [self.head[i](neck_outs[i]) for i in range(3)]

        # 训练模式：计算损失
        if self.training:
            if targets is None:
                raise ValueError("训练模式下必须提供目标标签")
            total_loss = 0
            for i in range(3):
                # 适配YOLOLoss的forward参数要求
                loss = self.loss_fun(i, outputs[i], targets)
                total_loss += loss * self.loss_fun.balance[i]  # 应用损失平衡系数
            return total_loss
        # 推理模式：解码输出
        else:
            # 解码并合并多尺度结果
            outputs = self.decode_box.decode_box(outputs)
            return torch.cat(outputs, dim=1)

    def save(self, path):
        """保存模型权重"""
        torch.save(self.state_dict(), path)
        print(f"模型权重已保存至: {path}")

    def load(self, path):
        """加载模型权重"""
        device = torch.device('cuda' if self.cuda and torch.cuda.is_available() else 'cpu')
        try:
            self.load_state_dict(torch.load(path, map_location=device))
            print(f"成功加载模型权重: {path}")
            return self
        except Exception as e:
            print(f"模型权重加载失败: {e}")
            raise

    def evaluate(self, x):
        """评估模式（关闭dropout/bn推理模式）"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)