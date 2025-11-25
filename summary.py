#   该部分代码用于查看网络结构
import torch
from torchinfo import summary

# 根据实际文件结构调整导入路径
from nets.yolov8 import YOLOv8Body  # 对应yolov8.py中的类名
from nets.CSPdarknet53_tiny import CSPDarkNet  # 导入tiny版本主干网络
from nets.CSPdarknet import CSPDarkNet as CSPDarkNet53  # 导入CSPDarknet53主干网络

if __name__ == "__main__":
    # 确定运行设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 查看YOLOv8网络结构（基于ultralytics的YOLOv8封装类）
    # 手势识别类别数为8（根据实际任务调整）
    num_classes = 8

    # 初始化YOLOv8模型（使用自定义类别数）
    yolov8_model = YOLOv8Body(
        model_type="yolov8n.pt",  # 模型类型
        num_classes=num_classes  # 指定手势类别数
    ).model.to(device)  # 获取内部模型实例

    # 打印YOLOv8网络结构（输入尺寸与训练保持一致，YOLOv8默认640x640）
    print("\nYOLOv8 Network Structure:")
    summary(yolov8_model, input_size=(1, 3, 640, 640), device=str(device))

    # 2. 查看CSPDarknet53主干网络结构（用于标准YOLOv4/YOLOv8）
    cspdarknet53 = CSPDarkNet53(layers=[1, 2, 8, 8, 4]).to(device)
    print("\nCSPDarknet53 Backbone Structure:")
    summary(cspdarknet53, input_size=(1, 3, 640, 640), device=str(device))

    # 3. 查看Tiny版本主干网络结构（CSPDarknet53-tiny）
    tiny_backbone = CSPDarkNet().to(device)
    print("\nCSPDarknet53-Tiny Backbone Structure:")
    summary(tiny_backbone, input_size=(1, 3, 416, 416), device=str(device))