#   该部分代码用于查看网络结构
import torch
from torchinfo import summary
from utils.utils import get_anchors

# 根据实际文件结构调整导入路径
from nets.yolov8 import YOLOv8  # 修正：使用正确的类名YOLOv8
from nets.CSPdarknet53_tiny import CSPDarkNet  # 导入tiny版本主干网络
from nets.CSPdarknet import CSPDarkNet as CSPDarkNet53  # 导入CSPDarknet53主干网络

if __name__ == "__main__":
    # 确定运行设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 查看YOLOv8网络结构
    # 手势识别类别数为18
    num_classes = 18
    
    # 加载锚点配置
    anchors_path = 'yolo_anchors.txt'
    try:
        anchors, num_anchors = get_anchors(anchors_path)
        anchors = anchors.tolist()  # 转换为列表格式
        print(f"✅ 加载锚点成功: {num_anchors}个锚点")
    except Exception as e:
        print(f"⚠️  加载锚点失败: {e}")
        print("使用默认锚点配置")
        # 默认锚点（如果文件不存在）
        anchors = [[21, 24], [33, 27], [30, 43], [44, 34], [37, 63], 
                   [50, 48], [67, 56], [53, 75], [83, 82]]

    # 初始化YOLOv8模型（使用正确的参数）
    yolov8_model = YOLOv8(
        num_classes=num_classes,
        anchors=anchors,
        input_shape=[640, 640],
        cuda=(device.type == 'cuda')
    ).to(device)

    # 打印YOLOv8网络结构（输入尺寸与训练保持一致，YOLOv8默认640x640）
    print("\n" + "=" * 60)
    print("YOLOv8 Network Structure:")
    print("=" * 60)
    summary(yolov8_model, input_size=(1, 3, 640, 640), device=str(device))

    # 2. 查看CSPDarknet53主干网络结构（用于标准YOLOv4/YOLOv8）
    cspdarknet53 = CSPDarkNet53(layers=[1, 2, 8, 8, 4]).to(device)
    print("\nCSPDarknet53 Backbone Structure:")
    summary(cspdarknet53, input_size=(1, 3, 640, 640), device=str(device))

    # 3. 查看Tiny版本主干网络结构（CSPDarknet53-tiny）
    tiny_backbone = CSPDarkNet().to(device)
    print("\nCSPDarknet53-Tiny Backbone Structure:")
    summary(tiny_backbone, input_size=(1, 3, 416, 416), device=str(device))