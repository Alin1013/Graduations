# -----------------------------------------------------------------------#
#   train.py 用于训练YOLOv8模型，支持冻结训练和正则化配置
# -----------------------------------------------------------------------#
import argparse
import ssl
from ultralytics import YOLO

# 解决SSL证书问题（用于下载预训练权重）
ssl._create_default_https_context = ssl._create_unverified_context


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='总训练轮数')
    parser.add_argument('--batch-size', '-bs', type=int, default=8, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--weights', '-w', type=str, default='yolov8n.pt', help='初始权重路径')
    parser.add_argument('--data', type=str, default='model_data/gesture.yaml', help='数据集配置文件')
    parser.add_argument('--device', type=str, default='cpu', help='设备，0为GPU，cpu为CPU（Mac建议用cpu）')
    parser.add_argument('--freeze', action='store_true', help='启用冻结训练')
    parser.add_argument('--freeze-epochs', type=int, default=50, help='冻结训练轮数')
    parser.add_argument('--lr0', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--l1-lambda', type=float, default=1e-4, help='L1正则化系数')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val'], help='运行模式')
    return parser.parse_args()


def train_model(opt):
    """训练模型主函数"""
    # 加载模型
    model = YOLO(opt.weights)

    # 配置训练参数（包含L1正则化）
    base_train_kwargs = {
        'data': opt.data,
        'imgsz': opt.imgsz,
        'batch': opt.batch_size,
        'device': opt.device,
        'lr0': opt.lr0,
        'workers': 4,
        'weight_decay': opt.l1_lambda,
    }

    # 冻结训练逻辑（使用Ultralytics YOLOv8的正确冻结方式）
    if opt.freeze:
        # 冻结主干网络（YOLOv8的backbone通常有10层左右，根据实际结构调整）
        # 冻结前10层（可根据模型结构调整层数）
        model.train(**{
            **base_train_kwargs,
            'epochs': opt.freeze_epochs,
            'freeze': 10,  # 指定冻结的层数
            'name': 'freeze_train'
        })

        # 解冻并微调剩余轮数
        remaining_epochs = opt.epochs - opt.freeze_epochs
        if remaining_epochs > 0:
            model.train(**{
                **base_train_kwargs,
                'epochs': remaining_epochs,
                'lr0': opt.lr0 / 10,  # 降低学习率
                'name': 'unfreeze_train',
                'resume': True  # 从上次训练继续
            })
    else:
        # 正常训练（不冻结）
        model.train(**{
            **base_train_kwargs,
            'epochs': opt.epochs,
            'name': 'normal_train'
        })

    # 验证模式（如果指定）
    if opt.mode == 'val':
        metrics = model.val(data=opt.data, device=opt.device)
        print("验证指标:", metrics)


if __name__ == "__main__":
    args = parse_args()
    train_model(args)