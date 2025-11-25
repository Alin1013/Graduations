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
    train_kwargs = {
        'data': opt.data,
        'epochs': opt.epochs,
        'imgsz': opt.imgsz,
        'batch': opt.batch_size,
        'device': opt.device,
        'lr0': opt.lr0,
        'workers': 4,  # 对应数据集配置中的workers
        'weight_decay': opt.l1_lambda,  # 使用L1正则化（ultralytics通过weight_decay支持）
        'name': 'normal_train'
    }

    # 冻结训练逻辑
    if opt.freeze:
        # 冻结主干网络
        model.freeze()
        # 冻结训练阶段
        model.train(
            **{**train_kwargs,
               'epochs': opt.freeze_epochs,
               'name': 'freeze_train'
               }
        )
        # 解冻网络
        model.unfreeze()
        # 计算剩余训练轮数
        remaining_epochs = opt.epochs - opt.freeze_epochs
        if remaining_epochs > 0:
            # 微调阶段（学习率降低）
            model.train(
                **{**train_kwargs,
                   'epochs': remaining_epochs,
                   'lr0': opt.lr0 / 10,  # 学习率衰减
                   'name': 'unfreeze_train',
                   'resume': True  # 从冻结训练的最后一个 checkpoint 继续
                   }
            )
    else:
        # 正常训练
        model.train(**train_kwargs)

    # 验证模式（如果指定）
    if opt.mode == 'val':
        metrics = model.val(data=opt.data, device=opt.device)
        print("验证指标:", metrics)


if __name__ == "__main__":
    args = parse_args()
    train_model(args)