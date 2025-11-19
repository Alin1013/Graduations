# 简化后的train.py
import argparse
from ultralytics import YOLO
from get_yaml import get_config
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', '-bs', type=int, default=8, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--weights', '-w', type=str, default='yolov8n.pt', help='初始权重路径')
    parser.add_argument('--data', type=str, default='model_data/gesture.yaml', help='数据集配置文件')
    parser.add_argument('--device', type=str, default='cpu', help='设备，0为GPU，cpu为CPU（Mac建议用cpu）')
    parser.add_argument('--freeze', action='store_true', help='冻结训练')
    parser.add_argument('--freeze-epochs', type=int, default=50, help='冻结训练轮数')
    parser.add_argument('--lr0', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--mode', type=str, default='train', help='运行模式：train/val')
    opt = parser.parse_args()

    # 加载模型
    model = YOLO(opt.weights)

    # 冻结训练配置
    freeze = opt.freeze
    if freeze:
        # 冻结主干网络
        model.freeze()
        # 先进行冻结训练（删除 lr_decay_type 参数）
        model.train(
            data=opt.data,
            epochs=opt.freeze_epochs,
            imgsz=opt.imgsz,
            batch=opt.batch_size,
            device=opt.device,
            lr0=opt.lr0,
            workers=4,  # 对应yaml中的workers配置
            name='freeze_train'
        )
        # 解冻所有层
        model.unfreeze()
        # 继续训练
        remaining_epochs = opt.epochs - opt.freeze_epochs
        if remaining_epochs > 0:
            model.train(
                data=opt.data,
                epochs=remaining_epochs,
                imgsz=opt.imgsz,
                batch=opt.batch_size,
                device=opt.device,
                lr0=opt.lr0/10,  # 学习率减半
                workers=4,
                name='unfreeze_train',
                resume=True
            )
    else:
        # 正常训练（删除 lr_decay_type 参数）
        model.train(
            data=opt.data,
            epochs=opt.epochs,
            imgsz=opt.imgsz,
            batch=opt.batch_size,
            device=opt.device,
            lr0=opt.lr0,
            workers=4,
            name='normal_train'
        )