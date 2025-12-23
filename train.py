from ultralytics import YOLO
import os
import torch
import argparse
from pathlib import Path

# -------------------------- 解析终端传入的参数 --------------------------
parser = argparse.ArgumentParser(description='YOLOv8 手势识别训练脚本')
# 核心训练参数（支持终端传参，同时设置默认值）
parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
parser.add_argument('--device', type=str, default=None, help='训练设备 (cpu/0/cuda)')
parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
parser.add_argument('--weights', type=str, default='yolov8n.pt', help='预训练权重路径')
args = parser.parse_args()

# -------------------------- 基础配置 --------------------------
# 项目根目录（动态计算，避免硬编码）
PROJECT_ROOT = Path(__file__).parent
native_yaml_path = PROJECT_ROOT / "native_data.yaml"

# 处理设备参数（优先终端传入，其次自动检测）
if args.device:
    device = args.device
else:
    device = '0' if torch.cuda.is_available() else 'cpu'

# -------------------------- 生成原生格式的 yaml 文件 --------------------------
try:
    # 动态获取数据集路径（与gesture.yaml保持一致）
    train_img_dir = PROJECT_ROOT / "VOCdevkit/VOC2026/images/train"
    val_img_dir = PROJECT_ROOT / "VOCdevkit/VOC2026/images/val"

    # 检查数据集目录是否存在
    if not train_img_dir.exists():
        raise FileNotFoundError(f"训练图像目录不存在：{train_img_dir}")
    if not val_img_dir.exists():
        raise FileNotFoundError(f"验证图像目录不存在：{val_img_dir}")

    with open(native_yaml_path, "w", encoding="utf-8") as f:
        f.write(f"""# YOLOv8 原生数据集格式（与gesture.yaml类别一致）
train: {train_img_dir}
val: {val_img_dir}
nc: 19
names: ["no_gesture","call","like","dislike","ok","fist","four","mute","one","palm","peace","peace_invered","rock","stop","stop_invered","three","three_two","two_up","two_up_invered"]
""")
    print(f"✅ 成功生成数据集配置文件：{native_yaml_path}")
except Exception as e:
    print(f"❌ 生成 YAML 文件失败：{e}")
    exit(1)

# -------------------------- 加载模型并训练 --------------------------
try:
    model = YOLO(args.weights)  # 加载指定的预训练模型
    print(f"🔧 使用设备：{device}（GPU 可用：{torch.cuda.is_available()}）")

    # 训练配置（保留原参数，新增余弦学习率调度）
    training_results = model.train(
        data=str(native_yaml_path),
        epochs=args.epochs,
        batch=args.batch_size,
        device=device,
        workers=min(os.cpu_count(), 4),  # 自适应CPU核心数
        imgsz=args.imgsz,
        pretrained=True,
        name='gesture_final_train',
        cache=False,
        verbose=True,
        # 数据增强（适合手势识别的参数）
        fliplr=0.5,          # 水平翻转
        hsv_h=0.015,         # 色调抖动
        hsv_s=0.7,           # 饱和度抖动
        hsv_v=0.4,           # 明度抖动
        translate=0.1,       # 平移变换
        erasing=0.4,         # 随机擦除
        # 优化器
        lr0=0.001,           # 初始学习率
        lrf=0.01,            # 最终学习率因子
        weight_decay=0.0005, # 权重衰减
        cos_lr=True,         # 新增：余弦学习率调度
        # 早停设置
        patience=10,         # 10轮无提升则停止
        val=True             # 启用验证
    )
except Exception as e:
    print(f"❌ 训练过程出错：{e}")
    exit(1)

# -------------------------- 清理与结果输出 --------------------------
# 删除临时YAML文件
try:
    if native_yaml_path.exists():
        native_yaml_path.unlink()
        print(f"\n🗑️  临时 yaml 文件已删除：{native_yaml_path}")
except PermissionError:
    print(f"\n⚠️  无权限删除临时文件：{native_yaml_path}，请手动删除")
except Exception as e:
    print(f"\n⚠️  删除临时文件失败：{e}")

# 打印训练结果
print("\n🎉 训练完成！")
print(f"📁 训练结果保存路径：{training_results.save_dir}")
best_pt_path = Path(training_results.save_dir) / "weights" / "best.pt"
print(f"💾 最佳模型路径：{best_pt_path}")
if hasattr(training_results, 'best_fitness'):
    print(f"📊 最佳模型 mAP50-95：{training_results.best_fitness:.4f}")