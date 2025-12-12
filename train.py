from ultralytics import YOLO
import os

# -------------------------- 生成原生格式的 yaml 文件 --------------------------
native_yaml_path = "/Users/alin/Graduation_Project/native_data.yaml"
with open(native_yaml_path, "w", encoding="utf-8") as f:
    f.write("""# YOLOv8 原生数据集格式（图像和标签目录对应）
train: /Users/alin/Graduation_Project/VOCdevkit/VOC2026/images/train  # 训练图像目录
val: /Users/alin/Graduation_Project/VOCdevkit/VOC2026/images/val      # 验证图像目录
nc: 19
names: ["no_gesture","call","like","dislike","ok","fist","four","mute","one","palm","peace","peace_invered","rock","stop","stop_invered","three","three_two","two_up","two_up_invered"]
# 标签目录默认与图像目录对应（images → labels），无需额外指定！
""")

# -------------------------- 加载模型并训练 --------------------------
model = YOLO('yolov8n.pt')

training_results = model.train(
    data=native_yaml_path,  # 原生格式 yaml
    epochs=50,
    batch=4,
    device='cpu',
    workers=0,
    imgsz=640,
    pretrained=True,
    name='gesture_final_train',  # 最终训练目录
    cache=False,
    verbose=True,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.1,
    erasing=0.4,
    lr0=0.001,
    weight_decay=0.0005
)

# 删除临时 yaml 文件
if os.path.exists(native_yaml_path):
    os.remove(native_yaml_path)
    print(f"\n🗑️  临时 yaml 文件已删除：{native_yaml_path}")

# 打印结果路径
print("\n🎉 训练完成！")
print(f"📁 训练结果保存路径：{training_results.save_dir}")
print(f"💾 最佳模型路径：{training_results.save_dir}/weights/best.pt")