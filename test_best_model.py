# 测试脚本

from ultralytics import YOLO
import cv2
import os  # 新增：导入os模块（解决路径打印报错）

# 数据集内的测试图像（已确认存在）
test_image = '/Users/alin/Graduation_Project/VOCdevkit/VOC2026/images/train/0a1eba8e-8671-47ad-9de3-48b0805c7ef5.jpg'

# 加载模型（根据训练进度选择）
# 选项1：训练未完成，用最新模型（last.pt）
#model = YOLO('runs/detect/gesture_final_train/weights/last.pt')
# 选项2：训练已完成，用最佳模型（best.pt）→ 注释上面，解开下面
model = YOLO('runs/detect/gesture_final_train/weights/best.pt')

# 降低置信度阈值（从0.5→0.3，避免漏检，适合训练初期模型）
results = model(test_image, conf=0.3)

# 显示检测结果（弹出窗口）
results[0].show()

# 保存检测结果到当前目录
results[0].save('detected_result.jpg')
print("✅ 检测完成！")
print(f"📁 原始图像路径：{test_image}")
print(f"💾 检测结果已保存为：{os.path.abspath('detected_result.jpg')}")

# 打印检测详情（方便排查是否有低置信度结果）
if len(results[0].boxes) > 0:
    print(f"\n📊 检测到 {len(results[0].boxes)} 个目标：")
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = model.names[cls]
        print(f"  - 类别：{cls_name}，置信度：{conf:.2f}")
else:
    print("\n⚠️  未检测到任何目标（可能是模型训练轮数不足，或图像本身无手势）")