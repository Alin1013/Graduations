import os
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
import yaml
import argparse
# 替换 YOLOv4 导入为 YOLOv8
from ultralytics import YOLO

if __name__ == "__main__":
    '''
    计算目标检测模型的 mAP，支持 YOLOv8 模型
    map_mode 0: 完整流程（预测结果+真实框+计算mAP）
    map_mode 1: 仅生成预测结果
    map_mode 2: 仅生成真实框
    map_mode 3: 仅计算 VOC 标准 mAP
    map_mode 4: 用 COCO 工具箱计算 mAP
    '''
    parser = argparse.ArgumentParser()
    # 替换为 YOLOv8 官方权重路径（支持本地路径或官方URL）
    parser.add_argument('--weights', type=str,
                        default='yolov8n.pt',  # 默认使用YOLOv8纳米模型
                        help='YOLOv8 weights path, e.g. yolov8n.pt, yolov8s.pt or custom path')
    parser.add_argument('--mode', type=int, default=0, help='get map的模式')
    parser.add_argument('--device', type=str, default='cpu', help='设备，cpu或0(GPU)')
    parser.add_argument('--shape', type=int, default=640, help='输入图像的shape（YOLOv8默认640）')
    parser.add_argument('--confidence', type=float, default=0.5, help='预测框置信度阈值')
    parser.add_argument('--nms_iou', type=float, default=0.3, help='非极大抑制IoU阈值')
    # 添加自定义权重参数
    parser.add_argument('--custom_weights', type=str, default=None,
                        help='自定义权重路径，优先级高于--weights')
    # 添加数据集配置文件路径（替代原get_config）
    parser.add_argument('--data', type=str, default='model_data/gesture.yaml', help='数据集配置文件路径')

    # 先解析参数
    opt = parser.parse_args()

    # 处理自定义权重（必须在解析参数之后）
    if opt.custom_weights and os.path.exists(opt.custom_weights):
        opt.weights = opt.custom_weights
        print(f"使用自定义权重: {opt.weights}")
    elif opt.custom_weights and not os.path.exists(opt.custom_weights):
        print(f"警告: 自定义权重路径不存在 {opt.custom_weights}，将使用默认权重 {opt.weights}")

    print(opt)

    # 从配置文件加载类别信息（YOLOv8标准yaml格式）
    with open(opt.data, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    class_names = data['names']  # 类别名称列表
    nc = data['nc']  # 类别数量

    # --------------------------------------------------------------------------#
    #   配置参数
    # --------------------------------------------------------------------------#
    map_mode = opt.mode
    MINOVERLAP = 0.5  # mAP计算的IoU阈值（如mAP@0.5）
    map_vis = False  # 是否可视化
    VOCdevkit_path = 'VOCdevkit'  # VOC格式数据集路径
    map_out_path = 'map_out'  # 结果输出路径

    # 获取验证集图像ID
    image_ids = open(os.path.join(VOCdevkit_path, "VOC2026/ImageSets/Main/val.txt")).read().strip().split()

    # 创建输出目录
    os.makedirs(map_out_path, exist_ok=True)
    os.makedirs(os.path.join(map_out_path, 'ground-truth'), exist_ok=True)
    os.makedirs(os.path.join(map_out_path, 'detection-results'), exist_ok=True)
    os.makedirs(os.path.join(map_out_path, 'images-optional'), exist_ok=True)

    # --------------------------------------------------------------------------#
    #   1. 生成预测结果（detection-results）
    # --------------------------------------------------------------------------#
    if map_mode == 0 or map_mode == 1:
        print("加载YOLOv8模型...")
        # 加载YOLOv8模型（支持本地权重或自动下载官方权重）
        model = YOLO(opt.weights)
        # 设置模型参数
        model.conf = 0.001  # 低置信度阈值，确保更多候选框用于mAP计算
        model.iou = 0.5  # NMS IoU阈值
        model.to(opt.device)  # 设置设备
        print("模型加载完成.")

        print("生成预测结果...")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, f"VOC2026/JPEGImages/{image_id}.jpg")
            image = Image.open(image_path)

            # 保存可视化图像（如果需要）
            if map_vis:
                image.save(os.path.join(map_out_path, f"images-optional/{image_id}.jpg"))

            # 模型预测
            results = model.predict(
                image,
                imgsz=opt.shape,
                device=opt.device,
                verbose=False  # 关闭详细输出
            )

            # 写入预测结果到txt（适配VOC格式）
            with open(os.path.join(map_out_path, f"detection-results/{image_id}.txt"), "w") as f:
                for result in results:
                    boxes = result.boxes  # 检测框
                    for box in boxes:
                        cls_id = int(box.cls[0])  # 类别ID
                        cls_name = class_names[cls_id]  # 类别名称
                        conf = float(box.conf[0])  # 置信度
                        # 坐标转换：中心xywh -> 左上角xy右下角xy（原图尺寸）
                        x1, y1, x2, y2 = box.xyxy[0].tolist()  # xyxy格式（左上角x,y，右下角x,y）
                        # 写入格式：类别 置信度 x1 y1 x2 y2
                        f.write(f"{cls_name} {conf:.6f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")
        print("预测结果生成完成.")

    # --------------------------------------------------------------------------#
    #   2. 生成真实框（ground-truth）
    # --------------------------------------------------------------------------#
    if map_mode == 0 or map_mode == 2:
        print("生成真实框标注...")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, f"ground-truth/{image_id}.txt"), "w") as f:
                # 解析VOC格式xml标注
                xml_path = os.path.join(VOCdevkit_path, f"VOC2026/Annotations/{image_id}.xml")
                root = ET.parse(xml_path).getroot()

                for obj in root.findall('object'):
                    # 处理difficult标签
                    difficult = obj.find('difficult').text if obj.find('difficult') else '0'
                    difficult_flag = (int(difficult) == 1)

                    # 类别名称
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue  # 跳过不在类别列表中的标注

                    # 边界框坐标
                    bndbox = obj.find('bndbox')
                    x1 = bndbox.find('xmin').text
                    y1 = bndbox.find('ymin').text
                    x2 = bndbox.find('xmax').text
                    y2 = bndbox.find('ymax').text

                    # 写入格式：类别 x1 y1 x2 y2 [difficult]
                    if difficult_flag:
                        f.write(f"{obj_name} {x1} {y1} {x2} {y2} difficult\n")
                    else:
                        f.write(f"{obj_name} {x1} {y1} {x2} {y2}\n")
        print("真实框标注生成完成.")

    # --------------------------------------------------------------------------#
    #   3. 计算mAP（保持原逻辑，使用VOC标准）
    # --------------------------------------------------------------------------#
    if map_mode == 0 or map_mode == 3:
        print("计算VOC mAP...")
        # 此处复用原get_map函数（确保utils.utils_map中的get_map兼容当前格式）
        from utils.utils_map import get_map

        get_map(MINOVERLAP, True, path=map_out_path)
        print("VOC mAP计算完成.")

    # --------------------------------------------------------------------------#
    #   4. 用COCO工具箱计算mAP
    # --------------------------------------------------------------------------#
    if map_mode == 4:
        print("计算COCO mAP...")
        from utils.utils_map import get_coco_map

        get_coco_map(class_names=class_names, path=map_out_path)
        print("COCO mAP计算完成.")