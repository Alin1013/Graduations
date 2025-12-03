import os
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
import yaml
import argparse
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')  # 忽略无关警告

# -------------------------- 内置简化版 get_map 函数（无需依赖外部 utils）--------------------------
# 避免依赖外部 utils.utils_map，直接内置核心 mAP 计算逻辑（VOC 标准）
def get_map(min_overlap=0.5, visualize=False, path="map_out"):
    """
    计算 VOC 标准 mAP（mean Average Precision）
    min_overlap: mAP@min_overlap（默认 0.5，即 mAP@0.5）
    visualize: 是否生成 PR 曲线
    path: 结果输出目录（包含 ground-truth 和 detection-results）
    """
    import numpy as np
    from collections import defaultdict

    # 1. 读取类别和标注文件
    gt_dir = os.path.join(path, "ground-truth")
    det_dir = os.path.join(path, "detection-results")
    image_ids = [f.split('.')[0] for f in os.listdir(gt_dir) if f.endswith('.txt')]
    if not image_ids:
        print("警告：未找到真实框标注文件！")
        return 0.0

    # 2. 统计所有类别（从真实框中提取）
    classes = set()
    gt_boxes = defaultdict(list)  # key: 类别, value: [(image_id, x1, y1, x2, y2, difficult)]
    for image_id in image_ids:
        with open(os.path.join(gt_dir, f"{image_id}.txt"), 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_name = parts[0]
            classes.add(cls_name)
            x1, y1, x2, y2 = map(float, parts[1:5])
            difficult = len(parts) > 5 and parts[5] == 'difficult'
            gt_boxes[cls_name].append((image_id, x1, y1, x2, y2, difficult))
    classes = sorted(list(classes))
    if not classes:
        print("警告：未检测到任何类别标注！")
        return 0.0

    # 3. 读取预测结果
    det_boxes = defaultdict(list)  # key: 类别, value: [(image_id, conf, x1, y1, x2, y2)]
    for image_id in image_ids:
        det_path = os.path.join(det_dir, f"{image_id}.txt")
        if not os.path.exists(det_path):
            continue
        with open(det_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue  # 跳过格式错误的行
            cls_name = parts[0]
            conf = float(parts[1])
            x1, y1, x2, y2 = map(float, parts[2:6])
            det_boxes[cls_name].append((image_id, conf, x1, y1, x2, y2))

    # 4. 计算每个类别的 AP
    aps = []
    print(f"\n{'='*50}")
    print(f"开始计算 mAP@{min_overlap}")
    print(f"类别数量：{len(classes)}")
    print(f"图像数量：{len(image_ids)}")
    print('='*50)

    for cls in classes:
        # 真实框
        gt = gt_boxes.get(cls, [])
        # 预测框（按置信度降序排序）
        det = sorted(det_boxes.get(cls, []), key=lambda x: x[1], reverse=True)

        if not gt:
            print(f"\n{cls}: 无真实框标注，AP=0.000")
            aps.append(0.0)
            continue
        if not det:
            print(f"\n{cls}: 无预测结果，AP=0.000")
            aps.append(0.0)
            continue

        # 初始化变量
        n_pos = sum(1 for g in gt if not g[5])  # 非difficult的真实框数量
        tp = np.zeros(len(det))  # True Positive
        fp = np.zeros(len(det))  # False Positive
        gt_detected = defaultdict(bool)  # 标记真实框是否已被匹配

        # 遍历预测框
        for i, (det_image_id, det_conf, det_x1, det_y1, det_x2, det_y2) in enumerate(det):
            # 查找当前图像中该类别的真实框
            matched_gt = None
            max_iou = 0
            for g_idx, (gt_image_id, gt_x1, gt_y1, gt_x2, gt_y2, gt_difficult) in enumerate(gt):
                if gt_image_id != det_image_id or gt_detected[g_idx] or gt_difficult:
                    continue
                # 计算 IoU
                inter_x1 = max(det_x1, gt_x1)
                inter_y1 = max(det_y1, gt_y1)
                inter_x2 = min(det_x2, gt_x2)
                inter_y2 = min(det_y2, gt_y2)
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
                gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
                iou = inter_area / (det_area + gt_area - inter_area + 1e-6)
                if iou > max_iou and iou >= min_overlap:
                    max_iou = iou
                    matched_gt = g_idx

            if matched_gt is not None:
                tp[i] = 1
                gt_detected[matched_gt] = True
            else:
                fp[i] = 1

        # 计算 Precision 和 Recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recall = tp_cumsum / (n_pos + 1e-6)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        # 计算 AP（VOC 11点插值法）
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.0

        aps.append(ap)
        print(f"\n{cls}: AP={ap:.3f}")

    # 计算 mAP
    mAP = np.mean(aps)
    print(f"\n{'='*50}")
    print(f"mAP@{min_overlap} = {mAP:.3f}")
    print('='*50)
    return mAP

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    '''
    计算 YOLOv8 模型的 mAP 评估指标
    map_mode 0: 完整流程（预测结果+真实框+计算mAP）
    map_mode 1: 仅生成预测结果
    map_mode 2: 仅生成真实框
    map_mode 3: 仅计算 VOC 标准 mAP
    '''
    parser = argparse.ArgumentParser(description="YOLOv8 模型 mAP 评估工具")
    # 模型权重配置
    parser.add_argument('--weights', type=str, default='yolov8n.pt',
                        help='YOLOv8 官方权重（如 yolov8n.pt/yolov8s.pt）或本地自定义权重路径')
    parser.add_argument('--custom_weights', type=str, default=None,
                        help='自定义权重路径（优先级高于 --weights）')
    # 运行配置
    parser.add_argument('--mode', type=int, default=0, choices=[0,1,2,3],
                        help='运行模式：0=完整流程，1=仅生成预测框，2=仅生成真实框，3=仅计算mAP')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='运行设备（CPU或GPU，GPU需安装CUDA）')
    parser.add_argument('--shape', type=int, default=640, help='模型输入图像尺寸（YOLOv8默认640）')
    parser.add_argument('--confidence', type=float, default=0.001,
                        help='预测置信度阈值（计算mAP时建议设低，保留更多候选框）')
    parser.add_argument('--nms_iou', type=float, default=0.5, help='非极大抑制IoU阈值')
    # 数据集配置
    parser.add_argument('--data', type=str, default='model_data/gesture.yaml',
                        help='YOLOv8 数据集配置文件路径')
    parser.add_argument('--voc_path', type=str, default='VOCdevkit',
                        help='VOC格式数据集根路径')
    parser.add_argument('--map_out', type=str, default='map_out',
                        help='mAP 结果输出目录（存储预测框、真实框）')
    parser.add_argument('--min_overlap', type=float, default=0.5,
                        help='mAP 计算的 IoU 阈值（默认 0.5，即 mAP@0.5）')
    parser.add_argument('--vis', action='store_true', help='是否保存可视化图像')

    opt = parser.parse_args()
    print("="*60)
    print("YOLOv8 mAP 评估工具 - 配置参数")
    print("="*60)
    for k, v in vars(opt).items():
        print(f"{k}: {v}")
    print("="*60)

    # -------------------------- 1. 权重处理 --------------------------
    # 优先使用自定义权重
    if opt.custom_weights:
        if os.path.exists(opt.custom_weights):
            opt.weights = opt.custom_weights
            print(f"\n✅ 使用自定义权重：{opt.weights}")
        else:
            print(f"\n⚠️  自定义权重路径不存在：{opt.custom_weights}，将使用默认权重：{opt.weights}")

    # -------------------------- 2. 加载数据集配置 --------------------------
    if not os.path.exists(opt.data):
        raise FileNotFoundError(f"数据集配置文件不存在：{opt.data}")
    with open(opt.data, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    class_names = data_cfg.get('names', [])  # 类别名称列表
    nc = data_cfg.get('nc', 0)  # 类别数量
    if not class_names or nc <= 0:
        raise ValueError("数据集配置文件中未正确配置 'names' 或 'nc' 字段")
    print(f"\n✅ 加载数据集配置：类别数={nc}，类别={class_names}")

    # -------------------------- 3. 数据集路径适配 --------------------------
    # 适配你的数据集结构（images/val 而非 JPEGImages）
    voc_devkit_path = opt.voc_path
    val_image_dir = os.path.join(voc_devkit_path, "VOC2026/images/val")  # 验证集图像目录
    val_label_dir = os.path.join(voc_devkit_path, "VOC2026/labels/val")  # 验证集XML目录
    val_list_path = os.path.join(voc_devkit_path, "VOC2026/ImageSets/Main/val.txt")  # 验证集图像ID列表

    # 检查验证集列表文件
    if not os.path.exists(val_list_path):
        # 自动生成 val.txt（从 images/val 目录提取图像ID）
        print(f"\n⚠️  未找到 val.txt，自动从 {val_image_dir} 生成...")
        os.makedirs(os.path.dirname(val_list_path), exist_ok=True)
        image_ids = [f.split('.')[0] for f in os.listdir(val_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        with open(val_list_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(image_ids))
        print(f"✅ 生成 val.txt，包含 {len(image_ids)} 个图像ID")
    else:
        image_ids = open(val_list_path, 'r', encoding='utf-8').read().strip().split()
        print(f"\n✅ 加载验证集：{len(image_ids)} 个图像")

    # 检查图像和标注目录
    if not os.path.exists(val_image_dir):
        raise FileNotFoundError(f"验证集图像目录不存在：{val_image_dir}")
    if not os.path.exists(val_label_dir) and opt.mode in [0,2]:
        raise FileNotFoundError(f"验证集XML标注目录不存在：{val_label_dir}")

    # -------------------------- 4. 创建输出目录 --------------------------
    map_out_path = opt.map_out
    os.makedirs(map_out_path, exist_ok=True)
    os.makedirs(os.path.join(map_out_path, 'ground-truth'), exist_ok=True)
    os.makedirs(os.path.join(map_out_path, 'detection-results'), exist_ok=True)
    if opt.vis:
        os.makedirs(os.path.join(map_out_path, 'images-optional'), exist_ok=True)
    print(f"\n✅ 输出目录准备完成：{map_out_path}")

    # -------------------------- 5. 生成预测结果（detection-results）--------------------------
    if opt.mode in [0, 1]:
        print("\n" + "="*50)
        print("开始生成预测结果...")
        print("="*50)
        # 加载 YOLOv8 模型
        model = YOLO(opt.weights)
        model.conf = opt.confidence  # 置信度阈值
        model.iou = opt.nms_iou     # NMS IoU阈值
        model.to(opt.device)        # 切换设备

        # 批量预测
        for image_id in tqdm(image_ids, desc="生成预测框"):
            # 图像路径（适配多种后缀）
            image_path = None
            for suffix in ['.jpg', '.jpeg', '.png']:
                temp_path = os.path.join(val_image_dir, f"{image_id}{suffix}")
                if os.path.exists(temp_path):
                    image_path = temp_path
                    break
            if not image_path:
                print(f"\n⚠️  未找到图像：{image_id}，跳过")
                continue

            # 保存可视化图像
            if opt.vis:
                Image.open(image_path).save(os.path.join(map_out_path, f"images-optional/{image_id}.jpg"))

            # 模型预测
            results = model.predict(
                image_path,
                imgsz=opt.shape,
                device=opt.device,
                verbose=False,
                show_labels=False,
                show_conf=False
            )

            # 写入预测结果（VOC mAP 格式：类别 置信度 x1 y1 x2 y2）
            det_lines = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id >= len(class_names):
                        continue  # 跳过未知类别
                    cls_name = class_names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # 左上角xy + 右下角xy（原图尺寸）
                    det_lines.append(f"{cls_name} {conf:.6f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}")

            with open(os.path.join(map_out_path, f"detection-results/{image_id}.txt"), 'w', encoding='utf-8') as f:
                f.write('\n'.join(det_lines))
        print("✅ 预测结果生成完成！")

    # -------------------------- 6. 生成真实框（ground-truth）- 从 YOLO 标签读取 --------------------------
    if opt.mode in [0, 2]:
        print("\n" + "=" * 50)
        print("开始生成真实框标注（从 YOLO 标签读取）...")
        print("=" * 50)

        # YOLO 标签目录（验证集）
        yolo_label_dir = os.path.join(voc_devkit_path, "VOC2026/labels/val")
        if not os.path.exists(yolo_label_dir):
            raise FileNotFoundError(f"YOLO 验证集标签目录不存在：{yolo_label_dir}")

        for image_id in tqdm(image_ids, desc="生成真实框"):
            # 找到对应的 YOLO 标签文件
            yolo_txt_path = os.path.join(yolo_label_dir, f"{image_id}.txt")
            if not os.path.exists(yolo_txt_path):
                print(f"\n⚠️  未找到 YOLO 标签：{image_id}.txt，跳过")
                continue

            # 读取图像尺寸（用于将 YOLO 归一化坐标转成像素坐标）
            img_path = None
            for suffix in ['.jpg', '.jpeg', '.png']:
                temp_path = os.path.join(val_image_dir, f"{image_id}{suffix}")
                if os.path.exists(temp_path):
                    img_path = temp_path
                    break
            if not img_path:
                print(f"\n⚠️  未找到图像：{image_id}，跳过")
                continue
            with Image.open(img_path) as img:
                img_w, img_h = img.size

            # 解析 YOLO 标签（格式：cls_id x_center y_center width height）
            gt_lines = []
            with open(yolo_txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue  # 跳过格式错误的行

                # 解析 YOLO 格式数据
                try:
                    cls_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                except (ValueError, IndexError):
                    continue

                # 校验类别ID
                if cls_id < 0 or cls_id >= len(class_names):
                    continue
                cls_name = class_names[cls_id]

                # 转换为 VOC 像素坐标（xmin, ymin, xmax, ymax）
                x_center_pix = x_center * img_w
                y_center_pix = y_center * img_h
                width_pix = width * img_w
                height_pix = height * img_h

                x1 = x_center_pix - width_pix / 2
                y1 = y_center_pix - height_pix / 2
                x2 = x_center_pix + width_pix / 2
                y2 = y_center_pix + height_pix / 2

                # 确保坐标在图像范围内
                x1 = max(0.0, x1)
                y1 = max(0.0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)

                # 写入真实框（无 difficult 标签）
                gt_lines.append(f"{cls_name} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}")

            # 保存真实框文件
            with open(os.path.join(map_out_path, f"ground-truth/{image_id}.txt"), 'w', encoding='utf-8') as f:
                f.write('\n'.join(gt_lines))
        print("✅ 真实框标注生成完成！")

    # -------------------------- 7. 计算 mAP --------------------------
    if opt.mode in [0, 3]:
        print("\n" + "="*50)
        print("开始计算 mAP...")
        print("="*50)
        mAP = get_map(
            min_overlap=opt.min_overlap,
            visualize=opt.vis,
            path=map_out_path
        )
        # 保存 mAP 结果到文件
        with open(os.path.join(map_out_path, 'mAP_result.txt'), 'w', encoding='utf-8') as f:
            f.write(f"mAP@{opt.min_overlap} = {mAP:.3f}\n")
            f.write(f"类别：{class_names}\n")
            f.write(f"验证集图像数：{len(image_ids)}\n")
            f.write(f"模型权重：{opt.weights}\n")
            f.write(f"输入尺寸：{opt.shape}\n")
        print(f"\n✅ mAP 结果已保存到：{os.path.join(map_out_path, 'mAP_result.txt')}")

    print("\n" + "="*60)
    print("YOLOv8 mAP 评估完成！")
    print("="*60)