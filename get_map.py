import os
import numpy as np
import yaml
import argparse
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import warnings

warnings.filterwarnings('ignore')  # 忽略无关警告


# -------------------------- 内置简化版 get_map 函数（无需依赖外部 utils）--------------------------
def get_map(min_overlap=0.5, visualize=False, path="map_out"):
    """
    计算 VOC 标准 mAP（mean Average Precision）
    min_overlap: mAP@min_overlap（默认 0.5，即 mAP@0.5）
    visualize: 是否生成 PR 曲线
    path: 结果输出目录（包含 ground-truth 和 detection-results）
    """
    from collections import defaultdict

    # 1. 检查输出目录
    gt_dir = os.path.join(path, "ground-truth")
    det_dir = os.path.join(path, "detection-results")
    if not os.path.exists(gt_dir) or not os.listdir(gt_dir):
        print("❌ 错误：未找到真实框标注文件！")
        return 0.0
    if not os.path.exists(det_dir) and visualize:
        print("⚠️  未找到预测结果文件，跳过PR曲线生成")
        visualize = False

    # 2. 读取图像ID和类别
    image_ids = [f.split('.')[0] for f in os.listdir(gt_dir) if f.endswith('.txt')]
    classes = set()
    gt_boxes = defaultdict(list)  # key: 类别, value: [(image_id, x1, y1, x2, y2, difficult)]

    # 解析真实框
    for image_id in image_ids:
        gt_path = os.path.join(gt_dir, f"{image_id}.txt")
        try:
            with open(gt_path, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_name = parts[0]
                classes.add(cls_name)
                x1, y1, x2, y2 = map(float, parts[1:5])
                difficult = len(parts) > 5 and parts[5] == 'difficult'
                gt_boxes[cls_name].append((image_id, x1, y1, x2, y2, difficult))
        except Exception as e:
            print(f"⚠️  读取真实框 {image_id}.txt 失败：{e}，跳过")
            continue

    classes = sorted(list(classes))
    if not classes:
        print("❌ 错误：未检测到任何有效类别标注！")
        return 0.0

    # 3. 解析预测框
    det_boxes = defaultdict(list)  # key: 类别, value: [(image_id, conf, x1, y1, x2, y2)]
    for image_id in image_ids:
        det_path = os.path.join(det_dir, f"{image_id}.txt")
        if not os.path.exists(det_path):
            continue
        try:
            with open(det_path, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            for line in lines:
                parts = line.split()
                if len(parts) < 6:
                    continue
                cls_name = parts[0]
                conf = float(parts[1])
                x1, y1, x2, y2 = map(float, parts[2:6])
                det_boxes[cls_name].append((image_id, conf, x1, y1, x2, y2))
        except Exception as e:
            print(f"⚠️  读取预测框 {image_id}.txt 失败：{e}，跳过")
            continue

    # 4. 计算每个类别的 AP
    aps = []
    print(f"\n{'=' * 50}")
    print(f"开始计算 mAP@{min_overlap}")
    print(f"📊 统计信息：类别数={len(classes)}，图像数={len(image_ids)}")
    print('=' * 50)

    for cls in classes:
        # 准备真实框和预测框
        gt = gt_boxes.get(cls, [])
        det = sorted(det_boxes.get(cls, []), key=lambda x: x[1], reverse=True)

        if not gt:
            print(f"📌 {cls}: 无真实框标注 → AP=0.000")
            aps.append(0.0)
            continue
        if not det:
            print(f"📌 {cls}: 无预测结果 → AP=0.000")
            aps.append(0.0)
            continue

        # 计算TP/FP
        n_pos = sum(1 for g in gt if not g[5])  # 非difficult真实框数
        tp = np.zeros(len(det))
        fp = np.zeros(len(det))
        gt_detected = {i: False for i in range(len(gt))}  # 标记真实框是否已匹配

        for i, (det_img_id, det_conf, dx1, dy1, dx2, dy2) in enumerate(det):
            max_iou = 0.0
            matched_idx = -1

            # 匹配当前图像的真实框
            for g_idx, (gt_img_id, gx1, gy1, gx2, gy2, g_diff) in enumerate(gt):
                if gt_img_id != det_img_id or gt_detected[g_idx] or g_diff:
                    continue

                # 计算IoU
                inter_x1 = max(dx1, gx1)
                inter_y1 = max(dy1, gy1)
                inter_x2 = min(dx2, gx2)
                inter_y2 = min(dy2, gy2)
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                det_area = (dx2 - dx1) * (dy2 - dy1)
                gt_area = (gx2 - gx1) * (gy2 - gy1)
                iou = inter_area / (det_area + gt_area - inter_area + 1e-8)

                if iou > max_iou and iou >= min_overlap:
                    max_iou = iou
                    matched_idx = g_idx

            if matched_idx >= 0:
                tp[i] = 1
                gt_detected[matched_idx] = True
            else:
                fp[i] = 1

        # 计算Precision/Recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recall = tp_cumsum / (n_pos + 1e-8)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

        # VOC 11点插值法计算AP
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            mask = recall >= t
            if np.any(mask):
                ap += np.max(precision[mask]) / 11.0

        aps.append(ap)
        print(f"📌 {cls}: AP={ap:.3f}")

    # 计算mAP并输出
    mAP = np.mean(aps)
    print(f"\n{'=' * 50}")
    print(f"🎯 mAP@{min_overlap} = {mAP:.3f}")
    print('=' * 50)

    # 可选：生成PR曲线
    if visualize:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 8))
            for cls in classes:
                gt = gt_boxes.get(cls, [])
                det = sorted(det_boxes.get(cls, []), key=lambda x: x[1], reverse=True)
                if not gt or not det:
                    continue
                n_pos = sum(1 for g in gt if not g[5])
                tp = np.zeros(len(det))
                fp = np.zeros(len(det))
                gt_detected = {i: False for i in range(len(gt))}

                # 重新计算PR
                for i, (det_img_id, det_conf, dx1, dy1, dx2, dy2) in enumerate(det):
                    max_iou = 0.0
                    matched_idx = -1
                    for g_idx, (gt_img_id, gx1, gy1, gx2, gy2, g_diff) in enumerate(gt):
                        if gt_img_id != det_img_id or gt_detected[g_idx] or g_diff:
                            continue
                        inter_x1 = max(dx1, gx1)
                        inter_y1 = max(dy1, gy1)
                        inter_x2 = min(dx2, gx2)
                        inter_y2 = min(dy2, gy2)
                        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                        det_area = (dx2 - dx1) * (dy2 - dy1)
                        gt_area = (gx2 - gx1) * (gy2 - gy1)
                        iou = inter_area / (det_area + gt_area - inter_area + 1e-8)
                        if iou > max_iou and iou >= min_overlap:
                            max_iou = iou
                            matched_idx = g_idx
                    if matched_idx >= 0:
                        tp[i] = 1
                        gt_detected[matched_idx] = True
                    else:
                        fp[i] = 1
                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(fp)
                recall = tp_cumsum / (n_pos + 1e-8)
                precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
                plt.plot(recall, precision,
                         label=f'{cls} (AP={np.mean([ap for c, ap in zip(classes, aps) if c == cls]):.3f})')

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'PR Curves (mAP@{min_overlap} = {mAP:.3f})')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(path, 'pr_curves.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ PR曲线已保存至：{os.path.join(path, 'pr_curves.png')}")
        except ImportError:
            print("⚠️  缺少matplotlib库，跳过PR曲线生成")
        except Exception as e:
            print(f"⚠️  生成PR曲线失败：{e}")

    return mAP


# -------------------------- 辅助函数 --------------------------
def get_image_path(image_id, image_dir):
    """获取图像完整路径（适配多种后缀）"""
    for suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
        img_path = os.path.join(image_dir, f"{image_id}{suffix}")
        if os.path.exists(img_path):
            return img_path
    return None


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

    # 模型配置
    parser.add_argument('--weights', type=str, default='yolov8n.pt',
                        help='YOLOv8 官方权重（如 yolov8n.pt/yolov8s.pt）或本地自定义权重路径')
    parser.add_argument('--custom_weights', type=str, default=None,
                        help='自定义权重路径（优先级高于 --weights）')

    # 运行配置
    parser.add_argument('--mode', type=int, default=0, choices=[0, 1, 2, 3],
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
    parser.add_argument('--vis', action='store_true', help='是否生成PR曲线和可视化图像')

    opt = parser.parse_args()

    # 打印配置信息
    print("=" * 60)
    print("YOLOv8 mAP 评估工具 - 配置参数")
    print("=" * 60)
    for k, v in vars(opt).items():
        print(f"🔧 {k}: {v}")
    print("=" * 60)

    # -------------------------- 1. 权重处理 --------------------------
    if opt.custom_weights:
        if os.path.exists(opt.custom_weights):
            opt.weights = opt.custom_weights
            print(f"\n✅ 使用自定义权重：{opt.weights}")
        else:
            print(f"\n⚠️  自定义权重路径不存在：{opt.custom_weights}，将使用默认权重：{opt.weights}")

    # 检查权重文件是否存在（本地权重）
    if not opt.weights.startswith('yolov8') and not os.path.exists(opt.weights):
        raise FileNotFoundError(f"❌ 权重文件不存在：{opt.weights}")

    # -------------------------- 2. 加载数据集配置 --------------------------
    if not os.path.exists(opt.data):
        raise FileNotFoundError(f"❌ 数据集配置文件不存在：{opt.data}")

    with open(opt.data, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)

    class_names = data_cfg.get('names', [])
    nc = data_cfg.get('nc', len(class_names))

    if not class_names or nc <= 0:
        raise ValueError("❌ 数据集配置文件中未正确配置 'names' 或 'nc' 字段")

    print(f"\n✅ 加载数据集配置：")
    print(f"   ├── 类别数：{nc}")
    print(f"   └── 类别列表：{class_names}")

    # -------------------------- 3. 数据集路径配置 --------------------------
    voc_devkit_path = opt.voc_path
    val_image_dir = os.path.join(voc_devkit_path, "VOC2026/images/val")
    val_label_dir = os.path.join(voc_devkit_path, "VOC2026/labels/val")
    val_list_path = os.path.join(voc_devkit_path, "VOC2026/ImageSets/Main/val.txt")

    # 自动生成验证集列表
    if not os.path.exists(val_list_path):
        print(f"\n⚠️  未找到 val.txt，自动从 {val_image_dir} 生成...")
        os.makedirs(os.path.dirname(val_list_path), exist_ok=True)

        # 提取图像ID
        image_ids = []
        if os.path.exists(val_image_dir):
            image_ids = [f.split('.')[0] for f in os.listdir(val_image_dir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        if not image_ids:
            raise FileNotFoundError(f"❌ 验证集图像目录为空或不存在：{val_image_dir}")

        with open(val_list_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(image_ids))
        print(f"✅ 生成 val.txt，包含 {len(image_ids)} 个图像ID")
    else:
        image_ids = [line.strip() for line in open(val_list_path, 'r', encoding='utf-8') if line.strip()]
        if not image_ids:
            raise ValueError(f"❌ val.txt 为空，请检查：{val_list_path}")
        print(f"\n✅ 加载验证集：{len(image_ids)} 个图像")

    # 检查关键目录
    if not os.path.exists(val_image_dir):
        raise FileNotFoundError(f"❌ 验证集图像目录不存在：{val_image_dir}")
    if opt.mode in [0, 2] and not os.path.exists(val_label_dir):
        raise FileNotFoundError(f"❌ 验证集标签目录不存在：{val_label_dir}")

    # -------------------------- 4. 创建输出目录 --------------------------
    map_out_path = opt.map_out
    os.makedirs(map_out_path, exist_ok=True)
    os.makedirs(os.path.join(map_out_path, 'ground-truth'), exist_ok=True)
    os.makedirs(os.path.join(map_out_path, 'detection-results'), exist_ok=True)
    if opt.vis:
        os.makedirs(os.path.join(map_out_path, 'images-optional'), exist_ok=True)
    print(f"\n✅ 输出目录准备完成：{map_out_path}")

    # -------------------------- 5. 生成预测结果 --------------------------
    if opt.mode in [0, 1]:
        print("\n" + "=" * 50)
        print("开始生成预测结果...")
        print("=" * 50)

        # 加载模型
        model = YOLO(opt.weights)
        model.conf = opt.confidence
        model.iou = opt.nms_iou
        model.to(opt.device)

        # 批量预测
        success_count = 0
        for image_id in tqdm(image_ids, desc="生成预测框"):
            img_path = get_image_path(image_id, val_image_dir)
            if not img_path:
                print(f"\n⚠️  未找到图像：{image_id}，跳过")
                continue

            # 保存可视化图像
            if opt.vis:
                try:
                    Image.open(img_path).save(os.path.join(map_out_path, f"images-optional/{image_id}.jpg"))
                except Exception as e:
                    print(f"\n⚠️  保存可视化图像 {image_id} 失败：{e}")

            # 模型预测
            try:
                results = model.predict(
                    img_path,
                    imgsz=opt.shape,
                    device=opt.device,
                    verbose=False,
                    show_labels=False,
                    show_conf=False,
                    save=False,
                    augment=False
                )

                # 解析预测结果
                det_lines = []
                for r in results:
                    if r.boxes is None:
                        continue
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        if cls_id >= len(class_names):
                            continue
                        cls_name = class_names[cls_id]
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        det_lines.append(f"{cls_name} {conf:.6f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}")

                # 保存预测结果
                with open(os.path.join(map_out_path, f"detection-results/{image_id}.txt"), 'w', encoding='utf-8') as f:
                    f.write('\n'.join(det_lines))
                success_count += 1

            except Exception as e:
                print(f"\n⚠️  预测 {image_id} 失败：{e}，跳过")
                continue

        print(f"✅ 预测结果生成完成！成功处理 {success_count}/{len(image_ids)} 张图像")

    # -------------------------- 6. 生成真实框 --------------------------
    if opt.mode in [0, 2]:
        print("\n" + "=" * 50)
        print("开始生成真实框标注（从 YOLO 标签读取）...")
        print("=" * 50)

        success_count = 0
        for image_id in tqdm(image_ids, desc="生成真实框"):
            # 读取YOLO标签
            yolo_txt_path = os.path.join(val_label_dir, f"{image_id}.txt")
            if not os.path.exists(yolo_txt_path):
                print(f"\n⚠️  未找到 YOLO 标签：{image_id}.txt，跳过")
                continue

            # 读取图像尺寸
            img_path = get_image_path(image_id, val_image_dir)
            if not img_path:
                print(f"\n⚠️  未找到图像：{image_id}，跳过")
                continue

            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except Exception as e:
                print(f"\n⚠️  读取图像尺寸 {image_id} 失败：{e}，跳过")
                continue

            # 解析YOLO标签
            gt_lines = []
            try:
                with open(yolo_txt_path, 'r', encoding='utf-8') as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]

                for line in lines:
                    parts = line.split()
                    if len(parts) != 5:
                        continue

                    # 解析YOLO格式
                    cls_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # 校验
                    if cls_id < 0 or cls_id >= len(class_names):
                        continue
                    if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1:
                        continue
                    if width <= 0 or width > 1 or height <= 0 or height > 1:
                        continue

                    # 转换为像素坐标
                    x1 = (x_center - width / 2) * img_w
                    y1 = (y_center - height / 2) * img_h
                    x2 = (x_center + width / 2) * img_w
                    y2 = (y_center + height / 2) * img_h

                    # 边界校验
                    x1 = max(0.0, x1)
                    y1 = max(0.0, y1)
                    x2 = min(img_w, x2)
                    y2 = min(img_h, y2)

                    cls_name = class_names[cls_id]
                    gt_lines.append(f"{cls_name} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}")

                # 保存真实框
                with open(os.path.join(map_out_path, f"ground-truth/{image_id}.txt"), 'w', encoding='utf-8') as f:
                    f.write('\n'.join(gt_lines))
                success_count += 1

            except Exception as e:
                print(f"\n⚠️  解析标签 {image_id}.txt 失败：{e}，跳过")
                continue

        print(f"✅ 真实框生成完成！成功处理 {success_count}/{len(image_ids)} 个标签")

    # -------------------------- 7. 计算 mAP --------------------------
    if opt.mode in [0, 3]:
        print("\n" + "=" * 50)
        print("开始计算 mAP...")
        print("=" * 50)

        mAP = get_map(
            min_overlap=opt.min_overlap,
            visualize=opt.vis,
            path=map_out_path
        )

        # 保存结果
        result_path = os.path.join(map_out_path, 'mAP_result.txt')
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(f"YOLOv8 mAP 评估结果\n")
            f.write(f"{'=' * 30}\n")
            f.write(f"评估时间：{os.popen('date').read().strip()}\n")
            f.write(f"模型权重：{opt.weights}\n")
            f.write(f"输入尺寸：{opt.shape}\n")
            f.write(f"置信度阈值：{opt.confidence}\n")
            f.write(f"NMS IoU阈值：{opt.nms_iou}\n")
            f.write(f"mAP IoU阈值：{opt.min_overlap}\n")
            f.write(f"验证集图像数：{len(image_ids)}\n")
            f.write(f"类别列表：{class_names}\n")
            f.write(f"{'=' * 30}\n")
            f.write(f"mAP @ {opt.min_overlap} = {mAP:.3f}\n")

        print(f"\n✅ mAP 结果已保存到：{result_path}")

    print("\n" + "=" * 60)
    print("🎉 YOLOv8 mAP 评估流程完成！")
    print("=" * 60)