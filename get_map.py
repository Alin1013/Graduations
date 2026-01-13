import os
import numpy as np
import yaml
import argparse
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import warnings
from datetime import datetime

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
    class_ap_dict = {}  # 存储每个类别的AP值，用于PR曲线绘制
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
            class_ap_dict[cls] = 0.0
            continue
        if not det:
            print(f"📌 {cls}: 无预测结果 → AP=0.000")
            aps.append(0.0)
            class_ap_dict[cls] = 0.0
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
        class_ap_dict[cls] = ap
        print(f"📌 {cls}: AP={ap:.3f}")

    # 计算mAP并输出
    mAP = np.mean(aps) if aps else 0.0
    print(f"\n{'=' * 50}")
    print(f"🎯 mAP@{min_overlap} = {mAP:.3f} ({mAP*100:.1f}%)")
    print('=' * 50)
    
    # 输出统计信息
    print(f"\n📊 详细统计：")
    print(f"   ├── 总类别数：{len(classes)}")
    print(f"   ├── 有真实框的类别：{len([c for c in classes if gt_boxes.get(c)])}")
    print(f"   ├── 有预测框的类别：{len([c for c in classes if det_boxes.get(c)])}")
    print(f"   └── 总真实框数：{sum(len(v) for v in gt_boxes.values())}")
    print(f"   └── 总预测框数：{sum(len(v) for v in det_boxes.values())}")

    # 可选：生成PR曲线
    if visualize:
        try:
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            import matplotlib.pyplot as plt
            
            # 设置中文字体支持
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            plt.figure(figsize=(12, 8))
            colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))
            
            for idx, cls in enumerate(classes):
                gt = gt_boxes.get(cls, [])
                det = sorted(det_boxes.get(cls, []), key=lambda x: x[1], reverse=True)
                if not gt or not det:
                    continue
                
                n_pos = sum(1 for g in gt if not g[5])
                if n_pos == 0:
                    continue
                    
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
                
                # 确保recall和precision数组有效
                if len(recall) == 0 or len(precision) == 0:
                    continue
                
                # 使用预先计算的AP值
                ap_value = class_ap_dict.get(cls, 0.0)
                plt.plot(recall, precision, 
                        color=colors[idx], 
                        linewidth=2,
                        label=f'{cls} (AP={ap_value:.3f})',
                        alpha=0.8)

            plt.xlabel('Recall', fontsize=12, fontweight='bold')
            plt.ylabel('Precision', fontsize=12, fontweight='bold')
            plt.title(f'Precision-Recall Curves (mAP@{min_overlap} = {mAP:.3f})', 
                     fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=9, ncol=2)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.tight_layout()
            plt.savefig(os.path.join(path, 'pr_curves.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ PR曲线已保存至：{os.path.join(path, 'pr_curves.png')}")
        except ImportError as e:
            print(f"⚠️  缺少matplotlib库，跳过PR曲线生成：{e}")
        except Exception as e:
            print(f"⚠️  生成PR曲线失败：{e}")
            import traceback
            traceback.print_exc()

    return mAP, class_ap_dict


# -------------------------- 辅助函数 --------------------------
def get_image_path(image_id, image_dir):
    """获取图像完整路径（适配多种后缀，支持大小写不敏感）"""
    # 首先尝试精确匹配
    for suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIF']:
        img_path = os.path.join(image_dir, f"{image_id}{suffix}")
        if os.path.exists(img_path):
            return img_path
    
    # 如果精确匹配失败，尝试在目录中搜索（大小写不敏感）
    if os.path.exists(image_dir):
        image_id_lower = image_id.lower()
        for filename in os.listdir(image_dir):
            filename_base = os.path.splitext(filename)[0]
            if filename_base.lower() == image_id_lower:
                # 检查是否是图像文件
                ext = os.path.splitext(filename)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
                    return os.path.join(image_dir, filename)
    
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

    # 检查可视化依赖
    if opt.vis:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️  缺少matplotlib库，自动禁用可视化功能")
            opt.vis = False

    # 打印配置信息
    print("=" * 60)
    print("YOLOv8 mAP 评估工具 - 配置参数")
    print("=" * 60)
    for k, v in vars(opt).items():
        print(f"🔧 {k}: {v}")
    print("=" * 60)

    # -------------------------- 1. 加载数据集配置（先加载，因为可能包含权重路径） --------------------------
    if not os.path.exists(opt.data):
        raise FileNotFoundError(f"❌ 数据集配置文件不存在：{opt.data}")

    with open(opt.data, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)

    class_names = data_cfg.get('names', [])
    nc = data_cfg.get('nc', len(class_names))

    if not class_names or nc <= 0:
        raise ValueError("❌ 数据集配置文件中未正确配置 'names' 或 'nc' 字段")

    # 从配置文件读取模型权重（如果存在）
    config_weights = data_cfg.get('weights', None)

    print(f"\n✅ 加载数据集配置：")
    print(f"   ├── 类别数：{nc}")
    print(f"   ├── 类别列表：{class_names}")
    if config_weights:
        print(f"   └── 配置文件中的权重路径：{config_weights}")

    # -------------------------- 2. 权重处理（优先级：命令行custom_weights > 配置文件weights > 命令行weights > 默认值） --------------------------
    # 确定最终使用的权重路径
    final_weights = None
    
    # 优先级1：命令行指定的custom_weights
    if opt.custom_weights:
        if os.path.exists(opt.custom_weights):
            final_weights = opt.custom_weights
            print(f"\n✅ 使用命令行指定的自定义权重：{final_weights}")
        else:
            print(f"\n⚠️  命令行指定的权重路径不存在：{opt.custom_weights}")
    
    # 优先级2：配置文件中的weights（如果命令行没有指定或指定的不存在）
    if not final_weights and config_weights:
        if os.path.exists(config_weights):
            final_weights = config_weights
            print(f"✅ 使用配置文件中的权重：{final_weights}")
        else:
            print(f"⚠️  配置文件中指定的权重路径不存在：{config_weights}")
    
    # 优先级3：命令行指定的weights（默认参数）
    if not final_weights:
        final_weights = opt.weights
        if final_weights.startswith('yolov8'):
            print(f"✅ 使用默认预训练权重：{final_weights}")
        else:
            print(f"✅ 使用命令行指定的权重：{final_weights}")
    
    # 更新opt.weights为最终确定的权重
    opt.weights = final_weights

    # 检查权重文件是否存在（本地权重）
    if not opt.weights.startswith('yolov8') and not os.path.exists(opt.weights):
        raise FileNotFoundError(f"❌ 权重文件不存在：{opt.weights}")

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
        
        # 诊断：检查图像ID匹配情况
        if os.path.exists(val_image_dir):
            sample_image_ids = image_ids[:min(10, len(image_ids))]
            actual_files = [f for f in os.listdir(val_image_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            print(f"\n📋 诊断信息：")
            print(f"   val.txt 中的前5个图像ID：{sample_image_ids[:5]}")
            if actual_files:
                actual_ids = [os.path.splitext(f)[0] for f in actual_files[:5]]
                print(f"   图像目录中的前5个文件名：{actual_ids}")
                # 检查匹配情况
                matched = sum(1 for img_id in sample_image_ids 
                            if get_image_path(img_id, val_image_dir) is not None)
                match_rate = matched / len(sample_image_ids) if sample_image_ids else 0
                print(f"   匹配情况：{matched}/{len(sample_image_ids)} 个图像ID能找到对应文件 ({match_rate*100:.1f}%)")
                
                # 如果匹配率低于30%，提示重新生成val.txt
                if match_rate < 0.3 and len(actual_files) > 0:
                    print(f"\n⚠️  警告：图像ID匹配率过低 ({match_rate*100:.1f}%)！")
                    print(f"   建议：删除 val.txt 文件，让脚本自动从图像目录重新生成")
                    print(f"   执行命令：rm {val_list_path}")
                    print(f"   或者：脚本将自动使用实际存在的图像文件")
                    
                    # 自动修复：使用实际存在的图像文件
                    print(f"\n🔄 自动修复：从图像目录重新生成图像ID列表...")
                    new_image_ids = [os.path.splitext(f)[0] for f in actual_files
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    new_image_ids = sorted(list(set(new_image_ids)))  # 去重并排序
                    
                    if new_image_ids:
                        # 备份旧的val.txt
                        backup_path = val_list_path + '.backup'
                        import shutil
                        shutil.copy2(val_list_path, backup_path)
                        print(f"   ✅ 已备份原 val.txt 到 {backup_path}")
                        
                        # 写入新的val.txt
                        with open(val_list_path, 'w', encoding='utf-8') as f:
                            f.write('\n'.join(new_image_ids))
                        print(f"   ✅ 已重新生成 val.txt，包含 {len(new_image_ids)} 个图像ID")
                        image_ids = new_image_ids
                    else:
                        print(f"   ⚠️  无法自动修复：图像目录中没有找到有效的图像文件")
            else:
                print(f"   ⚠️  图像目录中没有找到图像文件")

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
        not_found_count = 0
        not_found_ids = []
        for image_id in tqdm(image_ids, desc="生成预测框"):
            img_path = get_image_path(image_id, val_image_dir)
            if not img_path:
                not_found_count += 1
                if not_found_count <= 5:  # 只显示前5个未找到的图像
                    print(f"\n⚠️  未找到图像：{image_id}，跳过")
                elif not_found_count == 6:
                    print(f"\n⚠️  ... (还有更多图像未找到，将在最后汇总)")
                not_found_ids.append(image_id)
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
                # 单张图像预测结果处理
                r = results[0]
                if r.boxes is not None:
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
        if not_found_count > 0:
            print(f"⚠️  警告：有 {not_found_count} 张图像未找到")
            if not_found_count <= 10:
                print(f"   未找到的图像ID：{', '.join(not_found_ids[:10])}")
            else:
                print(f"   未找到的图像ID（前10个）：{', '.join(not_found_ids[:10])}...")
            print(f"   提示：请检查 val.txt 中的图像ID是否与实际图像文件名匹配")

    # -------------------------- 6. 生成真实框 --------------------------
    if opt.mode in [0, 2]:
        print("\n" + "=" * 50)
        print("开始生成真实框标注（从 YOLO 标签读取）...")
        print("=" * 50)

        success_count = 0
        not_found_label_count = 0
        not_found_image_count = 0
        for image_id in tqdm(image_ids, desc="生成真实框"):
            # 读取YOLO标签
            yolo_txt_path = os.path.join(val_label_dir, f"{image_id}.txt")
            if not os.path.exists(yolo_txt_path):
                not_found_label_count += 1
                if not_found_label_count <= 5:
                    print(f"\n⚠️  未找到 YOLO 标签：{image_id}.txt，跳过")
                continue

            # 读取图像尺寸
            img_path = get_image_path(image_id, val_image_dir)
            if not img_path:
                not_found_image_count += 1
                if not_found_image_count <= 5:
                    print(f"\n⚠️  未找到图像：{image_id}，跳过")
                continue

            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')  # 确保图像格式正确
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
                        print(f"\n⚠️  类别ID {cls_id} 超出范围（0-{len(class_names)-1}），跳过")
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
        if not_found_label_count > 0:
            print(f"⚠️  警告：有 {not_found_label_count} 个标签文件未找到")
        if not_found_image_count > 0:
            print(f"⚠️  警告：有 {not_found_image_count} 张图像未找到")

    # -------------------------- 7. 计算 mAP --------------------------
    if opt.mode in [0, 3]:
        print("\n" + "=" * 50)
        print("开始计算 mAP...")
        print("=" * 50)

        mAP, class_ap_dict = get_map(
            min_overlap=opt.min_overlap,
            visualize=opt.vis,
            path=map_out_path
        )

        # 保存结果
        result_path = os.path.join(map_out_path, 'mAP_result.txt')
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(f"YOLOv8 mAP 评估结果\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"评估时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型权重：{opt.weights}\n")
            f.write(f"输入尺寸：{opt.shape}\n")
            f.write(f"置信度阈值：{opt.confidence}\n")
            f.write(f"NMS IoU阈值：{opt.nms_iou}\n")
            f.write(f"mAP IoU阈值：{opt.min_overlap}\n")
            f.write(f"验证集图像数：{len(image_ids)}\n")
            f.write(f"类别数：{len(class_names)}\n")
            f.write(f"{'=' * 50}\n\n")
            
            # 总体mAP
            f.write(f"🎯 mAP@{opt.min_overlap} = {mAP:.3f} ({mAP*100:.1f}%)\n\n")
            
            # 每个类别的AP（只显示有真实框的类别）
            f.write(f"{'=' * 50}\n")
            f.write(f"各类别AP详情：\n")
            f.write(f"{'=' * 50}\n")
            # 按class_names顺序显示，但只显示有真实框的类别
            for cls_name in class_names:
                if cls_name in class_ap_dict:
                    ap_value = class_ap_dict[cls_name]
                    f.write(f"  {cls_name:20s}: {ap_value:.3f} ({ap_value*100:.1f}%)\n")
                else:
                    # 标记没有真实框的类别
                    f.write(f"  {cls_name:20s}: 0.000 (0.0%) [无真实框标注]\n")
            f.write(f"{'=' * 50}\n")

        print(f"\n✅ mAP 结果已保存到：{result_path}")

    print("\n" + "=" * 60)
    print("🎉 YOLOv8 mAP 评估流程完成！")
    print("=" * 60)