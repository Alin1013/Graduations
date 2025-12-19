import glob
import json
import math
import operator
import os
import shutil
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''


def log_average_miss_rate(precision, fp_cumsum, num_images):
    """
        log-average miss rate:
            计算9个均匀分布在10e-2到10e0之间的FPPI点的平均漏检率
    """
    if precision.size == 0:
        return 0.0, 1.0, 0.0

    fppi = fp_cumsum / float(num_images)
    mr = (1 - precision)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        # 找到最后一个小于等于ref_i的索引
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # 计算对数平均，避免log(0)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))
    return lamr, mr, fppi


def error(msg):
    """抛出错误并退出"""
    print(f"❌ {msg}")
    sys.exit(1)


def is_float_between_0_and_1(value):
    """检查是否为0到1之间的浮点数"""
    try:
        val = float(value)
        return 0.0 < val < 1.0
    except ValueError:
        return False


def voc_ap(rec, prec):
    """
    计算VOC标准的平均精度(AP)
    1. 计算单调递减的精度曲线
    2. 数值积分计算曲线下面积
    """
    # 插入首尾点
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec.copy()

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec.copy()

    # 使精度单调递减
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 找到召回率变化的点
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    # 计算AP（积分面积）
    ap = 0.0
    for i in i_list:
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap, mrec, mpre


def file_lines_to_list(path):
    """读取文件内容到列表（去除空白字符）"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        error(f"读取文件失败: {path}，错误: {str(e)}")


def draw_text_in_image(img, text, pos, color, line_width):
    """在图像上绘制文本"""
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    lineType = 1
    cv2.putText(img, text, pos, font, fontScale, color, lineType)
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    return img, (line_width + line_width)


def adjust_axes(r, t, fig, axes):
    """调整坐标轴以适应文本"""
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label,
                   output_path, to_show, plot_color, true_p_bar=""):
    """绘制各类统计图表"""
    # 按值排序
    sorted_items = sorted(dictionary.items(), key=operator.itemgetter(1))
    sorted_keys, sorted_values = zip(*sorted_items) if sorted_items else ([], [])

    if true_p_bar:
        # 绘制TP/FP对比图
        fp_sorted = [dictionary[key] - true_p_bar[key] for key in sorted_keys]
        tp_sorted = [true_p_bar[key] for key in sorted_keys]

        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='假阳性(FP)')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen',
                 label='真阳性(TP)', left=fp_sorted)
        plt.legend(loc='lower right')

        # 添加数值标签
        fig = plt.gcf()
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            t = plt.text(val, i, f" {fp_val} {tp_val}", color='forestgreen',
                         va='center', fontweight='bold')
            plt.text(val, i, f" {fp_val}", color='crimson', va='center', fontweight='bold')
            if i == len(sorted_values) - 1:
                adjust_axes(r, t, fig, axes)
    else:
        # 绘制普通柱状图
        plt.barh(range(n_classes), sorted_values, color=plot_color)

        # 添加数值标签
        fig = plt.gcf()
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = f" {val:.2f}" if val < 1.0 else f" {val}"
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            if i == len(sorted_values) - 1:
                adjust_axes(r, t, fig, axes)

    # 设置图表属性
    fig.canvas.manager.set_window_title(window_title)
    plt.yticks(range(n_classes), sorted_keys, fontsize=12)

    # 调整高度
    init_height = fig.get_figheight()
    dpi = fig.dpi
    height_pt = n_classes * (12 * 1.4)
    height_in = height_pt / dpi
    top_margin = 0.15
    bottom_margin = 0.05
    figure_height = height_in / (1 - top_margin - bottom_margin)
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    plt.title(plot_title, fontsize=14)
    plt.xlabel(x_label, fontsize='large')
    fig.tight_layout()

    # 保存和显示
    try:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    except Exception as e:
        print(f"⚠️  保存图表失败: {str(e)}")
    if to_show:
        plt.show()
    plt.close()


def parse_class_name(line_split, difficult_flag=False):
    """解析类别名称（处理包含空格的类别名）"""
    if difficult_flag:
        class_name_parts = line_split[:-5]
    else:
        class_name_parts = line_split[:-4]
    return ' '.join(class_name_parts).strip()


def get_map(MINOVERLAP=0.5, draw_plot=True, path='./map_out'):
    """计算VOC标准的mAP"""
    # 路径定义
    GT_PATH = os.path.join(path, 'ground-truth')
    DR_PATH = os.path.join(path, 'detection-results')
    IMG_PATH = os.path.join(path, 'images-optional')
    TEMP_FILES_PATH = os.path.join(path, '.temp_files')
    RESULTS_FILES_PATH = os.path.join(path, 'results')

    # 检查图像路径是否存在
    show_animation = False
    if os.path.exists(IMG_PATH):
        for _, _, files in os.walk(IMG_PATH):
            if files:
                show_animation = True
                break

    # 创建临时目录
    try:
        os.makedirs(TEMP_FILES_PATH, exist_ok=True)
    except Exception as e:
        error(f"创建临时目录失败: {str(e)}")

    # 清理结果目录
    if os.path.exists(RESULTS_FILES_PATH):
        try:
            shutil.rmtree(RESULTS_FILES_PATH)
        except Exception as e:
            error(f"清理结果目录失败: {str(e)}")

    # 创建结果子目录
    if draw_plot:
        for subdir in ['AP', 'F1', 'Recall', 'Precision']:
            os.makedirs(os.path.join(RESULTS_FILES_PATH, subdir), exist_ok=True)
    if show_animation:
        os.makedirs(os.path.join(RESULTS_FILES_PATH, 'images', 'detections_one_by_one'), exist_ok=True)

    # 读取真实框文件
    ground_truth_files = glob.glob(os.path.join(GT_PATH, '*.txt'))
    if not ground_truth_files:
        error("未找到真实框标注文件!")
    ground_truth_files.sort()

    gt_counter_per_class = {}
    counter_images_per_class = {}

    # 处理真实框
    for txt_file in ground_truth_files:
        file_id = os.path.splitext(os.path.basename(txt_file))[0]
        dr_file = os.path.join(DR_PATH, f"{file_id}.txt")

        if not os.path.exists(dr_file):
            error(f"未找到对应的检测结果文件: {dr_file}")

        lines = file_lines_to_list(txt_file)
        bounding_boxes = []
        already_seen_classes = []

        for line in lines:
            is_difficult = "difficult" in line
            line_split = line.split()

            try:
                if is_difficult:
                    class_name = parse_class_name(line_split, difficult_flag=True)
                    left, top, right, bottom = line_split[-5:-1]
                else:
                    class_name = parse_class_name(line_split)
                    left, top, right, bottom = line_split[-4:]
            except:
                error(f"解析标注文件错误: {txt_file}，行: {line}")

            # 保存框信息
            bbox = f"{left} {top} {right} {bottom}"
            if is_difficult:
                bounding_boxes.append({
                    "class_name": class_name,
                    "bbox": bbox,
                    "used": False,
                    "difficult": True
                })
            else:
                bounding_boxes.append({
                    "class_name": class_name,
                    "bbox": bbox,
                    "used": False
                })

                # 更新类别计数
                gt_counter_per_class[class_name] = gt_counter_per_class.get(class_name, 0) + 1
                if class_name not in already_seen_classes:
                    counter_images_per_class[class_name] = counter_images_per_class.get(class_name, 0) + 1
                    already_seen_classes.append(class_name)

        # 保存临时JSON
        temp_gt_path = os.path.join(TEMP_FILES_PATH, f"{file_id}_ground_truth.json")
        with open(temp_gt_path, 'w', encoding='utf-8') as f:
            json.dump(bounding_boxes, f, ensure_ascii=False)

    # 检查是否有类别
    if not gt_counter_per_class:
        error("未检测到任何有效类别标注!")
    gt_classes = sorted(gt_counter_per_class.keys())
    n_classes = len(gt_classes)

    # 处理检测结果
    dr_files = glob.glob(os.path.join(DR_PATH, '*.txt'))
    dr_files.sort()

    for class_name in gt_classes:
        bounding_boxes = []
        for txt_file in dr_files:
            file_id = os.path.splitext(os.path.basename(txt_file))[0]
            gt_file = os.path.join(GT_PATH, f"{file_id}.txt")

            if not os.path.exists(gt_file):
                error(f"未找到对应的真实框文件: {gt_file}")

            lines = file_lines_to_list(txt_file)
            for line in lines:
                line_split = line.split()
                try:
                    tmp_class_name = parse_class_name(line_split[:-5])
                    confidence = line_split[-5]
                    left, top, right, bottom = line_split[-4:]
                except:
                    error(f"解析检测文件错误: {txt_file}，行: {line}")

                if tmp_class_name == class_name:
                    bounding_boxes.append({
                        "confidence": confidence,
                        "file_id": file_id,
                        "bbox": f"{left} {top} {right} {bottom}"
                    })

        # 按置信度排序并保存
        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
        temp_dr_path = os.path.join(TEMP_FILES_PATH, f"{class_name}_dr.json")
        with open(temp_dr_path, 'w', encoding='utf-8') as f:
            json.dump(bounding_boxes, f, ensure_ascii=False)

    # 计算每个类别的AP
    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}

    # 保存结果文件
    results_file_path = os.path.join(RESULTS_FILES_PATH, 'results.txt')
    with open(results_file_path, 'w', encoding='utf-8') as results_file:
        results_file.write("# 每个类别的AP、精确率和召回率\n")
        count_true_positives = {cls: 0 for cls in gt_classes}

        for class_name in gt_classes:
            dr_file = os.path.join(TEMP_FILES_PATH, f"{class_name}_dr.json")
            try:
                with open(dr_file, 'r', encoding='utf-8') as f:
                    dr_data = json.load(f)
            except Exception as e:
                error(f"读取检测结果JSON失败: {str(e)}")

            nd = len(dr_data)
            tp = [0] * nd
            fp = [0] * nd
            score = [0.0] * nd
            score05_idx = 0

            for idx, detection in enumerate(dr_data):
                file_id = detection["file_id"]
                score[idx] = float(detection["confidence"])
                if score[idx] > 0.5:
                    score05_idx = idx

                # 处理可视化
                img = None
                img_cumulative = None
                if show_animation:
                    img_files = glob.glob(os.path.join(IMG_PATH, f"{file_id}.*"))
                    if not img_files:
                        error(f"未找到图像文件: {file_id}")
                    if len(img_files) > 1:
                        error(f"存在多个同名图像: {file_id}")

                    img_path = img_files[0]
                    img = cv2.imread(img_path)
                    if img is None:
                        error(f"无法读取图像: {img_path}")

                    img_cumulative_path = os.path.join(RESULTS_FILES_PATH, 'images', os.path.basename(img_path))
                    img_cumulative = cv2.imread(img_cumulative_path) if os.path.exists(
                        img_cumulative_path) else img.copy()

                    # 添加底部边框用于显示文本
                    bottom_border = 60
                    img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0,
                                             cv2.BORDER_CONSTANT, value=[0, 0, 0])

                # 计算IoU
                gt_file = os.path.join(TEMP_FILES_PATH, f"{file_id}_ground_truth.json")
                try:
                    with open(gt_file, 'r', encoding='utf-8') as f:
                        ground_truth_data = json.load(f)
                except Exception as e:
                    error(f"读取真实框JSON失败: {str(e)}")

                ovmax = -1.0
                gt_match = None
                bb = [float(x) for x in detection["bbox"].split()]

                for obj in ground_truth_data:
                    if obj["class_name"] == class_name and not obj.get("difficult", False):
                        bbgt = [float(x) for x in obj["bbox"].split()]
                        # 计算交集
                        bi = [
                            max(bb[0], bbgt[0]),
                            max(bb[1], bbgt[1]),
                            min(bb[2], bbgt[2]),
                            min(bb[3], bbgt[3])
                        ]
                        iw = max(0, bi[2] - bi[0] + 1e-8)
                        ih = max(0, bi[3] - bi[1] + 1e-8)

                        if iw > 0 and ih > 0:
                            # 计算并集
                            ua = (bb[2] - bb[0] + 1e-8) * (bb[3] - bb[1] + 1e-8) + \
                                 (bbgt[2] - bbgt[0] + 1e-8) * (bbgt[3] - bbgt[1] + 1e-8) - \
                                 iw * ih
                            ov = (iw * ih) / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                # 判断TP/FP
                status = "未找到匹配"
                if ovmax >= MINOVERLAP and gt_match is not None:
                    if not gt_match["used"]:
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1
                        # 更新真实框文件
                        with open(gt_file, 'w', encoding='utf-8') as f:
                            json.dump(ground_truth_data, f, ensure_ascii=False)
                        status = "匹配成功"
                    else:
                        fp[idx] = 1
                        status = "重复匹配"
                else:
                    fp[idx] = 1
                    if ovmax > 0:
                        status = "重叠不足"

                # 绘制动画
                if show_animation and img is not None:
                    height, width = img.shape[:2]
                    white = (255, 255, 255)
                    light_blue = (255, 200, 100)
                    green = (0, 255, 0)
                    light_red = (30, 30, 255)
                    margin = 10

                    # 第一行文本
                    v_pos = int(height - margin - (bottom_border / 2.0))
                    text = f"图像: {os.path.basename(img_path)} "
                    img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)

                    class_idx = gt_classes.index(class_name)
                    text = f"类别 [{class_idx + 1}/{n_classes}]: {class_name} "
                    img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue,
                                                         line_width)

                    # IoU信息
                    if ovmax != -1:
                        color = light_red if status == "重叠不足" else green
                        iou_text = f"IoU: {ovmax * 100:.2f}% {'<' if status == '重叠不足' else '>'}={MINOVERLAP * 100:.2f}% "
                        img, _ = draw_text_in_image(img, iou_text, (margin + line_width, v_pos), color, line_width)

                    # 第二行文本
                    v_pos += int(bottom_border / 2.0)
                    text = f"检测 #秩: {idx + 1} 置信度: {float(detection['confidence']) * 100:.2f}% "
                    img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)

                    color = light_red if status != "匹配成功" else green
                    text = f"结果: {status} "
                    img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)

                    # 绘制边框
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    if ovmax > 0 and gt_match:
                        bbgt = [int(round(float(x))) for x in gt_match["bbox"].split()]
                        cv2.rectangle(img, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 2)
                        cv2.rectangle(img_cumulative, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 2)
                        cv2.putText(img_cumulative, class_name, (bbgt[0], bbgt[1] - 5),
                                    font, 0.6, light_blue, 1, cv2.LINE_AA)

                    bb_int = [int(round(x)) for x in bb]
                    cv2.rectangle(img, (bb_int[0], bb_int[1]), (bb_int[2], bb_int[3]), color, 2)
                    cv2.rectangle(img_cumulative, (bb_int[0], bb_int[1]), (bb_int[2], bb_int[3]), color, 2)
                    cv2.putText(img_cumulative, class_name, (bb_int[0], bb_int[1] - 5),
                                font, 0.6, color, 1, cv2.LINE_AA)

                    # 显示和保存
                    cv2.imshow("动画演示", img)
                    cv2.waitKey(20)
                    output_img_path = os.path.join(RESULTS_FILES_PATH, 'images', 'detections_one_by_one',
                                                   f"{class_name}_detection{idx}.jpg")
                    cv2.imwrite(output_img_path, img)
                    cv2.imwrite(img_cumulative_path, img_cumulative)

            # 计算累计TP/FP
            cumsum = 0
            for idx in range(len(fp)):
                fp[idx] += cumsum
                cumsum += fp[idx]

            cumsum = 0
            for idx in range(len(tp)):
                tp[idx] += cumsum
                cumsum += tp[idx]

            # 计算召回率和精确率
            total_gt = max(gt_counter_per_class[class_name], 1)
            rec = [tp[i] / total_gt for i in range(len(tp))]
            prec = [tp[i] / max(tp[i] + fp[i], 1) for i in range(len(tp))]

            # 计算AP
            ap, mrec, mprec = voc_ap(rec.copy(), prec.copy())
            sum_AP += ap

            # 计算F1分数
            f1 = []
            for p, r in zip(prec, rec):
                if p + r == 0:
                    f1.append(0.0)
                else:
                    f1.append(2 * p * r / (p + r))

            # 记录结果
            ap_str = f"{ap * 100:.2f}% = {class_name} AP"
            results_file.write(f"{ap_str}\n")
            results_file.write(f"  精确率: {[round(p, 2) for p in prec]}\n")
            results_file.write(f"  召回率: {[round(r, 2) for r in rec]}\n\n")

            # 打印结果
            if len(prec) > 0 and score05_idx < len(f1):
                print(f"{ap_str}\t||\t置信度阈值=0.5 : F1={f1[score05_idx]:.2f} ; "
                      f"召回率={rec[score05_idx] * 100:.2f}% ; 精确率={prec[score05_idx] * 100:.2f}%")
            else:
                print(f"{ap_str}\t||\t置信度阈值=0.5 : F1=0.00 ; 召回率=0.00% ; 精确率=0.00%")

            ap_dictionary[class_name] = ap

            # 计算log-average miss rate
            n_images = counter_images_per_class.get(class_name, 1)
            lamr, _, _ = log_average_miss_rate(np.array(rec), np.array(fp), n_images)
            lamr_dictionary[class_name] = lamr

            # 绘制PR等曲线
            if draw_plot and len(prec) > 0:
                # PR曲线
                plt.plot(rec, prec, '-o')
                area_x = mrec[:-1] + [mrec[-2], mrec[-1]]
                area_y = mprec[:-1] + [0.0, mprec[-1]]
                plt.fill_between(area_x, 0, area_y, alpha=0.2, edgecolor='r')
                plt.title(f'类别: {ap_str}')
                plt.xlabel('召回率')
                plt.ylabel('精确率')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.savefig(os.path.join(RESULTS_FILES_PATH, 'AP', f"{class_name}.png"))
                plt.clf()

                # F1曲线
                plt.plot(score, f1, "-", color='orangered')
                plt.title(f'类别: {class_name} F1分数\n置信度阈值=0.5')
                plt.xlabel('置信度阈值')
                plt.ylabel('F1分数')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.savefig(os.path.join(RESULTS_FILES_PATH, 'F1', f"{class_name}.png"))
                plt.clf()

                # 召回率曲线
                plt.plot(score, rec, "-H", color='gold')
                plt.title(f'类别: {class_name} 召回率\n置信度阈值=0.5')
                plt.xlabel('置信度阈值')
                plt.ylabel('召回率')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.savefig(os.path.join(RESULTS_FILES_PATH, 'Recall', f"{class_name}.png"))
                plt.clf()

                # 精确率曲线
                plt.plot(score, prec, "-s", color='palevioletred')
                plt.title(f'类别: {class_name} 精确率\n置信度阈值=0.5')
                plt.xlabel('置信度阈值')
                plt.ylabel('精确率')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.savefig(os.path.join(RESULTS_FILES_PATH, 'Precision', f"{class_name}.png"))
                plt.clf()

        # 关闭动画窗口
        if show_animation:
            cv2.destroyAllWindows()

        # 计算并保存mAP
        mAP = sum_AP / n_classes
        mAP_str = f"mAP = {mAP * 100:.2f}%"
        results_file.write(f"\n# 所有类别的mAP\n{mAP_str}\n")
        print(mAP_str)

    # 清理临时文件
    try:
        shutil.rmtree(TEMP_FILES_PATH)
    except Exception as e:
        print(f"⚠️  清理临时文件失败: {str(e)}")

    # 统计检测结果数量
    det_counter_per_class = {}
    for txt_file in dr_files:
        lines = file_lines_to_list(txt_file)
        for line in lines:
            class_name = parse_class_name(line.split()[:-5])
            det_counter_per_class[class_name] = det_counter_per_class.get(class_name, 0) + 1

    # 补充未检测到的类别
    for class_name in gt_classes:
        if class_name not in det_counter_per_class:
            det_counter_per_class[class_name] = 0

    # 写入各类统计信息
    with open(results_file_path, 'a', encoding='utf-8') as results_file:
        results_file.write("\n# 每个类别的真实框数量\n")
        for class_name in sorted(gt_counter_per_class):
            results_file.write(f"{class_name}: {gt_counter_per_class[class_name]}\n")

        results_file.write("\n# 每个类别的检测结果数量\n")
        for class_name in sorted(det_counter_per_class):
            n_det = det_counter_per_class[class_name]
            tp = count_true_positives.get(class_name, 0)
            fp = n_det - tp
            results_file.write(f"{class_name}: {n_det} (tp:{tp}, fp:{fp})\n")

    # 绘制各类统计图表
    if draw_plot:
        # 真实框类别分布
        draw_plot_func(
            gt_counter_per_class,
            n_classes,
            "真实框信息",
            f"真实框分布\n({len(ground_truth_files)}个文件，{n_classes}个类别)",
            "每个类别的目标数量",
            os.path.join(RESULTS_FILES_PATH, "ground-truth-info.png"),
            False,
            'forestgreen'
        )

        # 绘制lamr图表
        draw_plot_func(
            lamr_dictionary,
            n_classes,
            "log-average漏检率",
            "log-average漏检率",
            "log-average漏检率",
            os.path.join(RESULTS_FILES_PATH, "lamr.png"),
            False,
            'royalblue'
        )

        # 绘制mAP图表
        draw_plot_func(
            ap_dictionary,
            n_classes,
            "mAP",
            f"mAP = {mAP * 100:.2f}%",
            "平均精度(AP)",
            os.path.join(RESULTS_FILES_PATH, "mAP.png"),
            True,
            'royalblue'
        )

    return mAP


def preprocess_gt(gt_path, class_names):
    """预处理真实框为COCO格式"""
    image_ids = os.listdir(gt_path)
    results = {"images": [], "categories": [], "annotations": []}

    # 处理图像信息
    for image_id in image_ids:
        if not image_id.endswith('.txt'):
            continue
        img_name = os.path.splitext(image_id)[0]
        image = {
            "file_name": f"{img_name}.jpg",
            "width": 1,  # 实际使用时应从图像读取
            "height": 1,
            "id": img_name
        }
        results["images"].append(image)

        # 处理框信息
        lines = file_lines_to_list(os.path.join(gt_path, image_id))
        for line in lines:
            is_difficult = "difficult" in line
            line_split = line.split()

            try:
                if is_difficult:
                    class_name = parse_class_name(line_split, difficult_flag=True)
                    left, top, right, bottom = map(float, line_split[-5:-1])
                    difficult = 1
                else:
                    class_name = parse_class_name(line_split)
                    left, top, right, bottom = map(float, line_split[-4:])
                    difficult = 0
            except:
                print(f"⚠️  解析标注文件错误: {image_id}，行: {line}，已跳过")
                continue

            if class_name not in class_names:
                continue

            cls_id = class_names.index(class_name) + 1
            w = right - left
            h = bottom - top
            area = w * h - 10.0  # 调整值，避免面积为0

            results["annotations"].append({
                "area": area,
                "category_id": cls_id,
                "image_id": img_name,
                "iscrowd": difficult,
                "bbox": [left, top, w, h],
                "id": len(results["annotations"])  # 自增ID
            })

    # 处理类别信息
    for i, cls in enumerate(class_names):
        results["categories"].append({
            "supercategory": cls,
            "name": cls,
            "id": i + 1
        })

    return results


def preprocess_dr(dr_path, class_names):
    """预处理检测结果为COCO格式"""
    results = []
    dr_files = os.listdir(dr_path)

    for dr_file in dr_files:
        if not dr_file.endswith('.txt'):
            continue
        image_id = os.path.splitext(dr_file)[0]
        lines = file_lines_to_list(os.path.join(dr_path, dr_file))

        for line in lines:
            line_split = line.split()
            try:
                class_name = parse_class_name(line_split[:-5])
                confidence = float(line_split[-5])
                left, top, right, bottom = map(float, line_split[-4:])
            except:
                print(f"⚠️  解析检测文件错误: {dr_file}，行: {line}，已跳过")
                continue

            if class_name not in class_names:
                continue

            cls_id = class_names.index(class_name) + 1
            w = right - left
            h = bottom - top

            results.append({
                "image_id": image_id,
                "category_id": cls_id,
                "bbox": [left, top, w, h],
                "score": confidence
            })

    return results


def get_coco_map(class_names, path='./map_out'):
    """计算COCO标准的mAP"""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        error("请安装pycocotools: pip install pycocotools")

    GT_PATH = os.path.join(path, 'ground-truth')
    DR_PATH = os.path.join(path, 'detection-results')
    COCO_PATH = os.path.join(path, 'coco_eval')
    os.makedirs(COCO_PATH, exist_ok=True)

    # 生成COCO格式的JSON
    GT_JSON_PATH = os.path.join(COCO_PATH, 'instances_gt.json')
    DR_JSON_PATH = os.path.join(COCO_PATH, 'instances_dr.json')

    try:
        with open(GT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(preprocess_gt(GT_PATH, class_names), f, ensure_ascii=False, indent=4)

        with open(DR_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(preprocess_dr(DR_PATH, class_names), f, ensure_ascii=False, indent=4)
    except Exception as e:
        error(f"生成COCO格式JSON失败: {str(e)}")

    # 评估
    try:
        cocoGt = COCO(GT_JSON_PATH)
        cocoDt = cocoGt.loadRes(DR_JSON_PATH)
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        return cocoEval.stats[0]  # 返回mAP@[0.5:0.95]
    except Exception as e:
        error(f"COCO评估失败: {str(e)}")