import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def cas_iou(box, cluster):
    """计算单个框与聚类中心的IoU（针对锚框聚类的IoU计算）"""
    x = np.minimum(cluster[:, 0], box[0])
    y = np.minimum(cluster[:, 1], box[1])

    intersection = x * y
    area1 = box[0] * box[1]
    area2 = cluster[:, 0] * cluster[:, 1]
    iou = intersection / (area1 + area2 - intersection)

    return iou


def avg_iou(box, cluster):
    """计算所有框与对应聚类中心的平均IoU"""
    return np.mean([np.max(cas_iou(box[i], cluster)) for i in range(box.shape[0])])


def kmeans(box, k):
    """K-means聚类算法（锚框专用）"""
    # 取出框的总数
    row = box.shape[0]

    # 存储每个框到各聚类中心的距离
    distance = np.empty((row, k))

    # 记录每个框最终所属的聚类
    last_clu = np.zeros((row,))

    np.random.seed(0)  # 固定随机种子，结果可复现

    # 随机选择k个框作为初始聚类中心
    cluster = box[np.random.choice(row, k, replace=False)]

    iter = 0
    while True:
        # 计算每个框到聚类中心的距离（1-IoU）
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i], cluster)

        # 找到每个框最近的聚类中心
        near = np.argmin(distance, axis=1)

        # 若聚类结果不再变化，停止迭代
        if (last_clu == near).all():
            break

        # 更新聚类中心（取中位数）
        for j in range(k):
            if np.sum(near == j) > 0:  # 避免空聚类
                cluster[j] = np.median(box[near == j], axis=0)

        last_clu = near
        if iter % 5 == 0:
            print(f'iter: {iter}. avg_iou:{avg_iou(box, cluster):.2f}')
        iter += 1

    return cluster, near


def load_data(path):
    """
    从YOLO格式TXT标注文件加载框的宽高数据
    path: labels目录路径（包含train/val子目录或直接是TXT文件）
    返回：归一化的宽高数组 [w, h]
    """
    data = []
    supported_ext = ('.txt',)

    # 遍历所有TXT文件（包括train/val子目录）
    txt_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(supported_ext):
                txt_files.append(os.path.join(root, file))

    if len(txt_files) == 0:
        raise ValueError(f"❌ 在{path}目录下未找到任何TXT标注文件！")

    # 解析每个TXT文件
    for txt_file in tqdm(txt_files, desc="加载标注文件"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # YOLO格式：class_id x_center y_center width height（均为归一化值）
                parts = line.split()
                if len(parts) < 5:
                    print(f"⚠️  {txt_file} 中存在无效行：{line}，跳过！")
                    continue

                # 提取宽高（已归一化）
                w = float(parts[3])
                h = float(parts[4])

                # 过滤无效框（宽高为0或负数）
                if w <= 0 or h <= 0 or w > 1 or h > 1:
                    print(f"⚠️  {txt_file} 中存在无效框（w={w}, h={h}），跳过！")
                    continue

                data.append([w, h])

        except Exception as e:
            print(f"⚠️  读取{txt_file}失败：{str(e)}，跳过！")
            continue

    if len(data) == 0:
        raise ValueError("❌ 未加载到任何有效框数据，请检查标注格式！")

    return np.array(data)


if __name__ == '__main__':
    np.random.seed(0)

    # -------------------------- 配置参数 --------------------------
    input_shape = [224, 224]  # 模型输入尺寸 [height, width]
    anchors_num = 9  # 聚类的锚框数量（YOLOv8默认9个，也可设为3/6）
    labels_path = 'VOCdevkit/VOC2026/labels'  # TXT标注根目录

    # -------------------------- 加载数据 --------------------------
    print('开始加载TXT标注文件...')
    data = load_data(labels_path)
    print(f'加载完成！共解析到 {len(data)} 个有效框')

    # -------------------------- K-means聚类 --------------------------
    print('开始K-means聚类...')
    cluster, near = kmeans(data, anchors_num)
    print('聚类完成！')

    # -------------------------- 转换为实际像素尺寸 --------------------------
    # 将归一化的锚框转换为输入尺寸下的像素值
    cluster_pixel = cluster * np.array([input_shape[1], input_shape[0]])
    data_pixel = data * np.array([input_shape[1], input_shape[0]])

    # -------------------------- 绘图可视化 --------------------------
    print('生成聚类可视化图...')
    plt.figure(figsize=(10, 8))
    for j in range(anchors_num):
        # 绘制每个聚类的框分布
        plt.scatter(data_pixel[near == j][:, 0], data_pixel[near == j][:, 1], alpha=0.6, label=f'cluster {j + 1}')
        # 绘制聚类中心（锚框）
        plt.scatter(cluster_pixel[j][0], cluster_pixel[j][1], marker='x', s=200, c='black', linewidths=3)

    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.title(f'K-means Anchors Clustering (num={anchors_num}, input_shape={input_shape})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("kmeans_for_anchors.jpg", dpi=150, bbox_inches='tight')
    plt.show()
    print('✅ 聚类可视化图已保存为 kmeans_for_anchors.jpg')

    # -------------------------- 输出结果 --------------------------
    # 按锚框面积排序（小→大）
    cluster_pixel = cluster_pixel[np.argsort(cluster_pixel[:, 0] * cluster_pixel[:, 1])]
    avg_iou_value = avg_iou(data_pixel, cluster_pixel)

    print('\n==================== 聚类结果 ====================')
    print(f'平均IoU：{avg_iou_value:.4f}')
    print(f'锚框尺寸（像素）：')
    for i, (w, h) in enumerate(cluster_pixel):
        print(f'  anchor {i + 1}: {int(round(w))}, {int(round(h))} (面积：{int(round(w * h))})')

    # 保存锚框到文件
    with open("yolo_anchors.txt", 'w', encoding='utf-8') as f:
        for i, (w, h) in enumerate(cluster_pixel):
            if i == 0:
                f.write(f"{int(round(w))},{int(round(h))}")
            else:
                f.write(f", {int(round(w))},{int(round(h))}")

    print('\n✅ 锚框已保存到 yolo_anchors.txt')
    print('==================================================')