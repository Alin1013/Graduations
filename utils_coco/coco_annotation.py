# -------------------------------------------------------#
#   用于处理COCO数据集，根据json文件生成txt文件用于训练
# -------------------------------------------------------#
import json
import os
import argparse
from collections import defaultdict


def convert_coco_to_txt(train_datasets_path, val_datasets_path,
                        train_annotation_path, val_annotation_path,
                        train_output_path, val_output_path):
    """
    将COCO格式的标注转换为YOLO训练所需的txt格式
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_output_path), exist_ok=True)

    # 处理训练集
    print("开始处理训练集标注...")
    process_annotation(
        annotation_path=train_annotation_path,
        datasets_path=train_datasets_path,
        output_path=train_output_path
    )

    # 处理验证集
    print("开始处理验证集标注...")
    process_annotation(
        annotation_path=val_annotation_path,
        datasets_path=val_datasets_path,
        output_path=val_output_path
    )

    print("标注转换完成!")


def process_annotation(annotation_path, datasets_path, output_path):
    """处理单个COCO标注文件"""
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"标注文件不存在: {annotation_path}")

    name_box_id = defaultdict(list)

    with open(annotation_path, encoding='utf-8') as f:
        data = json.load(f)

    annotations = data['annotations']
    for ant in annotations:
        image_id = ant['image_id']
        # COCO图像文件名格式为%012d.jpg
        image_name = os.path.join(datasets_path, f'%012d.jpg' % image_id)
        category_id = ant['category_id']

        # COCO类别ID映射（去除无用类别）
        category_id = map_coco_category(category_id)
        if category_id is None:
            continue  # 跳过不需要的类别

        name_box_id[image_name].append([ant['bbox'], category_id])

    # 写入txt文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for image_path in name_box_id.keys():
            if not os.path.exists(image_path):
                print(f"警告: 图像文件不存在 {image_path}")
                continue

            f.write(image_path)
            box_infos = name_box_id[image_path]
            for info in box_infos:
                bbox, cat = info
                x_min = int(bbox[0])
                y_min = int(bbox[1])
                x_max = x_min + int(bbox[2])
                y_max = y_min + int(bbox[3])
                f.write(f" {x_min},{y_min},{x_max},{y_max},{cat}")
            f.write('\n')


def map_coco_category(cat_id):
    """映射COCO类别ID到连续索引（去除人、背景等无用类别）"""
    if 1 <= cat_id <= 11:
        return cat_id - 1
    elif 13 <= cat_id <= 25:
        return cat_id - 2
    elif 27 <= cat_id <= 28:
        return cat_id - 3
    elif 31 <= cat_id <= 44:
        return cat_id - 5
    elif 46 <= cat_id <= 65:
        return cat_id - 6
    elif cat_id == 67:
        return cat_id - 7
    elif cat_id == 70:
        return cat_id - 9
    elif 72 <= cat_id <= 82:
        return cat_id - 10
    elif 84 <= cat_id <= 90:
        return cat_id - 11
    else:
        return None  # 跳过未定义的类别


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='COCO标注转换为YOLO格式')
    parser.add_argument('--train_datasets', default='coco_dataset/train2017',
                        help='训练集图像路径')
    parser.add_argument('--val_datasets', default='coco_dataset/val2017',
                        help='验证集图像路径')
    parser.add_argument('--train_anno', default='coco_dataset/annotations/instances_train2017.json',
                        help='训练集标注路径')
    parser.add_argument('--val_anno', default='coco_dataset/annotations/instances_val2017.json',
                        help='验证集标注路径')
    parser.add_argument('--train_out', default='coco_train.txt',
                        help='训练集输出txt路径')
    parser.add_argument('--val_out', default='coco_val.txt',
                        help='验证集输出txt路径')
    args = parser.parse_args()

    # 执行转换
    convert_coco_to_txt(
        train_datasets_path=args.train_datasets,
        val_datasets_path=args.val_datasets,
        train_annotation_path=args.train_anno,
        val_annotation_path=args.val_anno,
        train_output_path=args.train_out,
        val_output_path=args.val_out
    )