import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

# 配置路径
voc_xml_dir = "/Users/alin/Graduation_Project/VOCdevkit/VOC2026/Annotations"  # VOC的xml标签目录
yolo_txt_dir = "/Users/alin/Graduation_Project/VOCdevkit/VOC2026/Annotations"  # 输出YOLO格式txt的目录
image_dir = "/Users/alin/Graduation_Project/VOCdevkit/VOC2026/JPEGImages"  # 图片目录
classes = ["up", "down", "left", "right", "front", "back", "clockwise", "anticlockwise"]  # 类别列表

# 创建输出目录
os.makedirs(yolo_txt_dir, exist_ok=True)


# 转换单个xml文件为txt
def convert_xml_to_yolo(xml_path, txt_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取图片宽高
    size = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)

    with open(txt_path, 'w') as f:
        for obj in root.iter('object'):
            # 获取类别ID
            cls_name = obj.find('name').text
            if cls_name not in classes:
                continue
            cls_id = classes.index(cls_name)

            # 获取边界框坐标（VOC格式为xmin, ymin, xmax, ymax）
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # 转换为YOLO格式（归一化中心坐标和宽高）
            x_center = (xmin + xmax) / 2 / img_w
            y_center = (ymin + ymax) / 2 / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h

            # 写入txt（格式：class_id x_center y_center width height）
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


# 批量转换所有xml文件
xml_files = [f for f in os.listdir(voc_xml_dir) if f.endswith('.xml')]
for xml_file in tqdm(xml_files, desc="转换标签格式"):
    xml_path = os.path.join(voc_xml_dir, xml_file)
    txt_file = xml_file.replace('.xml', '.txt')  # 保持与图片同名（如a.jpg对应a.txt）
    txt_path = os.path.join(yolo_txt_dir, txt_file)
    convert_xml_to_yolo(xml_path, txt_path)

print("标签格式转换完成！")