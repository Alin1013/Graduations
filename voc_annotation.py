import os
import random
import xml.etree.ElementTree as ET
from get_yaml import get_config
#划分验证集和测试集
# --------------------------------------------------------------------------------------------------------------------------------#
#   配置项（已根据实际情况优化）
# --------------------------------------------------------------------------------------------------------------------------------#
annotation_mode = 0
trainval_percent = 1.0
train_percent = 0.9
VOCdevkit_path = 'VOCdevkit'
VOCdevkit_sets = [('2026', 'train'), ('2026', 'val')]

# 图像实际存储目录（已确认正确）
IMG_DIR = os.path.join(VOCdevkit_path, 'VOC2026', 'JPEGImages')
# 支持的图像后缀 + 自动适配 _000 后缀
SUPPORTED_IMG_FORMATS = ('.jpg', '.jpeg', '.png')
IMAGE_SUFFIX_ADDON = '_000'  # 图像文件名比image_id多的后缀（关键适配）

# 从gesture.yaml读取类别配置
config = get_config()
classes = config['names']
nc = config['nc']
print(f"✅ 从配置文件加载类别：{classes}（共{nc}类）")
print(f"✅ 图像目录：{IMG_DIR}")
print(f"✅ 支持后缀：{SUPPORTED_IMG_FORMATS}")
print(f"✅ 图像文件名附加后缀：{IMAGE_SUFFIX_ADDON}")


def convert_annotation(year, image_id, list_file):
    """过滤有效标签"""
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml' % (year, image_id)), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    has_valid_obj = False

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls in classes and int(difficult) == 0:
            has_valid_obj = True
            break

    return has_valid_obj


def get_real_img_path(image_id):
    """查找实际图像路径（适配 image_id + _000 + 后缀）"""
    # 先尝试带 _000 后缀的路径（主要适配你的图像）
    for ext in SUPPORTED_IMG_FORMATS:
        img_path = os.path.join(IMG_DIR, f"{image_id}{IMAGE_SUFFIX_ADDON}{ext}")
        if os.path.exists(img_path):
            return os.path.abspath(img_path)
    # 再尝试不带 _000 的路径（兼容其他情况）
    for ext in SUPPORTED_IMG_FORMATS:
        img_path = os.path.join(IMG_DIR, f"{image_id}{ext}")
        if os.path.exists(img_path):
            return os.path.abspath(img_path)
    # 都没找到返回None
    return None


if __name__ == "__main__":
    random.seed(0)

    # 步骤1：生成ImageSets中的划分文件
    if annotation_mode == 0 or annotation_mode == 1:
        print("\nGenerate txt in ImageSets.")
        xmlfilepath = os.path.join(VOCdevkit_path, 'VOC2026/Annotations')
        saveBasePath = os.path.join(VOCdevkit_path, 'VOC2026/ImageSets/Main')
        total_xml = [xml for xml in os.listdir(xmlfilepath) if xml.endswith(".xml")]

        num = len(total_xml)
        tv = int(num * trainval_percent)
        tr = int(tv * train_percent)
        trainval = random.sample(range(num), tv)
        train = random.sample(trainval, tr)

        print(f"train and val size: {tv}")
        print(f"train size: {tr}")
        print(f"val size: {tv - tr}")

        # 写入划分文件
        with open(os.path.join(saveBasePath, 'trainval.txt'), 'w') as ftrainval, \
                open(os.path.join(saveBasePath, 'test.txt'), 'w') as ftest, \
                open(os.path.join(saveBasePath, 'train.txt'), 'w') as ftrain, \
                open(os.path.join(saveBasePath, 'val.txt'), 'w') as fval:
            for i in range(num):
                name = total_xml[i][:-4] + '\n'  # 去除.xml后缀，得到image_id
                if i in trainval:
                    ftrainval.write(name)
                    if i in train:
                        ftrain.write(name)
                    else:
                        fval.write(name)
                else:
                    ftest.write(name)

        print("Generate txt in ImageSets done.")

    # 步骤2：生成YOLOv8所需的纯图像路径列表（关键适配_000后缀）
    if annotation_mode == 0 or annotation_mode == 2:
        print("\nGenerate yolo_train.txt and yolo_val.txt for train.")
        for year, image_set in VOCdevkit_sets:
            image_ids_path = os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt' % (year, image_set))
            image_ids = open(image_ids_path, encoding='utf-8').read().strip().split()

            output_file = f"yolo_{image_set}.txt"
            valid_count = 0

            with open(output_file, 'w', encoding='utf-8') as list_file:
                for idx, image_id in enumerate(image_ids):
                    img_path = get_real_img_path(image_id)

                    if img_path:
                        has_valid = convert_annotation(year, image_id, list_file)
                        if has_valid:
                            list_file.write(img_path + '\n')
                            valid_count += 1
                            # 每10个输出一次进度
                            if (idx + 1) % 10 == 0:
                                print(f"🔍 已处理 {idx + 1}/{len(image_ids)} 个图像，有效数：{valid_count}")
                        else:
                            print(f"⚠️  图像{image_id}{IMAGE_SUFFIX_ADDON}有文件但无有效标签，已跳过")
                    else:
                        print(f"❌ 未找到图像{image_id}（尝试了 {image_id}{IMAGE_SUFFIX_ADDON}{SUPPORTED_IMG_FORMATS}）")

            print(f"\n✅ {output_file} 生成完成！")
            print(f"📊 统计：总图像数 {len(image_ids)}，有效图像数 {valid_count}")

        print("\nGenerate yolo_train.txt and yolo_val.txt for train done.")