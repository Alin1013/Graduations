#转换格式脚本
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image
from PIL import UnidentifiedImageError  # 处理图像读取失败的异常

# -------------------------- 配置参数（已适配你的数据集结构）--------------------------
# 注意：你的数据集已重构为 images/train、images/val 目录，XML 应对应放在 Annotations/train、Annotations/val
voc_xml_root = "/Users/alin/Graduation_Project/VOCdevkit/VOC2026/Annotations"  # XML根目录（包含train/val子目录）
yolo_txt_root = "/Users/alin/Graduation_Project/VOCdevkit/VOC2026/labels"      # YOLO TXT根目录（会自动生成train/val子目录）
image_root = "/Users/alin/Graduation_Project/VOCdevkit/VOC2026/images"         # 图像根目录（包含train/val子目录）
classes = ["call", "no_gesture"]  # 类别列表（必须与XML中<name>完全一致，大小写敏感）
# -------------------------------------------------------------------------------------------

# 创建YOLO TXT根目录及train/val子目录
os.makedirs(yolo_txt_root, exist_ok=True)
os.makedirs(os.path.join(yolo_txt_root, "train"), exist_ok=True)
os.makedirs(os.path.join(yolo_txt_root, "val"), exist_ok=True)

# 统计转换情况
total_files = 0
success_files = 0
empty_files = 0
error_files = []


def convert_xml_to_yolo(xml_path, txt_path, image_dir):
    """
    转换单个XML文件为YOLO格式TXT文件
    xml_path: XML文件路径
    txt_path: 输出TXT文件路径
    image_dir: 对应图像所在目录（train/val）
    返回：True（成功有标注）、False（成功无标注）、None（失败）
    """
    global success_files, empty_files
    try:
        # 解析XML文件（处理XML格式错误）
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            print(f"\n警告：{os.path.basename(xml_path)} XML格式错误，跳过！原因：{str(e)}")
            return None

        # 1. 获取图像尺寸（优先XML，失败则从图像读取）
        img_w, img_h = None, None
        size = root.find('size')
        if size is not None:
            width_elem = size.find('width')
            height_elem = size.find('height')
            if width_elem is not None and height_elem is not None:
                try:
                    img_w = int(width_elem.text.strip()) if width_elem.text else None
                    img_h = int(height_elem.text.strip()) if height_elem.text else None
                except ValueError:
                    img_w, img_h = None, None  # 数值无效

        # 从图像文件读取尺寸（XML尺寸缺失/无效时）
        if not (img_w and img_h and img_w > 0 and img_h > 0):
            img_filename = os.path.splitext(os.path.basename(xml_path))[0]
            img_suffixes = ['.jpg', '.jpeg', '.png', '.bmp']
            img_path = None
            for suffix in img_suffixes:
                temp_path = os.path.join(image_dir, img_filename + suffix)
                if os.path.exists(temp_path):
                    img_path = temp_path
                    break
            if not img_path:
                print(f"\n警告：{os.path.basename(xml_path)} 未找到对应图像，跳过！")
                return None

            # 读取图像尺寸（处理图像损坏/格式不支持）
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except UnidentifiedImageError:
                print(f"\n警告：{os.path.basename(xml_path)} 对应图像损坏或不支持，跳过！")
                return None
            except PermissionError:
                print(f"\n警告：无权限读取 {os.path.basename(xml_path)} 对应图像，跳过！")
                return None

        # 2. 提取并转换目标标注
        yolo_lines = []
        for obj in root.iter('object'):
            # 获取类别名称
            cls_name_elem = obj.find('name')
            if not cls_name_elem or not cls_name_elem.text:
                print(f"\n警告：{os.path.basename(xml_path)} 存在无类别名称的目标，跳过！")
                continue
            cls_name = cls_name_elem.text.strip()

            # 校验类别
            if cls_name not in classes:
                print(f"\n警告：{os.path.basename(xml_path)} 出现未知类别 '{cls_name}'，跳过该目标！")
                continue
            cls_id = classes.index(cls_name)

            # 获取边界框（VOC格式：xmin, ymin, xmax, ymax）
            bndbox = obj.find('bndbox')
            if not bndbox:
                print(f"\n警告：{os.path.basename(xml_path)} 中类别 '{cls_name}' 无边界框，跳过！")
                continue

            # 读取并校验坐标
            try:
                xmin = float(bndbox.find('xmin').text.strip()) if bndbox.find('xmin').text else 0.0
                ymin = float(bndbox.find('ymin').text.strip()) if bndbox.find('ymin').text else 0.0
                xmax = float(bndbox.find('xmax').text.strip()) if bndbox.find('xmax').text else 0.0
                ymax = float(bndbox.find('ymax').text.strip()) if bndbox.find('ymax').text else 0.0
            except (AttributeError, ValueError):
                print(f"\n警告：{os.path.basename(xml_path)} 中类别 '{cls_name}' 坐标无效，跳过！")
                continue

            # 修正无效边界框
            if xmin >= xmax or ymin >= ymax:
                print(f"\n警告：{os.path.basename(xml_path)} 中类别 '{cls_name}' 边界框无效（xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}），跳过！")
                continue
            xmin = max(0.0, xmin)
            ymin = max(0.0, ymin)
            xmax = min(float(img_w), xmax)
            ymax = min(float(img_h), ymax)

            # 转换为YOLO格式（归一化中心坐标+宽高）
            x_center = (xmin + xmax) / 2 / img_w
            y_center = (ymin + ymax) / 2 / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h

            # 确保坐标在0-1范围内（避免训练报错）
            x_center = max(0.001, min(0.999, x_center))
            y_center = max(0.001, min(0.999, y_center))
            width = max(0.001, min(0.999, width))
            height = max(0.001, min(0.999, height))

            # 添加到结果（保留6位小数，符合YOLO标准）
            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # 3. 写入TXT文件（空文件也写入，避免训练时找不到标签）
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))

        # 统计
        if len(yolo_lines) > 0:
            success_files += 1
            return True
        else:
            empty_files += 1
            return False

    except Exception as e:
        error_files.append(os.path.basename(xml_path))
        print(f"\n错误：转换 {os.path.basename(xml_path)} 失败，原因：{str(e)}")
        return None


# -------------------------- 批量转换入口（处理train/val子目录）--------------------------
if __name__ == "__main__":
    # 遍历train和val子目录
    for split in ["train", "val"]:
        voc_xml_dir = os.path.join(voc_xml_root, split)
        yolo_txt_dir = os.path.join(yolo_txt_root, split)
        image_dir = os.path.join(image_root, split)

        # 检查XML目录是否存在
        if not os.path.exists(voc_xml_dir):
            print(f"警告：未找到 {voc_xml_dir} 目录，跳过 {split} 集转换！")
            continue

        # 获取该目录下所有XML文件
        xml_files = [f for f in os.listdir(voc_xml_dir) if f.endswith('.xml')]
        current_total = len(xml_files)
        total_files += current_total

        if current_total == 0:
            print(f"警告：{voc_xml_dir} 中无XML文件，跳过 {split} 集转换！")
            continue

        print(f"\n开始转换 {split} 集（{current_total} 个文件）...")
        for xml_file in tqdm(xml_files, desc=f"转换 {split} 集标签"):
            xml_path = os.path.join(voc_xml_dir, xml_file)
            txt_file = os.path.splitext(xml_file)[0] + '.txt'  # 保持与图像同名
            txt_path = os.path.join(yolo_txt_dir, txt_file)
            convert_xml_to_yolo(xml_path, txt_path, image_dir)

    # 输出转换统计报告
    print("\n" + "="*60)
    print("VOC XML → YOLO TXT 转换完成！统计结果：")
    print(f"总处理文件数：{total_files}")
    print(f"成功转换（含有效标注）：{success_files} 个")
    print(f"生成空文件（无有效标注）：{empty_files} 个")
    print(f"转换失败：{len(error_files)} 个")
    if error_files:
        print(f"失败文件列表：{error_files[:10]}...")  # 只显示前10个失败文件
    print("="*60)
    print(f"YOLO格式标签保存路径：{yolo_txt_root}")
    print(f"  - 训练集标签：{os.path.join(yolo_txt_root, 'train')}")
    print(f"  - 验证集标签：{os.path.join(yolo_txt_root, 'val')}")