import os
import random
import shutil
from tqdm import tqdm

# -------------------------- 核心配置参数 --------------------------
# 数据集根路径（根据你的实际路径修改）
dataset_root = "/Users/alin/Graduation_Project/VOCdevkit/VOC2026"
# 原始图像/标注目录（上传的所有文件都放在这里）
src_images_dir = os.path.join(dataset_root, "images")  # 所有上传的图像
src_labels_dir = os.path.join(dataset_root, "labels")  # 所有上传的TXT标注
# 划分后输出目录（自动生成train/val子目录）
dst_images_dir = os.path.join(dataset_root, "images")  # 复用原目录，仅生成划分索引
dst_labels_dir = os.path.join(dataset_root, "labels")

# 数据集划分比例
train_percent = 0.9  # 训练集占比（验证集=1-0.9=0.1）
random_seed = 0  # 随机种子（保证划分结果可复现）
SUPPORTED_IMG_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG')  # 支持的图像格式

# -------------------------- 初始化目录 --------------------------
# 创建划分后的labels/train、labels/val目录
os.makedirs(os.path.join(dst_labels_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(dst_labels_dir, "val"), exist_ok=True)
# 创建划分后的images/train、images/val目录（可选：如需物理分隔图像，取消注释）
# os.makedirs(os.path.join(dst_images_dir, "train"), exist_ok=True)
# os.makedirs(os.path.join(dst_images_dir, "val"), exist_ok=True)

# 统计变量
total_valid_pairs = 0  # 有效图像+标注对数量
train_count = 0  # 训练集数量
val_count = 0  # 验证集数量
missing_labels = []  # 缺少标注的图像
missing_images = []  # 缺少图像的标注


# -------------------------- 第一步：筛选有效数据并划分 --------------------------
def get_valid_data_pairs():
    """
    遍历图像和标注目录，筛选出 图像-TXT标注 一一对应的有效数据对
    返回：所有有效数据的ID（文件名，不含后缀）
    """
    # 1. 获取所有图像ID（不含后缀）
    image_ids = set()
    for img_file in os.listdir(src_images_dir):
        img_name, img_ext = os.path.splitext(img_file)
        if img_ext.lower() in SUPPORTED_IMG_FORMATS:
            image_ids.add(img_name)

    # 2. 获取所有标注ID（不含后缀）
    label_ids = set()
    for label_file in os.listdir(src_labels_dir):
        label_name, label_ext = os.path.splitext(label_file)
        if label_ext.lower() == '.txt':
            label_ids.add(label_name)

    # 3. 筛选有效数据对（图像和标注都存在）
    valid_ids = list(image_ids & label_ids)  # 交集
    total_valid = len(valid_ids)

    # 4. 统计缺失情况
    global missing_labels, missing_images
    missing_labels = list(image_ids - label_ids)  # 有图像但无标注
    missing_images = list(label_ids - image_ids)  # 有标注但无图像

    # 输出数据校验结果
    print("📊 数据集校验结果：")
    print(f"   总图像数：{len(image_ids)}")
    print(f"   总标注数：{len(label_ids)}")
    print(f"   有效数据对（图像+标注）：{total_valid}")
    if missing_labels:
        print(f"   ⚠️  缺少标注的图像：{len(missing_labels)} 个（示例：{missing_labels[:5]}...）")
    if missing_images:
        print(f"   ⚠️  缺少图像的标注：{len(missing_images)} 个（示例：{missing_images[:5]}...）")

    if total_valid == 0:
        raise ValueError("❌ 未找到任何有效图像+标注对，请检查文件路径和命名！")

    return valid_ids


def split_train_val(valid_ids):
    """
    按比例划分训练集/验证集
    valid_ids: 有效数据ID列表
    返回：train_ids, val_ids
    """
    random.seed(random_seed)  # 固定种子，划分结果可复现
    num_train = int(len(valid_ids) * train_percent)
    train_ids = random.sample(valid_ids, num_train)
    val_ids = [id for id in valid_ids if id not in train_ids]

    print(f"\n📤 数据集划分结果：")
    print(f"   训练集：{len(train_ids)} 个")
    print(f"   验证集：{len(val_ids)} 个")
    return train_ids, val_ids


# -------------------------- 第二步：同步划分图像和标注 --------------------------
def copy_files_to_split_dirs(data_ids, split_type):
    """
    将指定ID的图像和标注复制/移动到对应split目录（train/val）
    data_ids: 要处理的数据ID列表
    split_type: "train" 或 "val"
    """
    global train_count, val_count
    count = 0

    # 目标目录
    dst_label_dir = os.path.join(dst_labels_dir, split_type)
    # 如需物理分隔图像，取消以下注释
    # dst_img_dir = os.path.join(dst_images_dir, split_type)
    # os.makedirs(dst_img_dir, exist_ok=True)

    for data_id in tqdm(data_ids, desc=f"处理{split_type}集"):
        # 1. 处理标注文件（复制到split目录）
        src_label = os.path.join(src_labels_dir, f"{data_id}.txt")
        dst_label = os.path.join(dst_label_dir, f"{data_id}.txt")
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)  # 复制（保留原文件），如需移动用shutil.move

        # 2. 处理图像文件（可选：物理分隔图像，取消以下注释）
        # 查找对应图像文件（匹配所有支持的格式）
        # src_img = None
        # for ext in SUPPORTED_IMG_FORMATS:
        #     temp_img = os.path.join(src_images_dir, f"{data_id}{ext}")
        #     if os.path.exists(temp_img):
        #         src_img = temp_img
        #         break
        # if src_img:
        #     dst_img = os.path.join(dst_img_dir, os.path.basename(src_img))
        #     shutil.copy2(src_img, dst_img)

        count += 1

    # 更新统计
    if split_type == "train":
        train_count = count
    else:
        val_count = count
    print(f"✅ {split_type}集处理完成：{count} 个数据对")


# -------------------------- 第三步：生成YOLO训练索引文件 --------------------------
def generate_yolo_index_files(train_ids, val_ids):
    """
    生成yolo_train.txt和yolo_val.txt（包含图像的绝对路径），用于YOLO训练
    """
    # 生成训练集索引
    with open(os.path.join(dataset_root, "yolo_train.txt"), "w", encoding="utf-8") as f:
        for data_id in train_ids:
            # 查找图像绝对路径
            for ext in SUPPORTED_IMG_FORMATS:
                img_path = os.path.join(src_images_dir, f"{data_id}{ext}")
                if os.path.exists(img_path):
                    f.write(os.path.abspath(img_path) + "\n")
                    break

    # 生成验证集索引
    with open(os.path.join(dataset_root, "yolo_val.txt"), "w", encoding="utf-8") as f:
        for data_id in val_ids:
            # 查找图像绝对路径
            for ext in SUPPORTED_IMG_FORMATS:
                img_path = os.path.join(src_images_dir, f"{data_id}{ext}")
                if os.path.exists(img_path):
                    f.write(os.path.abspath(img_path) + "\n")
                    break

    print(f"\n📜 YOLO训练索引文件生成完成：")
    print(f"   - 训练集索引：{os.path.join(dataset_root, 'yolo_train.txt')}")
    print(f"   - 验证集索引：{os.path.join(dataset_root, 'yolo_val.txt')}")


# -------------------------- 主函数入口 --------------------------
if __name__ == "__main__":
    try:
        # 1. 筛选有效数据对
        valid_ids = get_valid_data_pairs()

        # 2. 划分训练集/验证集
        train_ids, val_ids = split_train_val(valid_ids)

        # 3. 同步划分标注文件（图像可选物理分隔）
        copy_files_to_split_dirs(train_ids, "train")
        copy_files_to_split_dirs(val_ids, "val")

        # 4. 生成YOLO训练所需的索引文件
        generate_yolo_index_files(train_ids, val_ids)

        # 最终统计
        print("\n" + "=" * 60)
        print("🎉 数据集划分全部完成！最终统计：")
        print(f"   总有效数据对：{len(valid_ids)}")
        print(f"   训练集：{train_count} 个（标注已复制到 {os.path.join(dst_labels_dir, 'train')}）")
        print(f"   验证集：{val_count} 个（标注已复制到 {os.path.join(dst_labels_dir, 'val')}）")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 执行失败：{str(e)}")
        exit(1)