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
# 划分后输出目录
dst_images_dir = os.path.join(dataset_root, "images")  # 图像划分后目录
dst_labels_dir = os.path.join(dataset_root, "labels")  # 标注划分后目录

# 数据集划分比例
train_percent = 0.9  # 训练集占比（验证集=1-0.9=0.1）
random_seed = 0  # 随机种子（保证划分结果可复现）
SUPPORTED_IMG_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG')  # 支持的图像格式

# -------------------------- 初始化目录 --------------------------
# 创建划分后的目录结构
dirs_to_create = [
    os.path.join(dst_labels_dir, "train"),
    os.path.join(dst_labels_dir, "val"),
    os.path.join(dst_images_dir, "train"),
    os.path.join(dst_images_dir, "val")
]
for dir_path in dirs_to_create:
    os.makedirs(dir_path, exist_ok=True)

# 统计变量
total_valid_pairs = 0  # 有效图像+标注对数量
train_count = 0  # 训练集数量
val_count = 0  # 验证集数量
missing_labels = []  # 缺少标注的图像
missing_images = []  # 缺少图像的标注
moved_files = {"train": {"images": [], "labels": []}, "val": {"images": [], "labels": []}}  # 记录移动的文件


# -------------------------- 工具函数 --------------------------
def safe_remove(file_path):
    """安全删除文件（处理不存在的情况）"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
    except Exception as e:
        print(f"⚠️ 删除文件失败 {file_path}: {e}")
    return False


def get_image_file_path(data_id, img_dir):
    """根据data_id查找对应图像文件的完整路径"""
    for ext in SUPPORTED_IMG_FORMATS:
        img_path = os.path.join(img_dir, f"{data_id}{ext}")
        if os.path.exists(img_path):
            return img_path
    return None


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
        if img_ext.lower() in SUPPORTED_IMG_FORMATS and not os.path.isdir(os.path.join(src_images_dir, img_file)):
            image_ids.add(img_name)

    # 2. 获取所有标注ID（不含后缀）
    label_ids = set()
    for label_file in os.listdir(src_labels_dir):
        label_name, label_ext = os.path.splitext(label_file)
        if label_ext.lower() == '.txt' and not os.path.isdir(os.path.join(src_labels_dir, label_file)):
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


# -------------------------- 第二步：移动文件到对应目录 --------------------------
def move_files_to_split_dirs(data_ids, split_type):
    """
    将指定ID的图像和标注移动到对应split目录（train/val）
    data_ids: 要处理的数据ID列表
    split_type: "train" 或 "val"
    """
    global train_count, val_count
    count = 0

    # 目标目录
    dst_label_dir = os.path.join(dst_labels_dir, split_type)
    dst_img_dir = os.path.join(dst_images_dir, split_type)

    for data_id in tqdm(data_ids, desc=f"处理{split_type}集"):
        # 1. 处理标注文件（移动到split目录）
        src_label = os.path.join(src_labels_dir, f"{data_id}.txt")
        dst_label = os.path.join(dst_label_dir, f"{data_id}.txt")

        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)
            moved_files[split_type]["labels"].append(dst_label)

        # 2. 处理图像文件（移动到split目录）
        src_img = get_image_file_path(data_id, src_images_dir)
        if src_img:
            dst_img = os.path.join(dst_img_dir, os.path.basename(src_img))
            shutil.move(src_img, dst_img)
            moved_files[split_type]["images"].append(dst_img)

        count += 1

    # 更新统计
    if split_type == "train":
        train_count = count
    else:
        val_count = count
    print(f"✅ {split_type}集处理完成：{count} 个数据对")


# -------------------------- 第三步：清理根目录冗余文件 --------------------------
def clean_root_dirs():
    """
    清理labels根目录下的所有TXT文件（已移动到train/val）
    清理images根目录下的孤立文件（无对应标注的）
    """
    print("\n🧹 开始清理根目录冗余文件...")

    # 1. 清理labels根目录下的所有TXT文件
    label_files_removed = 0
    for file in os.listdir(src_labels_dir):
        file_path = os.path.join(src_labels_dir, file)
        if file.lower().endswith('.txt') and not os.path.isdir(file_path):
            if safe_remove(file_path):
                label_files_removed += 1

    # 2. 清理images根目录下无标注的图像文件（可选：保留或删除）
    img_files_removed = 0
    for file in os.listdir(src_images_dir):
        file_path = os.path.join(src_images_dir, file)
        img_ext = os.path.splitext(file)[1].lower()

        if img_ext in SUPPORTED_IMG_FORMATS and not os.path.isdir(file_path):
            # 检查是否有对应的标注（在train/val目录中）
            img_id = os.path.splitext(file)[0]
            has_label = False

            # 检查train目录
            if os.path.exists(os.path.join(dst_labels_dir, "train", f"{img_id}.txt")):
                has_label = True
            # 检查val目录
            elif os.path.exists(os.path.join(dst_labels_dir, "val", f"{img_id}.txt")):
                has_label = True

            # 如果没有对应标注，删除图像文件
            if not has_label:
                if safe_remove(file_path):
                    img_files_removed += 1

    print(f"✅ 清理完成：")
    print(f"   - Labels根目录删除TXT文件：{label_files_removed} 个")
    print(f"   - Images根目录删除孤立图像：{img_files_removed} 个")


# -------------------------- 第四步：生成YOLO训练索引文件 --------------------------
def generate_yolo_index_files(train_ids, val_ids):
    """
    生成yolo_train.txt和yolo_val.txt（包含图像的绝对路径），用于YOLO训练
    """
    # 生成训练集索引
    with open(os.path.join(dataset_root, "yolo_train.txt"), "w", encoding="utf-8") as f:
        for data_id in train_ids:
            img_path = get_image_file_path(data_id, os.path.join(dst_images_dir, "train"))
            if img_path:
                f.write(os.path.abspath(img_path) + "\n")

    # 生成验证集索引
    with open(os.path.join(dataset_root, "yolo_val.txt"), "w", encoding="utf-8") as f:
        for data_id in val_ids:
            img_path = get_image_file_path(data_id, os.path.join(dst_images_dir, "val"))
            if img_path:
                f.write(os.path.abspath(img_path) + "\n")

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

        # 3. 移动文件到对应目录（物理分隔）
        move_files_to_split_dirs(train_ids, "train")
        move_files_to_split_dirs(val_ids, "val")

        # 4. 清理根目录冗余文件
        clean_root_dirs()

        # 5. 生成YOLO训练所需的索引文件
        generate_yolo_index_files(train_ids, val_ids)

        # 最终统计
        print("\n" + "=" * 60)
        print("🎉 数据集划分全部完成！最终统计：")
        print(f"   总有效数据对：{len(valid_ids)}")
        print(f"   训练集：{train_count} 个")
        print(f"     - 图像：{len(moved_files['train']['images'])} 个（{os.path.join(dst_images_dir, 'train')}）")
        print(f"     - 标注：{len(moved_files['train']['labels'])} 个（{os.path.join(dst_labels_dir, 'train')}）")
        print(f"   验证集：{val_count} 个")
        print(f"     - 图像：{len(moved_files['val']['images'])} 个（{os.path.join(dst_images_dir, 'val')}）")
        print(f"     - 标注：{len(moved_files['val']['labels'])} 个（{os.path.join(dst_labels_dir, 'val')}）")
        print("=" * 60)

        # 最终目录结构提示
        print("\n📁 最终目录结构：")
        print(f"""
        {dataset_root}/
        ├── images/
        │   ├── train/       # 训练集图像
        │   └── val/         # 验证集图像
        ├── labels/
        │   ├── train/       # 训练集标注
        │   └── val/         # 验证集标注
        ├── yolo_train.txt   # 训练集索引
        └── yolo_val.txt     # 验证集索引
        """)

    except Exception as e:
        print(f"\n❌ 执行失败：{str(e)}")
        # 可选：出错时恢复文件（如需容错，可取消注释）
        # for split_type in ["train", "val"]:
        #     # 恢复图像
        #     for img_path in moved_files[split_type]["images"]:
        #         if os.path.exists(img_path):
        #             shutil.move(img_path, src_images_dir)
        #     # 恢复标注
        #     for label_path in moved_files[split_type]["labels"]:
        #         if os.path.exists(label_path):
        #             shutil.move(label_path, src_labels_dir)
        exit(1)