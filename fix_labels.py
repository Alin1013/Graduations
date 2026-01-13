import os
import glob

# ========== 配置你的数据集labels根路径 ==========
label_root = "/Users/alin/Graduation_Project/VOCdevkit/VOC2026/labels/"

# 遍历所有txt标签文件
txt_paths = glob.glob(os.path.join(label_root, "**/*.txt"), recursive=True)
print(f"检测到 {len(txt_paths)} 个标签文件，开始修正...")

for txt in txt_paths:
    with open(txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 拆分YOLO格式：class_id x y w h
        parts = line.split()
        class_id = int(parts[0])

        # ========== 核心修正逻辑 ==========
        # 情况1：如果ID=18是多余的错误标签 → 直接跳过该行（删除该标注框）
        if class_id == 18:
            continue
        # 情况2：如果你的类别ID整体多了1（比如应该是0-17，标成了1-18）→ 执行下面一行，注释上面一行
        # class_id = class_id - 1

        # 修正后重新拼接行
        new_line = " ".join([str(class_id)] + parts[1:]) + "\n"
        new_lines.append(new_line)

    # 写入修正后的内容
    with open(txt, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

print("✅ 所有标签文件修正完成！")