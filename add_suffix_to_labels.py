# 脚本后添加后缀_000脚本
import os

# 标签文件所在目录（替换为你的标签目录）
LABEL_DIR = "/Users/alin/Graduation_Project/VOCdevkit/VOC2026/labels"
SUFFIX = "_000"  # 需要添加的后缀

for label_file in os.listdir(LABEL_DIR):
    if label_file.endswith(".txt") and SUFFIX not in label_file:
        # 去除 .txt 后缀，添加 _000，再加上 .txt
        new_name = label_file[:-4] + SUFFIX + ".txt"
        old_path = os.path.join(LABEL_DIR, label_file)
        new_path = os.path.join(LABEL_DIR, new_name)
        os.rename(old_path, new_path)
        print(f"✅ 已重命名：{label_file} → {new_name}")

print("\n所有标签文件后缀添加完成！")