import os
import sys
import yaml
from typing import Dict, Any, Optional


def get_config(yaml_path: str = 'model_data/gesture.yaml') -> Dict[str, Any]:
    """
    加载并返回YAML配置文件内容，包含严格的合法性校验

    Args:
        yaml_path: YAML配置文件路径，默认为'model_data/gesture.yaml'

    Returns:
        配置字典（包含所有YAML中的配置项）

    Raises:
        FileNotFoundError: 配置文件不存在时抛出
        ValueError: 配置文件为空或关键配置不合法时抛出
        yaml.YAMLError: YAML文件解析错误时抛出
        KeyError: 缺少必要配置项时抛出
        NotADirectoryError: 配置的路径不是有效目录时抛出
    """
    # 标准化路径（处理相对路径/绝对路径）
    yaml_path = os.path.abspath(yaml_path)

    # 1. 检查文件是否存在
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"配置文件不存在: {yaml_path}")

    # 2. 检查文件是否为空
    if os.path.getsize(yaml_path) == 0:
        raise ValueError(f"配置文件为空: {yaml_path}")

    try:
        # 3. 读取并解析YAML文件
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # 校验解析结果是否为字典
        if not isinstance(config, dict):
            raise ValueError(f"配置文件格式错误: 根节点必须是字典，实际为{type(config)}")

        # 4. 验证必要配置项是否存在
        required_keys = ['train', 'val', 'nc', 'names']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise KeyError(f"配置文件缺少必要项: {', '.join(missing_keys)}")

        # 5. 验证类别配置合法性
        # 5.1 校验nc为正整数
        if not isinstance(config['nc'], int) or config['nc'] <= 0:
            raise ValueError(f"类别数量(nc)必须为正整数，实际为{config['nc']}")

        # 5.2 校验names为非空列表且元素为字符串
        if not isinstance(config['names'], list) or len(config['names']) == 0:
            raise ValueError(f"类别名称(names)必须为非空列表，实际为{type(config['names'])}")

        for idx, name in enumerate(config['names']):
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"类别名称必须为非空字符串，索引{idx}的值为{repr(name)}")

        # 5.3 校验类别数量匹配
        if len(config['names']) != config['nc']:
            raise ValueError(
                f"类别数量不匹配: 配置的nc={config['nc']}, 但names列表长度为{len(config['names'])}"
            )

        # 6. 验证路径配置（兼容文件列表路径和目录路径）
        for path_key in ['train', 'val']:
            path = config[path_key]
            if not isinstance(path, str):
                raise ValueError(f"{path_key}路径必须为字符串，实际为{type(path)}")

            # 标准化路径（基于配置文件所在目录）
            config_dir = os.path.dirname(yaml_path)
            abs_path = os.path.join(config_dir, path) if not os.path.isabs(path) else path
            abs_path = os.path.normpath(abs_path)

            # 更新配置中的路径为绝对路径
            config[path_key] = abs_path

            # 路径存在性校验（兼容文件/目录）
            if not os.path.exists(abs_path):
                raise NotADirectoryError(f"{path_key}路径不存在: {abs_path}")

            # 可选：校验路径可访问
            if not os.access(abs_path, os.R_OK):
                raise PermissionError(f"无读取权限: {abs_path} (配置项: {path_key})")

        # 7. 补充默认配置（提升兼容性）
        default_configs = {
            'confidence': 0.5,
            'nms_iou': 0.5,
            'dir_detect_path': 'VOCdevkit/VOC2026/images',
            'detect_save_path': 'auto_annotations',
            'visualize_attention': False
        }
        for key, default_val in default_configs.items():
            if key not in config:
                config[key] = default_val

        return config

    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML文件解析错误: {str(e)} (文件: {yaml_path})")
    except Exception as e:
        # 统一异常类型，方便上层处理
        raise Exception(f"加载配置失败: {str(e)} (文件: {yaml_path})")


def print_config(config: Dict[str, Any], yaml_path: str) -> None:
    """
    格式化打印配置信息（美观且易读）

    Args:
        config: 加载后的配置字典
        yaml_path: 配置文件路径
    """
    print("=" * 60)
    print(f"📄 配置文件加载成功")
    print(f"🔍 文件路径: {os.path.abspath(yaml_path)}")
    print("-" * 60)
    print(f"📁 训练集路径: {config['train']}")
    print(f"📁 验证集路径: {config['val']}")
    print(f"📊 类别数量: {config['nc']}")
    print(f"🏷️  类别名称: {', '.join(config['names'])}")
    print(f"🎯 置信度阈值: {config['confidence']}")
    print(f"🔗 NMS IoU阈值: {config['nms_iou']}")
    print(f"🔍 检测图像目录: {config['dir_detect_path']}")
    print(f"💾 检测结果保存目录: {config['detect_save_path']}")
    print(f"👁️  注意力可视化: {config['visualize_attention']}")
    print("=" * 60)


if __name__ == "__main__":
    # 支持命令行指定配置文件路径
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else 'model_data/gesture.yaml'

    try:
        # 加载配置
        config = get_config(yaml_path)

        # 格式化输出配置信息
        print_config(config, yaml_path)

    except Exception as e:
        # 错误输出到标准错误流
        print(f"\n❌ 错误: {str(e)}", file=sys.stderr)
        sys.exit(1)