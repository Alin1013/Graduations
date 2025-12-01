import os
import sys
import yaml
from typing import Dict, Any


def get_config(yaml_path: str = 'model_data/gesture.yaml') -> Dict[str, Any]:
    """
    加载并返回YAML配置文件内容

    Args:
        yaml_path: YAML配置文件路径，默认为'model_data/gesture.yaml'

    Returns:
        配置字典

    Raises:
        FileNotFoundError: 配置文件不存在时抛出
        yaml.YAMLError: YAML文件解析错误时抛出
    """
    # 检查文件是否存在
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"配置文件不存在: {os.path.abspath(yaml_path)}")

    # 检查文件是否为空
    if os.path.getsize(yaml_path) == 0:
        raise ValueError(f"配置文件为空: {os.path.abspath(yaml_path)}")

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # 验证必要配置项是否存在
        required_keys = ['train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in config:
                raise KeyError(f"配置文件缺少必要项: {key}")

        # 验证路径是否有效
        for path_key in ['train', 'val']:
            path = config[path_key]
            if not os.path.exists(path):
                raise NotADirectoryError(f"路径不存在: {path} (配置项: {path_key})")

        # 验证类别数量与名称列表长度一致
        if len(config['names']) != config['nc']:
            raise ValueError(
                f"类别数量不匹配: 配置的nc={config['nc']}, 但names列表长度为{len(config['names'])}"
            )

        return config

    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML文件解析错误: {str(e)}")
    except Exception as e:
        raise Exception(f"加载配置失败: {str(e)}")


if __name__ == "__main__":
    try:
        config = get_config()

        # 格式化输出配置信息
        print("=" * 50)
        print("配置文件加载成功:")
        print(f"配置文件路径: {os.path.abspath('model_data/gesture.yaml')}")
        print("-" * 50)
        print(f"训练集路径: {config['train']}")
        print(f"验证集路径: {config['val']}")
        print(f"类别数量: {config['nc']}")
        print(f"类别名称: {', '.join(config['names'])}")
        print(f"置信度阈值: {config.get('confidence', '未设置')}")
        print(f"NMS IoU阈值: {config.get('nms_iou', '未设置')}")
        print("=" * 50)

    except Exception as e:
        print(f"错误: {str(e)}", file=sys.stderr)
        sys.exit(1)