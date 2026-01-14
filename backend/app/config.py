"""
应用配置文件
"""
import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent.parent.parent

class Settings:
    """应用设置"""
    # 数据库配置
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/backend/data/gesture_detection.db")
    
    # 文件存储路径
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(BASE_DIR / "backend" / "uploads"))
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(BASE_DIR / "backend" / "outputs"))
    
    # 模型配置
    DEFAULT_MODEL_PATH = os.getenv(
        "DEFAULT_MODEL_PATH",
        str(BASE_DIR / "runs" / "detect" / "gesture_final_train" / "weights" / "best.pt")
    )
    
    # API配置
    API_V1_PREFIX = "/api"
    
    # 文件大小限制（100MB）
    MAX_UPLOAD_SIZE = 100 * 1024 * 1024

settings = Settings()
