"""
数据库配置和初始化
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from app.config import settings

# 创建数据库引擎
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 声明基类
Base = declarative_base()


def get_db():
    """
    获取数据库会话（依赖注入）
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    初始化数据库，创建所有表
    """
    # 确保数据库目录存在
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    if db_path and not db_path.startswith(":"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # 导入所有模型以确保它们被注册
    from app.models import VideoRecord, DetectionResult
    
    # 创建所有表
    Base.metadata.create_all(bind=engine)
    print(f"数据库初始化完成: {settings.DATABASE_URL}")
