"""
Milvus配置管理
"""
from typing import Optional
from pymilvus import MilvusClient
from core import settings
import logging

logger = logging.getLogger(__name__)


class MilvusConfig:
    """Milvus配置类"""
    
    def __init__(self):
        """
        从settings初始化Milvus配置
        """
        self.uri = settings.MILVUS.get("hosts", "http://localhost:19530")
        self.user = settings.MILVUS.get("username", "")
        self.password = settings.MILVUS.get("password", "")
        self.db_name = settings.MILVUS.get("db_name", "")
        self.token = settings.MILVUS.get("token", "")
        self.timeout = settings.MILVUS.get("timeout", None)
        self.kwargs = settings.MILVUS.get("kwargs", {})
        self._client = None
    
    def get_client(self) -> MilvusClient:
        """
        获取Milvus客户端实例 (单例模式)
        
        Returns:
            MilvusClient实例
        """
        if self._client is None:
            try:
                self._client = MilvusClient(
                    uri=self.uri,
                    user=self.user,
                    password=self.password,
                    db_name=self.db_name,
                    token=self.token,
                    timeout=self.timeout,
                    **self.kwargs
                )
                logger.info(f"成功连接到Milvus: {self.uri}")
            except Exception as e:
                logger.error(f"连接Milvus失败: {e}")
                raise
        
        return self._client
    
    def close_connection(self):
        """关闭Milvus连接"""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Milvus连接已关闭")


# 默认配置实例
default_milvus_config = MilvusConfig()


def get_default_milvus_client() -> MilvusClient:
    """
    获取默认的Milvus客户端
    
    Returns:
        MilvusClient实例
    """
    return default_milvus_config.get_client()


# 房源推荐服务配置
HOUSE_RECO_CONFIG = {
    "collection_name": "house_recommendation",
    "vector_dim": 1024,
    "max_batch_size": 1000,
    "search_timeout": 30.0,
    "index_type": "AUTOINDEX",
    "metric_type": "COSINE"
}