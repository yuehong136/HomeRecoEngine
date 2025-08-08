"""
向量化工具类
用于处理房源数据的语义向量化
"""
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from core.llm.embedding_model import DefaultEmbedding
from api.db.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


class VectorizationUtils:
    """向量化工具类"""
    
    def __init__(self):
        """初始化向量化工具"""
        self.embedding_model = None
        self._init_embedding_model()
    
    def _init_embedding_model(self):
        """初始化嵌入模型"""
        try:
            # 使用默认的BGE模型进行向量化
            self.embedding_model = DefaultEmbedding(
                key="", 
                model_name="BAAI/bge-large-zh-v1.5"
            )
            logger.info("向量化模型初始化成功")
        except Exception as e:
            logger.error(f"向量化模型初始化失败: {e}")
            self.embedding_model = None
    
    def create_semantic_vector(self, house_data: Dict[str, Any]) -> Optional[List[float]]:
        """
        为房源数据创建语义向量
        
        Args:
            house_data: 房源数据字典
            
        Returns:
            语义向量列表，如果失败返回None
        """
        if not self.embedding_model:
            logger.warning("嵌入模型未初始化，返回零向量")
            return [0.0] * 1024
            
        try:
            # 合并所有语义字段
            semantic_text = self._combine_semantic_fields(house_data)
            
            if not semantic_text.strip():
                logger.warning("语义文本为空，返回零向量")
                return [0.0] * 1024
            
            # 生成向量
            vector, token_count = self.embedding_model.encode_queries(semantic_text)
            logger.debug(f"为文本生成向量，长度: {len(vector)}, tokens: {token_count}")
            
            return vector.tolist()
            
        except Exception as e:
            logger.error(f"向量化失败: {e}")
            return [0.0] * 1024
    
    def _combine_semantic_fields(self, house_data: Dict[str, Any]) -> str:
        """仅使用 semantic_str 作为语义文本，不做任何兜底拼接。"""
        return str(house_data.get('semantic_str') or '').strip()
    
    def create_query_vector(self, query_text: str) -> Optional[List[float]]:
        """
        为用户查询创建向量
        
        Args:
            query_text: 查询文本
            
        Returns:
            查询向量列表
        """
        if not self.embedding_model:
            logger.warning("嵌入模型未初始化，返回零向量")
            return [0.0] * 1024
            
        try:
            if not query_text.strip():
                return [0.0] * 1024
                
            vector, token_count = self.embedding_model.encode_queries(query_text)
            logger.debug(f"为查询生成向量，长度: {len(vector)}, tokens: {token_count}")
            
            return vector.tolist()
            
        except Exception as e:
            logger.error(f"查询向量化失败: {e}")
            return [0.0] * 1024


# 全局向量化工具实例
_vectorization_utils = None

def get_vectorization_utils() -> VectorizationUtils:
    """获取向量化工具实例"""
    global _vectorization_utils
    if _vectorization_utils is None:
        _vectorization_utils = VectorizationUtils()
    return _vectorization_utils