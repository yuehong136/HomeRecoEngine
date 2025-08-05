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
        """
        合并语义字段为单一文本
        
        Args:
            house_data: 房源数据字典
            
        Returns:
            合并后的语义文本
        """
        semantic_fields = [
            ('xqtd', '小区特点'),
            ('xqmd', '小区卖点'), 
            ('xq', '学区'),
            ('xqph', '小区偏好'),
            ('zb', '周边环境'),
            ('fyb', '房源标签'),
            ('fyhx', '房源户型')
        ]
        
        text_parts = []
        
        # 添加基本信息作为语义背景
        if house_data.get('xqmc'):
            if isinstance(house_data['xqmc'], list):
                text_parts.append(f"小区：{', '.join(house_data['xqmc'])}")
            else:
                text_parts.append(f"小区：{house_data['xqmc']}")
        
        if house_data.get('qy'):
            text_parts.append(f"区域：{house_data['qy']}")
            
        if house_data.get('mj'):
            # 处理面积字段，提取数值
            area_text = str(house_data['mj'])
            if '平方米' in area_text:
                area_text = area_text.replace('平方米', '')
            text_parts.append(f"面积：{area_text}平方米")
        
        # 添加语义字段
        for field_key, field_name in semantic_fields:
            value = house_data.get(field_key)
            if value and str(value).strip() and str(value) != 'nan':
                text_parts.append(f"{field_name}：{value}")
        
        # 添加用户查询文本（如果有）
        if house_data.get('user_query_text'):
            text_parts.append(f"用户需求：{house_data['user_query_text']}")
        
        combined_text = "。".join(text_parts)
        logger.debug(f"合并语义文本: {combined_text[:200]}...")
        
        return combined_text
    
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