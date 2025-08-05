"""
房源推荐服务
用于处理房源数据的向量化存储、检索和推荐
"""
import logging
import math
from typing import Dict, List, Optional, Any, Union
from pymilvus import MilvusClient, DataType
from pymilvus.milvus_client.index import IndexParams
from api.utils.vectorization_utils import get_vectorization_utils

logger = logging.getLogger(__name__)


class HouseRecoService:
    """房源推荐服务类"""
    
    # 集合名称
    COLLECTION_NAME = "house_recommendation"
    
    # 向量维度 (BGE-large-zh-v1.5模型维度为1024)
    VECTOR_DIM = 1024
    
    def __init__(self, milvus_client: MilvusClient):
        """
        初始化房源推荐服务
        
        Args:
            milvus_client: Milvus客户端实例
        """
        self.client = milvus_client
        self.vectorization_utils = get_vectorization_utils()
        
    def create_collection_schema(self):
        """创建房源推荐集合的schema"""
        
        # 创建schema
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
            description="房源推荐集合"
        )
        
        # 主键字段
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
            description="房源主键ID"
        )
        
        # ========== 过滤字段 ==========
        
        # 小区名称（多个）- 支持数组查询
        schema.add_field(
            field_name="xqmc",  # 小区名称
            datatype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=10,  # 最多10个小区名称
            max_length=100,
            description="小区名称列表"
        )
        
        # 区域
        schema.add_field(
            field_name="qy",  # 区域
            datatype=DataType.VARCHAR,
            max_length=50,
            description="区域"
        )
        
        # 地址
        schema.add_field(
            field_name="dz",  # 地址
            datatype=DataType.VARCHAR,
            max_length=200,
            description="详细地址"
        )
        
        # 经度
        schema.add_field(
            field_name="jd",  # 经度
            datatype=DataType.DOUBLE,
            description="经度"
        )
        
        # 纬度
        schema.add_field(
            field_name="wd",  # 纬度
            datatype=DataType.DOUBLE,
            description="纬度"
        )
        
        # 产权年限
        schema.add_field(
            field_name="cqnx",  # 产权年限
            datatype=DataType.INT64,
            description="产权年限"
        )
        
        # 绿化率 (存储为百分比数值，如41.5表示41.5%)
        schema.add_field(
            field_name="lhl",  # 绿化率
            datatype=DataType.DOUBLE,
            description="绿化率百分比"
        )
        
        # 容积率
        schema.add_field(
            field_name="rjl",  # 容积率
            datatype=DataType.DOUBLE,
            description="容积率"
        )
        
        # 装修风格
        schema.add_field(
            field_name="zxfg",  # 装修风格
            datatype=DataType.VARCHAR,
            max_length=50,
            description="装修风格"
        )
        
        # 装修情况
        schema.add_field(
            field_name="zxqk",  # 装修情况
            datatype=DataType.VARCHAR,
            max_length=50,
            description="装修情况"
        )
        
        # 水电情况
        schema.add_field(
            field_name="sd",  # 水电
            datatype=DataType.VARCHAR,
            max_length=100,
            description="水电情况"
        )
        
        # 有无电梯
        schema.add_field(
            field_name="ywdt",  # 有无电梯
            datatype=DataType.VARCHAR,
            max_length=10,
            description="有无电梯"
        )
        
        # 有无车位
        schema.add_field(
            field_name="ywcw",  # 有无车位
            datatype=DataType.VARCHAR,
            max_length=10,
            description="有无车位"
        )
        
        # 面积 (单位：平方米)
        schema.add_field(
            field_name="mj",  # 面积
            datatype=DataType.DOUBLE,
            description="面积平方米"
        )
        
        # 朝向
        schema.add_field(
            field_name="cx",  # 朝向
            datatype=DataType.VARCHAR,
            max_length=20,
            description="朝向"
        )
        
        # 单价 (元/平方米)
        schema.add_field(
            field_name="dj",  # 单价
            datatype=DataType.DOUBLE,
            description="单价元每平方米"
        )
        
        # 总价 (万元)
        schema.add_field(
            field_name="zj",  # 总价
            datatype=DataType.DOUBLE,
            description="总价万元"
        )
        
        # 物业费 (元/m²/月)
        schema.add_field(
            field_name="wyf",  # 物业费
            datatype=DataType.DOUBLE,
            description="物业费元每平方米每月"
        )
        
        # 房屋年限
        schema.add_field(
            field_name="fwnx",  # 房屋年限
            datatype=DataType.INT64,
            description="房屋年限"
        )
        
        # 楼层情况
        schema.add_field(
            field_name="lc",  # 楼层
            datatype=DataType.VARCHAR,
            max_length=20,
            description="楼层情况"
        )
        
        # ========== 语义字段 (存储原始文本，用于显示) ==========
        
        # 小区特点
        schema.add_field(
            field_name="xqtd",  # 小区特点
            datatype=DataType.VARCHAR,
            max_length=500,
            description="小区特点"
        )
        
        # 小区卖点
        schema.add_field(
            field_name="xqmd",  # 小区卖点
            datatype=DataType.VARCHAR,
            max_length=500,
            description="小区卖点"
        )
        
        # 学区
        schema.add_field(
            field_name="xq",  # 学区
            datatype=DataType.VARCHAR,
            max_length=200,
            description="学区"
        )
        
        # 小区偏好
        schema.add_field(
            field_name="xqph",  # 小区偏好
            datatype=DataType.VARCHAR,
            max_length=300,
            description="小区偏好"
        )
        
        # 周边环境
        schema.add_field(
            field_name="zb",  # 周边
            datatype=DataType.VARCHAR,
            max_length=2000,
            description="周边环境"
        )
        
        # 房源标签
        schema.add_field(
            field_name="fyb",  # 房源标签
            datatype=DataType.VARCHAR,
            max_length=200,
            description="房源标签"
        )
        
        # 房源户型
        schema.add_field(
            field_name="fyhx",  # 房源户型
            datatype=DataType.VARCHAR,
            max_length=50,
            description="房源户型"
        )
        
        # 用户查询文本 (用于个性化推荐)
        schema.add_field(
            field_name="user_query_text",
            datatype=DataType.VARCHAR,
            max_length=500,
            description="用户查询文本"
        )
        
        # ========== 向量字段 (单一语义向量) ==========
        
        # 综合语义向量 - 合并所有语义信息
        schema.add_field(
            field_name="semantic_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.VECTOR_DIM,
            description="综合语义向量"
        )
        
        return schema
    
    def create_collection(self) -> bool:
        """
        创建房源推荐集合
        
        Returns:
            创建是否成功
        """
        try:
            # 检查集合是否已存在
            if self.client.has_collection(self.COLLECTION_NAME):
                logger.info(f"集合 {self.COLLECTION_NAME} 已存在")
                return True
            
            # 创建schema
            schema = self.create_collection_schema()
            
            # 创建索引参数
            index_params = IndexParams()
            
            # 为向量字段创建索引
            index_params.add_index(
                field_name="semantic_vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 500}
            )
            
            # 创建集合
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                schema=schema,
                index_params=index_params
            )
            
            logger.info(f"成功创建集合 {self.COLLECTION_NAME}")
            return True
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
    
    def insert_house_data(self, house_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> bool:
        """
        插入房源数据
        
        Args:
            house_data: 单个房源数据字典或房源数据列表
            
        Returns:
            插入是否成功
        """
        try:
            # 统一处理为列表格式
            if isinstance(house_data, dict):
                house_data_list = [house_data]
            else:
                house_data_list = house_data
            
            # 数据预处理和向量化
            processed_data = []
            for house in house_data_list:
                processed_house = self._preprocess_house_data(house)
                if processed_house:
                    processed_data.append(processed_house)
            
            if not processed_data:
                logger.warning("没有有效的房源数据需要插入")
                return False
            
            # 批量插入
            self.client.insert(
                collection_name=self.COLLECTION_NAME,
                data=processed_data
            )
            
            logger.info(f"成功插入 {len(processed_data)} 条房源数据")
            return True
            
        except Exception as e:
            logger.error(f"插入房源数据失败: {e}")
            return False
    
    def _preprocess_house_data(self, house_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        预处理单个房源数据
        
        Args:
            house_data: 原始房源数据
            
        Returns:
            预处理后的房源数据，失败返回None
        """
        try:
            processed_data = {}
            
            # 处理主键（处理numpy类型）
            id_raw = house_data.get('id', 0)
            processed_data['id'] = int(id_raw.item() if hasattr(id_raw, 'item') else id_raw)
            
            # 处理小区名称（可能包含多个，用逗号分隔）
            xqmc_raw = house_data.get('xqmc', '')
            if isinstance(xqmc_raw, str) and ',' in xqmc_raw:
                processed_data['xqmc'] = [name.strip() for name in xqmc_raw.split(',')]
            elif isinstance(xqmc_raw, list):
                processed_data['xqmc'] = xqmc_raw
            else:
                processed_data['xqmc'] = [str(xqmc_raw)] if xqmc_raw else []
            
            # 处理基本字符串字段
            string_fields = ['qy', 'dz', 'zxfg', 'zxqk', 'sd', 'ywdt', 'ywcw', 'cx', 'lc', 
                           'xqtd', 'xqmd', 'xq', 'xqph', 'zb', 'fyb', 'fyhx', 'user_query_text']
            for field in string_fields:
                value = house_data.get(field, '')
                processed_data[field] = str(value) if value and str(value) != 'nan' else ''
            
            # 处理数值字段（安全转换，处理None值和numpy类型）
            jd_raw = house_data.get('jd')
            if jd_raw is not None:
                processed_data['jd'] = float(jd_raw.item() if hasattr(jd_raw, 'item') else jd_raw)
            else:
                processed_data['jd'] = 0.0
            
            wd_raw = house_data.get('wd')
            if wd_raw is not None:
                processed_data['wd'] = float(wd_raw.item() if hasattr(wd_raw, 'item') else wd_raw)
            else:
                processed_data['wd'] = 0.0
            
            cqnx_raw = house_data.get('cqnx')
            if cqnx_raw is not None:
                processed_data['cqnx'] = int(cqnx_raw.item() if hasattr(cqnx_raw, 'item') else cqnx_raw)
            else:
                processed_data['cqnx'] = 0
            
            fwnx_raw = house_data.get('fwnx')
            if fwnx_raw is not None:
                processed_data['fwnx'] = int(fwnx_raw.item() if hasattr(fwnx_raw, 'item') else fwnx_raw)
            else:
                processed_data['fwnx'] = 0
            
            # 处理绿化率（去除%符号）
            lhl_raw = house_data.get('lhl')
            if lhl_raw is not None and isinstance(lhl_raw, str) and '%' in lhl_raw:
                processed_data['lhl'] = float(lhl_raw.replace('%', ''))
            elif lhl_raw is not None:
                processed_data['lhl'] = float(lhl_raw.item() if hasattr(lhl_raw, 'item') else lhl_raw)
            else:
                processed_data['lhl'] = 0.0
            
            rjl_raw = house_data.get('rjl')
            if rjl_raw is not None:
                processed_data['rjl'] = float(rjl_raw.item() if hasattr(rjl_raw, 'item') else rjl_raw)
            else:
                processed_data['rjl'] = 0.0
            
            # 处理面积（去除"平方米"文字）
            mj_raw = house_data.get('mj')
            if mj_raw is not None and isinstance(mj_raw, str) and '平方米' in mj_raw:
                mj_clean = mj_raw.replace('平方米', '').strip()
                processed_data['mj'] = float(mj_clean) if mj_clean else 0.0
            elif mj_raw is not None:
                processed_data['mj'] = float(mj_raw.item() if hasattr(mj_raw, 'item') else mj_raw)
            else:
                processed_data['mj'] = 0.0
            
            # 处理价格字段（去除单位文字）
            dj_raw = house_data.get('dj')
            if dj_raw is not None and isinstance(dj_raw, str):
                dj_clean = dj_raw.replace('元/平方米', '').replace('元', '').strip()
                processed_data['dj'] = float(dj_clean) if dj_clean else 0.0
            elif dj_raw is not None:
                processed_data['dj'] = float(dj_raw.item() if hasattr(dj_raw, 'item') else dj_raw)
            else:
                processed_data['dj'] = 0.0
                
            zj_raw = house_data.get('zj')
            if zj_raw is not None and isinstance(zj_raw, str):
                zj_clean = zj_raw.replace('万元', '').strip()
                processed_data['zj'] = float(zj_clean) if zj_clean else 0.0
            elif zj_raw is not None:
                processed_data['zj'] = float(zj_raw.item() if hasattr(zj_raw, 'item') else zj_raw)
            else:
                processed_data['zj'] = 0.0
                
            wyf_raw = house_data.get('wyf')
            if wyf_raw is not None and isinstance(wyf_raw, str):
                wyf_clean = wyf_raw.replace('元/m²/月', '').replace('元', '').strip()
                processed_data['wyf'] = float(wyf_clean) if wyf_clean else 0.0
            elif wyf_raw is not None:
                processed_data['wyf'] = float(wyf_raw.item() if hasattr(wyf_raw, 'item') else wyf_raw)
            else:
                processed_data['wyf'] = 0.0
            
            # 生成语义向量
            semantic_vector = self.vectorization_utils.create_semantic_vector(processed_data)
            processed_data['semantic_vector'] = semantic_vector
            
            return processed_data
            
        except Exception as e:
            logger.error(f"预处理房源数据失败: {e}, 数据: {house_data}")
            return None
    
    def _serialize_house_data(self, house_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        序列化房源数据，处理protobuf类型
        
        Args:
            house_data: 原始房源数据
            
        Returns:
            序列化后的房源数据
        """
        try:
            serialized_data = {}
            
            for key, value in house_data.items():
                # 跳过向量字段，避免返回给客户端
                if key == 'semantic_vector':
                    continue
                    
                if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                    # 处理数组类型（如小区名称列表、protobuf RepeatedScalarContainer）
                    try:
                        serialized_data[key] = list(value)
                    except (TypeError, ValueError):
                        serialized_data[key] = str(value)
                elif hasattr(value, 'item'):
                    # 处理numpy类型（如numpy.int64, numpy.float64等）
                    serialized_data[key] = value.item()
                elif str(type(value)).startswith('<class \'numpy.'):
                    # 处理其他numpy类型
                    try:
                        serialized_data[key] = value.item() if hasattr(value, 'item') else float(value)
                    except:
                        serialized_data[key] = str(value)
                elif str(type(value)).startswith('<class \'google._upb._message.'):
                    # 特殊处理protobuf类型
                    try:
                        if hasattr(value, '__iter__'):
                            serialized_data[key] = list(value)
                        else:
                            serialized_data[key] = str(value)
                    except:
                        serialized_data[key] = str(value)
                else:
                    # 其他类型直接使用
                    serialized_data[key] = value
            
            return serialized_data
            
        except Exception as e:
            logger.warning(f"数据序列化失败: {e}, 使用安全序列化")
            # 如果序列化失败，使用最安全的方法
            return self._safe_serialize(house_data)
    
    def _safe_serialize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        安全序列化，确保所有数据都能被JSON序列化
        """
        safe_data = {}
        
        for key, value in data.items():
            # 跳过向量字段
            if key == 'semantic_vector':
                continue
                
            try:
                # 测试是否可以JSON序列化
                import json
                json.dumps(value)
                safe_data[key] = value
            except (TypeError, ValueError):
                # 如果不能序列化，转换为字符串
                if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                    try:
                        safe_data[key] = list(value)
                    except:
                        safe_data[key] = str(value)
                else:
                    safe_data[key] = str(value)
        
        return safe_data
    
    def search_houses(self, search_params: Dict[str, Any], limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """
        搜索房源
        
        Args:
            search_params: 搜索参数字典
            limit: 返回结果数量限制
            offset: 结果偏移量
            
        Returns:
            搜索结果列表
        """
        try:
            # 构建过滤表达式
            filter_expr = self._build_filter_expression(search_params)
            
            # 检查是否有语义查询
            if search_params.get('user_query_text'):
                # 执行语义搜索
                query_vector = self.vectorization_utils.create_query_vector(search_params['user_query_text'])
                
                search_results = self.client.search(
                    collection_name=self.COLLECTION_NAME,
                    data=[query_vector],
                    filter=filter_expr if filter_expr else None,
                    limit=limit + offset,
                    output_fields=["*"]
                )
                
                # 处理结果并应用offset
                results = []
                if search_results and len(search_results) > 0:
                    for hit in search_results[0][offset:offset + limit]:
                        house_data = hit['entity']
                        house_data['similarity_score'] = hit['distance']
                        # 序列化数据以避免protobuf类型问题
                        serialized_data = self._serialize_house_data(house_data)
                        results.append(serialized_data)
                        
            else:
                # 纯过滤查询
                if filter_expr:
                    query_results = self.client.query(
                        collection_name=self.COLLECTION_NAME,
                        filter=filter_expr,
                        output_fields=["*"],
                        limit=limit,
                        offset=offset
                    )
                    # 序列化查询结果
                    results = [self._serialize_house_data(house) for house in query_results]
                else:
                    # 如果没有任何过滤条件，返回空结果
                    results = []
            
            # 如果是圆形区域搜索，需要进行精确距离计算和过滤
            if search_params.get('location') and self._is_circle_search(search_params['location']):
                results = self._apply_circle_distance_filter(results, search_params['location'], limit)
            
            logger.info(f"搜索完成，返回 {len(results)} 条结果")
            return results
            
        except Exception as e:
            logger.error(f"搜索房源失败: {e}")
            return []
    
    def _build_filter_expression(self, search_params: Dict[str, Any]) -> Optional[str]:
        """
        构建过滤表达式
        
        Args:
            search_params: 搜索参数字典
            
        Returns:
            过滤表达式字符串，无过滤条件时返回None
        """
        conditions = []
        
        # 小区名称过滤（数组字段）
        if search_params.get('xqmc'):
            xqmc_list = search_params['xqmc']
            if isinstance(xqmc_list, list) and xqmc_list:
                # 使用数组包含查询
                xqmc_conditions = []
                for name in xqmc_list:
                    xqmc_conditions.append(f'array_contains(xqmc, "{name}")')
                conditions.append(f"({' OR '.join(xqmc_conditions)})")
        
        # 基本字符串字段过滤
        string_fields = {
            'qy': '区域',
            'lc': '楼层',
            'zxqk': '装修情况',
            'cx': '朝向',
            'ywdt': '有无电梯',
            'ywcw': '有无车位'
        }
        
        for field, _ in string_fields.items():
            if search_params.get(field):
                conditions.append(f'{field} == "{search_params[field]}"')
        
        # 价格范围过滤
        if search_params.get('price_range'):
            price_range = search_params['price_range']
            if price_range.get('min_price') is not None:
                conditions.append(f'zj >= {price_range["min_price"]}')
            if price_range.get('max_price') is not None:
                conditions.append(f'zj <= {price_range["max_price"]}')
        
        # 面积范围过滤
        if search_params.get('area_range'):
            area_range = search_params['area_range']
            if area_range.get('min_area') is not None:
                conditions.append(f'mj >= {area_range["min_area"]}')
            if area_range.get('max_area') is not None:
                conditions.append(f'mj <= {area_range["max_area"]}')
        
        # 统一的地理位置过滤
        if search_params.get('location'):
            location_filter = self._build_location_filter(search_params['location'])
            if location_filter:
                conditions.append(location_filter)
        
        # 产权年限范围过滤
        if search_params.get('cqnx_range'):
            cqnx_range = search_params['cqnx_range']
            if cqnx_range.get('min_years') is not None:
                conditions.append(f'cqnx >= {cqnx_range["min_years"]}')
            if cqnx_range.get('max_years') is not None:
                conditions.append(f'cqnx <= {cqnx_range["max_years"]}')
        
        # 组合所有条件
        if conditions:
            return ' AND '.join(conditions)
        else:
            return None
    
    def _build_location_filter(self, location: Dict[str, Any]) -> Optional[str]:
        """
        构建统一的地理位置过滤表达式
        支持圆形区域和矩形区域两种模式
        
        Args:
            location: 地理位置参数字典，支持两种格式：
                - 圆形区域: {'center_longitude': 经度, 'center_latitude': 纬度, 'radius_km': 半径}
                - 矩形区域: {'min_longitude': 最小经度, 'max_longitude': 最大经度, 'min_latitude': 最小纬度, 'max_latitude': 最大纬度}
        
        Returns:
            过滤表达式字符串，参数无效时返回None
        """
        try:
            # 检查是否为圆形区域搜索
            if all(key in location for key in ['center_longitude', 'center_latitude', 'radius_km']):
                return self._build_circle_filter(location)
            
            # 检查是否为矩形区域搜索
            elif any(key in location for key in ['min_longitude', 'max_longitude', 'min_latitude', 'max_latitude']):
                return self._build_rectangle_filter(location)
            
            # 兼容旧的字段名
            elif any(key in location for key in ['min_jd', 'max_jd', 'min_wd', 'max_wd']):
                return self._build_rectangle_filter_legacy(location)
            
            else:
                logger.warning(f"无效的地理位置参数格式: {location}")
                return None
                
        except Exception as e:
            logger.error(f"构建地理位置过滤表达式失败: {e}")
            return None
    
    def _build_rectangle_filter(self, location: Dict[str, Any]) -> Optional[str]:
        """
        构建矩形区域过滤表达式
        
        Args:
            location: 包含矩形边界的字典
        
        Returns:
            过滤表达式字符串
        """
        conditions = []
        
        if location.get('min_longitude') is not None:
            conditions.append(f'jd >= {location["min_longitude"]}')
        if location.get('max_longitude') is not None:
            conditions.append(f'jd <= {location["max_longitude"]}')
        if location.get('min_latitude') is not None:
            conditions.append(f'wd >= {location["min_latitude"]}')
        if location.get('max_latitude') is not None:
            conditions.append(f'wd <= {location["max_latitude"]}')
        
        return f"({' AND '.join(conditions)})" if conditions else None
    
    def _build_rectangle_filter_legacy(self, location: Dict[str, Any]) -> Optional[str]:
        """
        构建矩形区域过滤表达式（兼容旧字段名）
        
        Args:
            location: 包含矩形边界的字典（使用旧字段名）
        
        Returns:
            过滤表达式字符串
        """
        conditions = []
        
        if location.get('min_jd') is not None:
            conditions.append(f'jd >= {location["min_jd"]}')
        if location.get('max_jd') is not None:
            conditions.append(f'jd <= {location["max_jd"]}')
        if location.get('min_wd') is not None:
            conditions.append(f'wd >= {location["min_wd"]}')
        if location.get('max_wd') is not None:
            conditions.append(f'wd <= {location["max_wd"]}')
        
        return f"({' AND '.join(conditions)})" if conditions else None

    def _build_circle_filter(self, location_circle: Dict[str, Any]) -> Optional[str]:
        """
        构建圆形区域过滤表达式
        使用方形边界框近似圆形区域，提高查询效率
        
        Args:
            location_circle: 包含中心坐标和半径的字典
                - center_longitude: 中心经度
                - center_latitude: 中心纬度  
                - radius_km: 半径（公里）
        
        Returns:
            过滤表达式字符串，参数无效时返回None
        """
        try:
            center_lng = location_circle.get('center_longitude')
            center_lat = location_circle.get('center_latitude')
            radius_km = location_circle.get('radius_km')
            
            if center_lng is None or center_lat is None or radius_km is None:
                return None
            
            # 计算边界框坐标（使用度数近似）
            # 1度纬度约等于111km，经度在不同纬度下有差异
            lat_degree_km = 111.32  # 纬度每度约111.32km
            lng_degree_km = 111.32 * math.cos(math.radians(center_lat))  # 经度每度随纬度变化
            
            # 计算边界框边界
            lat_delta = radius_km / lat_degree_km
            lng_delta = radius_km / lng_degree_km
            
            min_lat = center_lat - lat_delta
            max_lat = center_lat + lat_delta
            min_lng = center_lng - lng_delta
            max_lng = center_lng + lng_delta
            
            # 构建边界框过滤条件
            conditions = [
                f'wd >= {min_lat}',
                f'wd <= {max_lat}',
                f'jd >= {min_lng}',
                f'jd <= {max_lng}'
            ]
            
            return f"({' AND '.join(conditions)})"
            
        except Exception as e:
            logger.error(f"构建圆形区域过滤表达式失败: {e}")
            return None
    
    def _is_circle_search(self, location: Dict[str, Any]) -> bool:
        """
        检查是否为圆形区域搜索
        
        Args:
            location: 地理位置参数
            
        Returns:
            是否为圆形区域搜索
        """
        return all(key in location for key in ['center_longitude', 'center_latitude', 'radius_km'])
    
    def _apply_circle_distance_filter(self, results: List[Dict[str, Any]], 
                                    location: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """
        对圆形区域搜索结果应用精确距离计算和过滤
        
        Args:
            results: 搜索结果列表
            location: 圆形区域参数
            limit: 结果数量限制
            
        Returns:
            过滤并排序后的结果列表
        """
        try:
            center_lng = location['center_longitude']
            center_lat = location['center_latitude']
            radius_km = location['radius_km']
            
            # 计算精确距离并过滤
            filtered_results = []
            for house in results:
                if house.get('jd') is not None and house.get('wd') is not None:
                    distance = self.calculate_distance_km(
                        center_lat, center_lng,
                        house['wd'], house['jd']
                    )
                    
                    # 只保留在半径范围内的房源
                    if distance <= radius_km:
                        house['distance_km'] = round(distance, 2)
                        filtered_results.append(house)
            
            # 按距离排序
            filtered_results.sort(key=lambda x: x.get('distance_km', float('inf')))
            
            # 限制返回数量
            return filtered_results[:limit]
            
        except Exception as e:
            logger.error(f"应用圆形距离过滤失败: {e}")
            return results[:limit]
    
    def hybrid_search(self, semantic_query: str, filter_params: Dict[str, Any] = None, 
                     semantic_weight: float = 0.7, limit: int = 10) -> List[Dict[str, Any]]:
        """
        混合搜索（语义搜索 + 过滤）
        
        Args:
            semantic_query: 语义查询文本
            filter_params: 过滤参数
            semantic_weight: 语义搜索权重 (当前版本暂未使用，预留用于未来扩展)
            limit: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        try:
            # 合并搜索参数
            search_params = filter_params.copy() if filter_params else {}
            search_params['user_query_text'] = semantic_query
            
            # 注：semantic_weight参数预留用于未来的混合搜索权重调整功能
            # 当前版本直接使用语义搜索
            
            # 执行搜索
            results = self.search_houses(search_params, limit=limit)
            
            return results
            
        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
            return []
    
    def get_house_by_id(self, house_id: int) -> Optional[Dict[str, Any]]:
        """
        根据ID获取房源详情
        
        Args:
            house_id: 房源ID
            
        Returns:
            房源数据字典，未找到返回None
        """
        try:
            results = self.client.query(
                collection_name=self.COLLECTION_NAME,
                filter=f'id == {house_id}',
                output_fields=["*"],
                limit=1
            )
            
            if results:
                return self._serialize_house_data(results[0])
            else:
                return None
                
        except Exception as e:
            logger.error(f"获取房源详情失败: {e}")
            return None
    
    def delete_house(self, house_id: int) -> bool:
        """
        删除房源
        
        Args:
            house_id: 房源ID
            
        Returns:
            删除是否成功
        """
        try:
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                filter=f'id == {house_id}'
            )
            
            logger.info(f"成功删除房源 {house_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除房源失败: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            统计信息字典
        """
        try:
            # 获取集合信息
            if not self.client.has_collection(self.COLLECTION_NAME):
                return {"error": "集合不存在"}
            
            # 获取记录数量
            stats = self.client.get_collection_stats(self.COLLECTION_NAME)
            
            return {
                "collection_name": self.COLLECTION_NAME,
                "row_count": stats.get("row_count", 0),
                "created": True
            }
            
        except Exception as e:
            logger.error(f"获取集合统计信息失败: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def calculate_distance_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """
        使用Haversine公式计算两点间的球面距离
        
        Args:
            lat1, lng1: 第一个点的纬度和经度
            lat2, lng2: 第二个点的纬度和经度
            
        Returns:
            距离（公里）
        """
        # 地球半径（公里）
        R = 6371.0
        
        # 转换为弧度
        lat1_rad = math.radians(lat1)
        lng1_rad = math.radians(lng1)
        lat2_rad = math.radians(lat2)
        lng2_rad = math.radians(lng2)
        
        # 差值
        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad
        
        # Haversine公式
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        # 距离
        distance = R * c
        return distance
    
    def search_houses_by_location(self, center_longitude: float, center_latitude: float, 
                                 radius_km: float, additional_filters: Dict[str, Any] = None,
                                 limit: int = 10, sort_by_distance: bool = True) -> List[Dict[str, Any]]:
        """
        基于中心坐标和半径搜索房源
        
        Args:
            center_longitude: 中心经度
            center_latitude: 中心纬度
            radius_km: 搜索半径（公里）
            additional_filters: 额外的过滤条件
            limit: 返回结果数量限制
            sort_by_distance: 是否按距离排序
            
        Returns:
            搜索结果列表，每个结果包含distance_km字段
        """
        try:
            # 构建搜索参数
            search_params = additional_filters.copy() if additional_filters else {}
            search_params['location_circle'] = {
                'center_longitude': center_longitude,
                'center_latitude': center_latitude,
                'radius_km': radius_km
            }
            
            # 使用更大的limit进行初步筛选，然后精确计算距离
            initial_limit = min(limit * 3, 100)  # 获取更多候选结果
            
            # 执行搜索
            results = self.search_houses(search_params, limit=initial_limit)
            
            # 计算精确距离并过滤
            filtered_results = []
            for house in results:
                if house.get('jd') is not None and house.get('wd') is not None:
                    distance = self.calculate_distance_km(
                        center_latitude, center_longitude,
                        house['wd'], house['jd']
                    )
                    
                    # 只保留在半径范围内的房源
                    if distance <= radius_km:
                        house['distance_km'] = round(distance, 2)
                        filtered_results.append(house)
            
            # 按距离排序
            if sort_by_distance:
                filtered_results.sort(key=lambda x: x.get('distance_km', float('inf')))
            
            # 限制返回数量
            final_results = filtered_results[:limit]
            
            logger.info(f"在坐标({center_latitude}, {center_longitude})半径{radius_km}km内找到 {len(final_results)} 条房源")
            return final_results
            
        except Exception as e:
            logger.error(f"基于地理位置搜索房源失败: {e}")
            return []
    
    def search_houses_by_location_with_semantic(self, center_longitude: float, center_latitude: float,
                                               radius_km: float, semantic_query: str,
                                               additional_filters: Dict[str, Any] = None,
                                               limit: int = 10) -> List[Dict[str, Any]]:
        """
        基于地理位置和语义查询的混合搜索
        
        Args:
            center_longitude: 中心经度
            center_latitude: 中心纬度
            radius_km: 搜索半径（公里）
            semantic_query: 语义查询文本
            additional_filters: 额外的过滤条件
            limit: 返回结果数量限制
            
        Returns:
            搜索结果列表，包含distance_km和similarity_score字段
        """
        try:
            # 构建搜索参数
            search_params = additional_filters.copy() if additional_filters else {}
            search_params['location_circle'] = {
                'center_longitude': center_longitude,
                'center_latitude': center_latitude,
                'radius_km': radius_km
            }
            search_params['user_query_text'] = semantic_query
            
            # 获取更多候选结果
            initial_limit = min(limit * 3, 100)
            
            # 执行混合搜索
            results = self.search_houses(search_params, limit=initial_limit)
            
            # 计算精确距离并过滤
            filtered_results = []
            for house in results:
                if house.get('jd') is not None and house.get('wd') is not None:
                    distance = self.calculate_distance_km(
                        center_latitude, center_longitude,
                        house['wd'], house['jd']
                    )
                    
                    # 只保留在半径范围内的房源
                    if distance <= radius_km:
                        house['distance_km'] = round(distance, 2)
                        filtered_results.append(house)
            
            # 按语义相似度排序（如果有的话），否则按距离排序
            if filtered_results and 'similarity_score' in filtered_results[0]:
                filtered_results.sort(key=lambda x: x.get('similarity_score', 1.0))
            else:
                filtered_results.sort(key=lambda x: x.get('distance_km', float('inf')))
            
            # 限制返回数量
            final_results = filtered_results[:limit]
            
            logger.info(f"在坐标({center_latitude}, {center_longitude})半径{radius_km}km内语义搜索找到 {len(final_results)} 条房源")
            return final_results
            
        except Exception as e:
            logger.error(f"基于地理位置和语义的混合搜索失败: {e}")
            return []