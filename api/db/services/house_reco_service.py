"""
房源推荐服务
用于处理房源数据的向量化存储、检索和推荐
"""
import logging
import math
from typing import Dict, List, Optional, Any, Union
from pymilvus import MilvusClient, DataType, Function, FunctionType, WeightedRanker, AnnSearchRequest
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

    def _build_milvus_search_params(self, confidence: Optional[float]) -> Dict[str, Any]:
        """构建 Milvus search_params，将 confidence 作为底层阈值传入。
        说明：具体可用参数取决于索引/度量类型；此处设置 metric_type，并在提供置信度时附带阈值。
        """
        params: Dict[str, Any] = {"metric_type": "COSINE"}
        try:
            if confidence is not None:
                params["params"] = {"radius": float(confidence)}
        except Exception:
            pass
        return params
        
    def create_collection_schema(self):
        """创建房源推荐集合的schema"""
        analyzer_params = {
            "type": "chinese",
        }
        # 创建schema
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
            description="房源推荐集合"
        )
        
        # 主键字段（字符串主键）
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            max_length=128,
            is_primary=True,
            description="房源主键ID（字符串）"
        )
        
        # 房源分类（1-新房，2-二手房，3-出租房）
        schema.add_field(
            field_name="category",
            datatype=DataType.INT64,
            description="房源分类：1-新房，2-二手房，3-出租房"
        )

        # 新字段：基础信息
        schema.add_field(field_name="name", datatype=DataType.VARCHAR, max_length=200, description="房源名称")
        schema.add_field(field_name="region", datatype=DataType.VARCHAR, max_length=100, description="区域")
        schema.add_field(field_name="address", datatype=DataType.VARCHAR, max_length=300, description="地址")

        # 新字段：范围数值
        schema.add_field(field_name="min_area", datatype=DataType.DOUBLE, description="最小面积")
        schema.add_field(field_name="max_area", datatype=DataType.DOUBLE, description="最大面积")
        schema.add_field(field_name="min_unit_price", datatype=DataType.DOUBLE, description="最小单价")
        schema.add_field(field_name="max_unit_price", datatype=DataType.DOUBLE, description="最大单价")
        schema.add_field(field_name="min_total_price", datatype=DataType.DOUBLE, description="最小总价")
        schema.add_field(field_name="max_total_price", datatype=DataType.DOUBLE, description="最大总价")
        schema.add_field(field_name="rent", datatype=DataType.DOUBLE, description="租金")

        # 新字段：地理坐标
        schema.add_field(field_name="longitude", datatype=DataType.DOUBLE, description="经度")
        schema.add_field(field_name="latitude", datatype=DataType.DOUBLE, description="纬度")

        # 新字段：更多属性
        schema.add_field(field_name="type", datatype=DataType.VARCHAR, max_length=50, description="房源类型")
        schema.add_field(field_name="year_completion", datatype=DataType.VARCHAR, max_length=50, description="建成年代")
        schema.add_field(field_name="transaction_ownership", datatype=DataType.VARCHAR, max_length=100, description="交易权属")
        schema.add_field(field_name="property_right_duration", datatype=DataType.INT64, description="产权年限")
        schema.add_field(field_name="parking_space_ratio", datatype=DataType.VARCHAR, max_length=50, description="车位比")
        schema.add_field(field_name="management_company", datatype=DataType.VARCHAR, max_length=200, description="物业公司")
        schema.add_field(field_name="management_fee", datatype=DataType.DOUBLE, description="物业费")
        schema.add_field(field_name="developer", datatype=DataType.VARCHAR, max_length=200, description="开发商")
        schema.add_field(field_name="greening_rate", datatype=DataType.DOUBLE, description="绿化率")
        schema.add_field(field_name="plot_ratio", datatype=DataType.DOUBLE, description="容积率")
        schema.add_field(field_name="decoration_style", datatype=DataType.VARCHAR, max_length=100, description="装修风格")
        schema.add_field(field_name="decoration_status", datatype=DataType.VARCHAR, max_length=100, description="装修情况")
        schema.add_field(field_name="water_electricity", datatype=DataType.VARCHAR, max_length=100, description="水电")
        schema.add_field(field_name="has_elevator", datatype=DataType.VARCHAR, max_length=10, description="有无电梯")
        schema.add_field(field_name="has_parking", datatype=DataType.VARCHAR, max_length=10, description="有无车位")
        schema.add_field(field_name="orientation", datatype=DataType.VARCHAR, max_length=50, description="朝向")
        schema.add_field(field_name="building_age", datatype=DataType.VARCHAR, max_length=50, description="房屋年限")
        schema.add_field(field_name="furniture_facilities", datatype=DataType.VARCHAR, max_length=500, description="家具设施")
        schema.add_field(field_name="floor", datatype=DataType.VARCHAR, max_length=50, description="楼层")
        schema.add_field(field_name="rental_mode", datatype=DataType.VARCHAR, max_length=50, description="租赁模式")
        schema.add_field(field_name="payment_method", datatype=DataType.VARCHAR, max_length=50, description="付款方式")
        schema.add_field(field_name="lease_term", datatype=DataType.INT64, description="租期(月)")
        schema.add_field(field_name="preferences_tags", datatype=DataType.VARCHAR, max_length=1000, description="偏好与标签")
        schema.add_field(field_name="cover_url", datatype=DataType.VARCHAR, max_length=500, description="房源封面图，用于前端展示，不用于检索")
        schema.add_field(field_name="semantic_str", datatype=DataType.VARCHAR, max_length=3000, description="语义字符", enable_analyzer=True, analyzer_params=analyzer_params)
        # 注意：删除了旧版字段（如 xqmc/qy/dz/jd/wd 等），仅保留新字段集合
        
        # ========== 向量字段 (单一语义向量) ==========
        
        # 综合语义向量 - 合并所有语义信息
        schema.add_field(
            field_name="semantic_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.VECTOR_DIM,
            description="综合语义向量"
        )
        # 稀疏向量字段（用于BM25）
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR, description="稀疏向量")

        # 创建BM25函数
        bm25_function_name = f"bm25_function"
        bm25_function = Function(
            name=bm25_function_name,
            function_type=FunctionType.BM25,
            input_field_names=["semantic_str"],
            output_field_names="sparse_vector"
        )
        schema.add_function(bm25_function)


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
            
            # 为稠密向量创建索引
            index_params.add_index(
                field_name="semantic_vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 500}
            )

            # 为稀疏向量创建索引
            index_params.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={
                    "inverted_index_algo": "DAAT_WAND",
                    "bm25_k1": 1.5, 
                    "bm25_b": 0.75
                }
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
        """按最新字段规范预处理单个房源数据。"""
        try:
            processed_data: Dict[str, Any] = {}

            def as_str(value: Any) -> str:
                if value is None or str(value).strip() == '' or str(value) == 'nan':
                    return ''
                return str(value)

            def as_int(value: Any) -> int:
                try:
                    if hasattr(value, 'item'):
                        value = value.item()
                    return int(value)
                except Exception:
                    return 0

            def as_float(value: Any) -> float:
                try:
                    if value is None:
                        return 0.0
                    if isinstance(value, str):
                        cleaned = (
                            value.replace('万元', '')
                            .replace('元/m²/月', '')
                            .replace('元/平方米', '')
                            .replace('元/月', '')
                            .replace('元', '')
                            .replace('%', '')
                            .strip()
                        )
                        if cleaned == '':
                            return 0.0
                        return float(cleaned)
                    if hasattr(value, 'item'):
                        value = value.item()
                    return float(value)
                except Exception:
                    return 0.0

            # 必填
            processed_data['id'] = as_str(house_data.get('id'))
            processed_data['category'] = as_int(house_data.get('category'))
            processed_data['name'] = as_str(house_data.get('name'))
            processed_data['region'] = as_str(house_data.get('region'))
            processed_data['address'] = as_str(house_data.get('address'))
            processed_data['longitude'] = as_float(house_data.get('longitude'))
            processed_data['latitude'] = as_float(house_data.get('latitude'))
            processed_data['semantic_str'] = as_str(house_data.get('semantic_str'))

            # 可选字符串
            for key in [
                'type', 'year_completion', 'transaction_ownership', 'parking_space_ratio',
                'management_company', 'developer', 'decoration_style', 'decoration_status',
                'water_electricity', 'has_elevator', 'has_parking', 'orientation', 'building_age',
                'furniture_facilities', 'floor', 'rental_mode', 'payment_method', 'preferences_tags'
            ]:
                processed_data[key] = as_str(house_data.get(key))

            # 数值范围
            for key in [
                'min_area', 'max_area', 'min_unit_price', 'max_unit_price', 'min_total_price',
                'max_total_price', 'rent', 'greening_rate', 'plot_ratio', 'management_fee'
            ]:
                processed_data[key] = as_float(house_data.get(key))

            # 整数
            processed_data['property_right_duration'] = as_int(house_data.get('property_right_duration'))
            processed_data['lease_term'] = as_int(house_data.get('lease_term'))

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
                if key in ('semantic_vector', 'sparse_vector'):
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
            if key in ('semantic_vector', 'sparse_vector'):
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

    def _get_safe_output_fields(self) -> List[str]:
        """返回输出字段白名单，排除向量/稀疏向量等内部字段。"""
        # 可见字段（与schema一致，但排除向量字段）
        return [
            "id","category","name","region","address",
            "min_area","max_area","min_unit_price","max_unit_price",
            "min_total_price","max_total_price","rent",
            "longitude","latitude","type","year_completion","transaction_ownership",
            "property_right_duration","parking_space_ratio","management_company",
            "management_fee","developer","greening_rate","plot_ratio","decoration_style",
            "decoration_status","water_electricity","has_elevator","has_parking",
            "orientation","building_age","furniture_facilities","floor","rental_mode",
            "payment_method","lease_term","preferences_tags","cover_url","semantic_str"
        ]
    
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
            
            retrieval_type = (search_params.get('retrieval_type') or 'vector').lower()

            # 语义查询文本
            has_query_text = (search_params.get('user_query_text') or search_params.get('semantic_str')) is not None

            if retrieval_type == 'vector' and has_query_text:
                # 仅向量检索
                query_text = search_params.get('user_query_text') or search_params.get('semantic_str')
                query_vector = self.vectorization_utils.create_query_vector(query_text)
                # 强制确保为 Python float 列表
                try:
                    query_vector = [float(x) for x in (query_vector or [])]
                except Exception:
                    pass
                search_results = self.client.search(
                    collection_name=self.COLLECTION_NAME,
                    data=[query_vector],
                    filter=filter_expr if filter_expr else None,
                    limit=limit + offset,
                    output_fields=self._get_safe_output_fields(),
                    search_params=self._build_milvus_search_params(search_params.get('confidence')),
                    anns_field="semantic_vector"
                )
            elif retrieval_type == 'bm25' and has_query_text:
                # 仅BM25稀疏检索
                query_text = search_params.get('user_query_text') or search_params.get('semantic_str')
                search_results = self.client.search(
                    collection_name=self.COLLECTION_NAME,
                    data=[{"text": query_text}],
                    filter=filter_expr if filter_expr else None,
                    limit=limit + offset,
                    output_fields=self._get_safe_output_fields(),
                    search_params={"metric_type": "BM25"},
                    anns_field="sparse_vector"
                )
            elif retrieval_type == 'hybrid' and has_query_text:
                # 向量 + 稀疏（BM25）混合检索
                query_text = search_params.get('user_query_text') or search_params.get('semantic_str')
                query_vector = self.vectorization_utils.create_query_vector(query_text)

                vector_req = AnnSearchRequest(
                    data=[query_vector],
                    anns_field="semantic_vector",
                    param=self._build_milvus_search_params(search_params.get('confidence')),
                    limit=limit + offset,
                    expr=filter_expr or "",
                )
                bm25_req = AnnSearchRequest(
                    data=[query_text],
                    anns_field="sparse_vector",
                    param={"metric_type": "BM25"},
                    limit=limit + offset,
                    expr=filter_expr or "",
                )

                # 选择排序器：强制使用 WeightedRanker
                # if WeightedRanker is None:
                #     raise RuntimeError("WeightedRanker 不可用，请升级 pymilvus 到支持 WeightedRanker 的版本")
                hybrid_weights = search_params.get('hybrid_weights') or {}
                # 兼容旧参数名
                dense_weight = hybrid_weights.get('vector')
                sparse_weight = hybrid_weights.get('bm25')

                # 默认均分（未提供时）
                if dense_weight is None:
                    dense_weight = 0.5
                if sparse_weight is None:
                    sparse_weight = 0.5

                # 归一化分数（若提供），不同版本可能命名不同，统一用布尔值
                norm_score_flag = bool(hybrid_weights.get('norm_score', True))
                # 确保权重为 float
                dw = float(dense_weight)
                sw = float(sparse_weight)

                ranker = WeightedRanker(dw, sw, norm_score=norm_score_flag)  # type: ignore

                try:
                    search_results = self.client.hybrid_search(
                        collection_name=self.COLLECTION_NAME,
                        reqs=[vector_req, bm25_req],
                        ranker=ranker,
                        limit=limit + offset,
                        output_fields=self._get_safe_output_fields(),
                    )
                except Exception as hybrid_ex:
                    # 兼容：若环境未启用 BM25 Function 或稀疏向量输入格式不被支持，则回退为纯向量检索
                    logger.warning(
                        "混合检索失败，回退为向量检索: %s", hybrid_ex
                    )
                    search_results = self.client.search(
                        collection_name=self.COLLECTION_NAME,
                        data=[query_vector],
                        filter=filter_expr if filter_expr else None,
                        limit=limit + offset,
                        output_fields=self._get_safe_output_fields(),
                        search_params=self._build_milvus_search_params(search_params.get('confidence')),
                        anns_field="semantic_vector"
                    )
            else:
                search_results = None

            # 处理结果并应用offset + 置信度阈值
            results = []
            if search_results and len(search_results) > 0:
                for hit in search_results[0]:
                    score = hit['distance']
                    house_data = hit['entity']
                    house_data['similarity_score'] = score
                    serialized_data = self._serialize_house_data(house_data)
                    results.append(serialized_data)
                # 应用offset/limit裁剪
                results = results[offset:offset + limit]
            else:
                # 纯过滤查询
                if filter_expr:
                    query_results = self.client.query(
                        collection_name=self.COLLECTION_NAME,
                        filter=filter_expr,
                        output_fields=self._get_safe_output_fields(),
                        limit=limit,
                        offset=offset
                    )
                    # 序列化查询结果
                    results = [self._serialize_house_data(house) for house in query_results]
                else:
                    # 如果没有任何过滤条件，返回空结果
                    results = []
            
            # 如果是圆形区域搜索：使用 longitude/latitude 精确距离并按距离排序
            if search_params.get('location') and self._is_circle_search(search_params['location']):
                results = self._apply_circle_distance_filter(results, search_params['location'], limit)
            else:
                # 无圆形距离排序时：若包含语义分数，可按分数降序
                if results and 'similarity_score' in results[0]:
                    results.sort(key=lambda x: x.get('similarity_score', 0.0), reverse=True)
            
            logger.info(f"搜索完成，返回 {len(results)} 条结果")
            return results
            
        except Exception as e:
            logger.error(f"搜索房源失败: {e}")
            return []
    
    def _build_filter_expression(self, search_params: Dict[str, Any]) -> Optional[str]:
        """根据新字段规范构建过滤表达式。"""
        conditions: List[str] = []

        # 分类
        if 'category' in search_params and search_params['category'] is not None:
            cat_val = search_params['category']
            if isinstance(cat_val, (list, tuple, set)):
                cats = [int(v) for v in cat_val]
                if cats:
                    conditions.append(f"category in {cats}")
            else:
                conditions.append(f"category == {int(cat_val)}")

        # 名称
        if search_params.get('name') is not None:
            name_param = search_params['name']
            if isinstance(name_param, list) and name_param:
                names_join = ', '.join([f'"{n}"' for n in name_param])
                conditions.append(f"name in [{names_join}]")
            elif isinstance(name_param, str) and name_param:
                conditions.append(f'name == "{name_param}"')

        # 区域
        if search_params.get('region'):
            conditions.append(f'region == "{search_params["region"]}"')

        # 简单等值字符串字段
        simple_string_fields = [
            'type', 'year_completion', 'transaction_ownership', 'parking_space_ratio',
            'management_company', 'developer', 'decoration_style', 'decoration_status',
            'water_electricity', 'has_elevator', 'has_parking', 'orientation',
            'building_age', 'floor', 'rental_mode', 'payment_method'
        ]
        for f in simple_string_fields:
            if search_params.get(f):
                conditions.append(f'{f} == "{search_params[f]}"')

        # 偏好与标签（LIKE 或包含）
        if search_params.get('preferences_tags'):
            conditions.append(f'preferences_tags like "%{search_params["preferences_tags"]}%"')

        # 面积范围：max_area >= min && min_area <= max
        if search_params.get('area_range'):
            ar = search_params['area_range']
            if ar.get('min_area') is not None:
                conditions.append(f'max_area >= {ar["min_area"]}')
            if ar.get('max_area') is not None:
                conditions.append(f'min_area <= {ar["max_area"]}')

        # 单价范围
        if search_params.get('unit_price_range'):
            upr = search_params['unit_price_range']
            if upr.get('min_unit_price') is not None:
                conditions.append(f'max_unit_price >= {upr["min_unit_price"]}')
            if upr.get('max_unit_price') is not None:
                conditions.append(f'min_unit_price <= {upr["max_unit_price"]}')

        # 总价范围
        if search_params.get('total_price_range'):
            tpr = search_params['total_price_range']
            if tpr.get('min_total_price') is not None:
                conditions.append(f'max_total_price >= {tpr["min_total_price"]}')
            if tpr.get('max_total_price') is not None:
                conditions.append(f'min_total_price <= {tpr["max_total_price"]}')

        # 租金范围
        if search_params.get('rent_range'):
            rr = search_params['rent_range']
            if rr.get('min_rent') is not None:
                conditions.append(f'rent >= {rr["min_rent"]}')
            if rr.get('max_rent') is not None:
                conditions.append(f'rent <= {rr["max_rent"]}')

        # 产权年限
        if search_params.get('property_right_duration_range'):
            pr = search_params['property_right_duration_range']
            if pr.get('min_years') is not None:
                conditions.append(f'property_right_duration >= {pr["min_years"]}')
            if pr.get('max_years') is not None:
                conditions.append(f'property_right_duration <= {pr["max_years"]}')

        # 绿化率/容积率/物业费范围
        if search_params.get('greening_rate_range'):
            gr = search_params['greening_rate_range']
            if gr.get('min') is not None:
                conditions.append(f'greening_rate >= {gr["min"]}')
            if gr.get('max') is not None:
                conditions.append(f'greening_rate <= {gr["max"]}')
        if search_params.get('plot_ratio_range'):
            prr = search_params['plot_ratio_range']
            if prr.get('min') is not None:
                conditions.append(f'plot_ratio >= {prr["min"]}')
            if prr.get('max') is not None:
                conditions.append(f'plot_ratio <= {prr["max"]}')
        if search_params.get('management_fee_range'):
            mfr = search_params['management_fee_range']
            if mfr.get('min') is not None:
                conditions.append(f'management_fee >= {mfr["min"]}')
            if mfr.get('max') is not None:
                conditions.append(f'management_fee <= {mfr["max"]}')

        # 地理位置过滤
        if search_params.get('location'):
            location_filter = self._build_location_filter(search_params['location'])
            if location_filter:
                conditions.append(location_filter)

        return ' AND '.join(conditions) if conditions else None
    
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
            conditions.append(f'longitude >= {location["min_longitude"]}')
        if location.get('max_longitude') is not None:
            conditions.append(f'longitude <= {location["max_longitude"]}')
        if location.get('min_latitude') is not None:
            conditions.append(f'latitude >= {location["min_latitude"]}')
        if location.get('max_latitude') is not None:
            conditions.append(f'latitude <= {location["max_latitude"]}')
        
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
            conditions.append(f'longitude >= {location["min_jd"]}')
        if location.get('max_jd') is not None:
            conditions.append(f'longitude <= {location["max_jd"]}')
        if location.get('min_wd') is not None:
            conditions.append(f'latitude >= {location["min_wd"]}')
        if location.get('max_wd') is not None:
            conditions.append(f'latitude <= {location["max_wd"]}')
        
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
                f'latitude >= {min_lat}',
                f'latitude <= {max_lat}',
                f'longitude >= {min_lng}',
                f'longitude <= {max_lng}'
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
                if house.get('longitude') is not None and house.get('latitude') is not None:
                    distance = self.calculate_distance_km(
                        center_lat, center_lng,
                        house['latitude'], house['longitude']
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
            _ = semantic_weight  # 标记参数已被考虑但暂未使用
            
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
                if house.get('longitude') is not None and house.get('latitude') is not None:
                    distance = self.calculate_distance_km(
                        center_latitude, center_longitude,
                        house['latitude'], house['longitude']
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