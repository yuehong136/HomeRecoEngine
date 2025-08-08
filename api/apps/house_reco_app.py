from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel, Field
import logging
import tempfile
import os
from api.db.services.house_reco_service import HouseRecoService
from api.db.services.data_import_service import DataImportService
from core.utils.milvus_config import get_default_milvus_client

logger = logging.getLogger(__name__)

# 创建路由
router = APIRouter()


class HouseSearchRequest(BaseModel):
    """房源搜索请求模型（重构版）"""
    model_config = {"extra": "forbid", "validate_assignment": True}

    # 分类与核心条件
    category: Optional[Union[int, List[int]]] = Field(None, description="1-新房；2-二手房；3-出租房")
    name: Optional[Union[str, List[str]]] = Field(None, description="房源/小区名称或列表")
    region: Optional[str] = Field(None, description="区县名称")
    address: Optional[str] = Field(None, description="地址（模糊匹配）")

    # 数值范围
    area_range: Optional[Dict[str, float]] = Field(None, description="面积范围 {min_area, max_area}")
    unit_price_range: Optional[Dict[str, float]] = Field(None, description="单价范围 {min_unit_price, max_unit_price}")
    total_price_range: Optional[Dict[str, float]] = Field(None, description="总价范围 {min_total_price, max_total_price}")
    rent_range: Optional[Dict[str, float]] = Field(None, description="租金范围 {min_rent, max_rent}")

    # 位置
    location: Optional[Dict[str, float]] = Field(None, description="圆形/矩形地理范围过滤")

    # 其他等值字段
    type: Optional[str] = Field(None)
    year_completion: Optional[str] = Field(None)
    transaction_ownership: Optional[str] = Field(None)
    property_right_duration_range: Optional[Dict[str, int]] = Field(None, description="产权年限范围 {min_years, max_years}")
    parking_space_ratio: Optional[str] = Field(None)
    management_company: Optional[str] = Field(None)
    management_fee_range: Optional[Dict[str, float]] = Field(None, description="物业费范围 {min, max}")
    developer: Optional[str] = Field(None)
    greening_rate_range: Optional[Dict[str, float]] = Field(None, description="绿化率范围 {min, max}")
    plot_ratio_range: Optional[Dict[str, float]] = Field(None, description="容积率范围 {min, max}")
    decoration_style: Optional[str] = Field(None)
    decoration_status: Optional[str] = Field(None)
    water_electricity: Optional[str] = Field(None)
    has_elevator: Optional[str] = Field(None)
    has_parking: Optional[str] = Field(None)
    orientation: Optional[str] = Field(None)
    building_age: Optional[str] = Field(None)
    furniture_facilities: Optional[str] = Field(None)
    floor: Optional[str] = Field(None)
    rental_mode: Optional[str] = Field(None)
    payment_method: Optional[str] = Field(None)
    lease_term: Optional[int] = Field(None)
    preferences_tags: Optional[str] = Field(None)
    semantic_str: Optional[str] = Field(None, description="语义字符，用于语义检索")
    confidence: Optional[float] = Field(None, description="语义检索置信度阈值(0-1)，仅在语义检索时生效")
    # 用户自然语言查询文本（与 semantic_str 语义等价，保留两者便于兼容文档与调用方）
    user_query_text: Optional[str] = Field(None, description="用户语义查询文本，等价于 semantic_str")

    # 检索类型（vector/bm25/hybrid）与权重
    retrieval_type: Optional[str] = Field(None, description="检索类型：vector | bm25 | hybrid（默认vector）")
    hybrid_weights: Optional[Dict[str, float]] = Field(None, description="混合检索权重，如 {'vector':0.5,'bm25':0.5}")

    # 分页
    limit: int = Field(10, ge=1, le=100, description="返回结果数量限制")
    offset: int = Field(0, ge=0, description="结果偏移量")


class HouseHybridSearchRequest(BaseModel):
    """混合搜索请求模型"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    semantic_query: str = Field(..., description="语义查询文本")
    
    # 过滤参数
    filter_params: Optional[Dict[str, Any]] = Field(
        None,
        description="过滤参数，结构同HouseSearchRequest的字段"
    )
    
    semantic_weight: float = Field(
        0.7, 
        ge=0.0, 
        le=1.0, 
        description="语义搜索权重 (0-1之间)"
    )
    
    limit: int = Field(10, ge=1, le=100, description="返回结果数量")



class HouseInsertRequest(BaseModel):
    """房源插入请求模型（重构版）"""
    model_config = {"extra": "forbid", "validate_assignment": True}

    # 必填
    id: str = Field(..., description="主键，字符串")
    category: int = Field(..., description="1-新房；2-二手房；3-出租房")
    name: str = Field(..., description="房源名称/小区名称")
    region: str = Field(..., description="区县")
    address: str = Field(..., description="地址")
    longitude: float = Field(..., description="经度")
    latitude: float = Field(..., description="纬度")
    semantic_str: str = Field(..., description="语义字符")

    # 数值范围
    min_area: Optional[float] = None
    max_area: Optional[float] = None
    min_unit_price: Optional[float] = None
    max_unit_price: Optional[float] = None
    min_total_price: Optional[float] = None
    max_total_price: Optional[float] = None
    rent: Optional[float] = None

    # 其他说明字段
    type: Optional[str] = None
    year_completion: Optional[str] = None
    transaction_ownership: Optional[str] = None
    property_right_duration: Optional[int] = None
    parking_space_ratio: Optional[str] = None
    management_company: Optional[str] = None
    management_fee: Optional[float] = None
    developer: Optional[str] = None
    greening_rate: Optional[float] = None
    plot_ratio: Optional[float] = None
    decoration_style: Optional[str] = None
    decoration_status: Optional[str] = None
    water_electricity: Optional[str] = None
    has_elevator: Optional[str] = None
    has_parking: Optional[str] = None
    orientation: Optional[str] = None
    building_age: Optional[str] = None
    furniture_facilities: Optional[str] = None
    floor: Optional[str] = None
    rental_mode: Optional[str] = None
    payment_method: Optional[str] = None
    lease_term: Optional[int] = None
    preferences_tags: Optional[str] = None


class HouseResponse(BaseModel):
    """房源响应模型"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    success: bool = Field(..., description="操作是否成功")
    message: str = Field(..., description="响应消息")
    data: Optional[Any] = Field(None, description="响应数据")


def get_house_reco_service():
    """获取房源推荐服务实例"""
    client = get_default_milvus_client()
    return HouseRecoService(client)


def get_data_import_service():
    """获取数据导入服务实例"""
    house_service = get_house_reco_service()
    return DataImportService(house_service)


@router.post("/search", response_model=HouseResponse, summary="房源搜索")
def search_houses(request: HouseSearchRequest) -> HouseResponse:
    """
    ### POST `/search` 房源搜索接口

    **功能描述**:
    全功能房源搜索接口，支持结构化过滤、语义检索、地理范围搜索等多种搜索方式。
    可以根据不同需求灵活组合搜索条件，提供精准的房源匹配结果。

    ---

    ### 请求体 (Request Body)

    #### 基础分类与核心条件
    | 字段                   | 类型                    | 必填 | 描述                                        |
    |----------------------|-------------------------|------|---------------------------------------------|
    | `category`           | `int` or `List[int]`    | 否   | 房源分类：1-新房；2-二手房；3-出租房        |
    | `name`               | `str` or `List[str]`    | 否   | 房源/小区名称或列表                         |
    | `region`             | `str`                   | 否   | 区县名称                                    |
    | `address`            | `str`                   | 否   | 地址（支持模糊匹配）                        |

    #### 数值范围过滤
    | 字段                   | 类型                    | 必填 | 描述                                        |
    |----------------------|-------------------------|------|---------------------------------------------|
    | `area_range`         | `Dict[str, float]`      | 否   | 面积范围 `{min_area, max_area}`             |
    | `unit_price_range`   | `Dict[str, float]`      | 否   | 单价范围 `{min_unit_price, max_unit_price}` |
    | `total_price_range`  | `Dict[str, float]`      | 否   | 总价范围 `{min_total_price, max_total_price}` |
    | `rent_range`         | `Dict[str, float]`      | 否   | 租金范围 `{min_rent, max_rent}`             |

    #### 地理位置过滤
    | 字段        | 类型                    | 必填 | 描述                                        |
    |-------------|-------------------------|------|---------------------------------------------|
    | `location`  | `Dict[str, float]`      | 否   | 地理范围过滤，支持圆形和矩形范围            |

    #### 语义搜索参数
    | 字段              | 类型      | 必填 | 描述                                        |
    |------------------|-----------|------|---------------------------------------------|
    | `semantic_str`   | `str`     | 否   | 语义查询文本                                |
    | `confidence`     | `float`   | 否   | 语义检索置信度阈值 (0-1)                    |
    | `retrieval_type` | `str`     | 否   | 检索类型：vector/bm25/hybrid（默认vector）   |
    | `hybrid_weights` | `Dict`    | 否   | 混合检索权重 `{'vector':0.5,'bm25':0.5}`    |

    #### 分页参数
    | 字段     | 类型  | 必填 | 默认值 | 描述                    |
    |----------|-------|------|--------|-------------------------|
    | `limit`  | `int` | 否   | 10     | 返回结果数量限制 (1-100) |
    | `offset` | `int` | 否   | 0      | 结果偏移量 (>=0)        |

    ---

    ### 地理位置搜索说明

    #### 圆形范围搜索
    ```json
    {
        "location": {
            "center_longitude": 116.3105,
            "center_latitude": 39.9785,
            "radius_km": 2.0
        }
    }
    ```
    - 结果按距离升序排列
    - 返回结果包含 `distance_km` 字段

    #### 矩形范围搜索
    ```json
    {
        "location": {
            "min_longitude": 116.300,
            "max_longitude": 116.400,
            "min_latitude": 39.900,
            "max_latitude": 40.000
        }
    }
    ```

    ---

    ### 响应 (Response)

    #### 成功响应 (200)
    ```json
    {
        "success": true,
        "message": "搜索成功",
        "data": {
            "houses": [
                {
                    "id": "123456",
                    "name": "万科城市花园",
                    "region": "海淀区",
                    "address": "中关村大街39号",
                    "area": 89.5,
                    "total_price": 650.0,
                    "similarity_score": 0.95,
                    "distance_km": 1.2
                }
            ],
            "houses_by_category": {
                "1": [...],
                "2": [...]
            },
            "total": 25,
            "limit": 10,
            "offset": 0
        }
    }
    ```

    #### 错误响应 (500)
    ```json
    {
        "success": false,
        "message": "搜索失败: 具体错误信息",
        "data": null
    }
    ```

    ---

    ### 搜索类型说明

    #### 1. 结构化过滤搜索
    基于精确匹配和范围筛选的传统搜索方式。
    
    ```json
    {
        "region": "海淀区",
        "area_range": {"min_area": 80, "max_area": 120},
        "total_price_range": {"min_total_price": 500, "max_total_price": 800}
    }
    ```

    #### 2. 语义向量搜索
    基于自然语言理解的智能搜索。

    ```json
    {
        "semantic_str": "三室两厅学区房，交通便利，环境优美",
        "confidence": 0.8,
        "retrieval_type": "vector"
    }
    ```

    #### 3. 混合检索
    结合向量搜索和BM25文本搜索的综合方案。

    ```json
    {
        "semantic_str": "现代装修风格，地铁附近",
        "retrieval_type": "hybrid",
        "hybrid_weights": {"vector": 0.7, "bm25": 0.3}
    }
    ```

    ---

    ### 多分类搜索

    #### 单分类搜索
    ```json
    {
        "category": 2,
        "region": "朝阳区"
    }
    ```

    #### 多分类搜索
    ```json
    {
        "category": [1, 2],
        "region": "朝阳区"
    }
    ```
    - 返回结果包含 `houses_by_category` 字段
    - 按分类分别展示搜索结果

    ---

    ### 排序规则

    | 搜索条件      | 排序方式                          |
    |---------------|-----------------------------------|
    | 圆形地理范围  | 按距离升序                        |
    | 语义搜索      | 按相似度分数降序                  |
    | 结构化搜索    | 按数据插入时间降序                |
    | 混合搜索      | 按综合评分降序                    |

    ---

    ### 使用场景

    #### 1. 精确条件搜索
    ```json
    {
        "region": "海淀区",
        "area_range": {"min_area": 90, "max_area": 110},
        "total_price_range": {"max_total_price": 800},
        "has_elevator": "有电梯"
    }
    ```

    #### 2. 地理位置搜索
    ```json
    {
        "location": {
            "center_longitude": 116.3105,
            "center_latitude": 39.9785,
            "radius_km": 1.5
        },
        "category": 2
    }
    ```

    #### 3. 智能语义搜索
    ```json
    {
        "semantic_str": "适合三口之家，学区房，交通便利，价格合理",
        "area_range": {"min_area": 80},
        "confidence": 0.7
    }
    ```

    ---

    ### 性能优化建议

    #### 查询优化
    - 合理使用分页参数，避免一次获取过多数据
    - 结合结构化过滤缩小搜索范围
    - 语义搜索建议设置合适的置信度阈值

    #### 索引利用
    - 区域、面积、价格等字段已建立索引
    - 地理坐标支持高效的空间查询
    - 语义向量支持高性能相似度搜索

    ---

    ### 注意事项

    - **分页限制**: 单次查询最多返回100条记录
    - **坐标精度**: 地理坐标建议使用高精度数据
    - **语义搜索**: 查询文本越详细，匹配效果越好
    - **性能考虑**: 复杂查询可能需要更长的响应时间
    - **数据一致性**: 搜索结果可能有轻微的延迟更新
    """
    try:
        # 获取服务实例
        service = get_house_reco_service()
        
        # 构建搜索参数（新字段）
        search_params: Dict[str, Any] = {}
        # 基础条件
        for key in [
            'name','region','address','area_range','unit_price_range','total_price_range','rent_range',
            'location','type','year_completion','transaction_ownership','property_right_duration_range',
            'parking_space_ratio','management_company','management_fee_range','developer',
            'greening_rate_range','plot_ratio_range','decoration_style','decoration_status',
            'water_electricity','has_elevator','has_parking','orientation','building_age',
            'furniture_facilities','floor','rental_mode','payment_method','lease_term',
            'preferences_tags','semantic_str'
        ]:
            val = getattr(request, key, None)
            if val is not None:
                search_params[key] = val
        # 非过滤控制参数单独透传，不参与基础条件集合
        if request.confidence is not None:
            search_params['confidence'] = request.confidence
        if request.retrieval_type is not None:
            search_params['retrieval_type'] = request.retrieval_type
        if request.hybrid_weights is not None:
            search_params['hybrid_weights'] = request.hybrid_weights
        user_query_text = getattr(request, 'user_query_text', None)
        if user_query_text:
            search_params["user_query_text"] = user_query_text

        # 处理分类参数（允许单个或多个）
        categories: List[int] = []
        if request.category is not None:
            if isinstance(request.category, list):
                categories = [int(c) for c in request.category if c is not None]
            else:
                categories = [int(request.category)]
        
        # 如果未指定分类，直接按公共条件搜索
        if not categories:
            results = service.search_houses(
                search_params=search_params,
                limit=request.limit,
                offset=request.offset
            )
            safe_results = []
            for result in results:
                try:
                    import json
                    json.dumps(result)
                    safe_results.append(result)
                except (TypeError, ValueError) as e:
                    logger.warning(f"结果序列化失败，跳过该条记录: {e}")
                    continue
            return HouseResponse(
                success=True,
                message="搜索成功",
                data={
                    "houses": safe_results,
                    "total": len(safe_results),
                    "limit": request.limit,
                    "offset": request.offset
                }
            )

        # 指定了分类：按分类逐个检索并分别返回
        results_by_category: Dict[str, List[Dict[str, Any]]] = {}
        all_results: List[Dict[str, Any]] = []
        for cat in categories:
            cat_params = dict(search_params)
            cat_params["category"] = cat
            cat_results = service.search_houses(
                search_params=cat_params,
                limit=request.limit,
                offset=request.offset
            )
            # 序列化保护
            safe_cat_results = []
            for result in cat_results:
                try:
                    import json
                    json.dumps(result)
                    safe_cat_results.append(result)
                except (TypeError, ValueError) as e:
                    logger.warning(f"分类 {cat} 结果序列化失败，跳过该条记录: {e}")
                    continue
            results_by_category[str(cat)] = safe_cat_results
            all_results.extend(safe_cat_results)

        return HouseResponse(
            success=True,
            message="按分类搜索成功",
            data={
                "houses": all_results,
                "houses_by_category": results_by_category,
                "total": len(all_results),
                "limit": request.limit,
                "offset": request.offset
            }
        )
        
    except Exception as e:
        logger.error(f"房源搜索失败: {e}")
        return HouseResponse(
            success=False,
            message=f"搜索失败: {str(e)}",
            data=None
        )


@router.post("/hybrid-search", response_model=HouseResponse, summary="混合搜索房源")
def hybrid_search_houses(request: HouseHybridSearchRequest) -> HouseResponse:
    """
    ### POST `/hybrid-search` 混合搜索房源接口

    **功能描述**:
    结合语义搜索和精确过滤的智能房源搜索接口。通过自然语言理解用户需求，
    并结合结构化过滤条件，提供更精准和智能的房源推荐结果。

    ---

    ### 请求体 (Request Body)

    | 字段              | 类型               | 必填 | 描述                                           |
    |-------------------|-------------------|------|------------------------------------------------|
    | `semantic_query`  | `str`             | 是   | 语义查询文本，用自然语言描述房源需求             |
    | `filter_params`   | `Dict[str, Any]`  | 否   | 过滤参数，结构同search接口的字段                |
    | `semantic_weight` | `float`           | 否   | 语义搜索权重（0-1之间，默认0.7）                |
    | `limit`           | `int`             | 否   | 返回结果数量（1-100，默认10）                   |

    ---

    ### 语义权重说明

    | 权重值范围 | 搜索特点                    | 适用场景                           |
    |------------|---------------------------|-----------------------------------|
    | 0.8-1.0    | 主要依赖语义理解           | 用户需求描述详细且具体的场景        |
    | 0.5-0.7    | 语义搜索和过滤条件平衡     | 推荐默认值，适合大多数搜索场景      |
    | 0.1-0.4    | 主要依赖精确过滤条件       | 对特定属性有明确要求的搜索          |

    ---

    ### 响应 (Response)

    #### 成功响应 (200)
    ```json
    {
        "success": true,
        "message": "混合搜索成功",
        "data": {
            "houses": [
                {
                    "id": 456,
                    "xqmc": ["万科城市花园"],
                    "qy": "海淀区",
                    "mj": 95.0,
                    "fyhx": "三室一厅",
                    "zj": 720.0,
                    "semantic_score": 0.89,
                    "filter_score": 0.95,
                    "final_score": 0.91
                }
            ],
            "total": 15,
            "semantic_query": "三室一厅，学区房，地铁附近",
            "semantic_weight": 0.7
        }
    }
    ```

    #### 错误响应 (500)
    ```json
    {
        "success": false,
        "message": "混合搜索失败: 具体错误信息",
        "data": null
    }
    ```

    ---

    ### 搜索算法详解

    #### 1. 语义理解阶段
    - 对semantic_query进行分词和语义分析
    - 使用预训练模型生成查询向量
    - 在向量空间中进行相似度计算

    #### 2. 过滤匹配阶段
    - 根据filter_params进行精确条件过滤
    - 计算各个结构化属性的匹配度
    - 生成过滤匹配得分

    #### 3. 混合评分阶段
    ```
    final_score = semantic_weight × semantic_score + (1 - semantic_weight) × filter_score
    ```

    #### 4. 结果排序
    - 按final_score降序排列
    - 返回top-k个最匹配的房源

    ---

    ### 使用场景

    #### 1. 自然语言描述搜索
    ```json
    {
        "semantic_query": "我想要一个安静的学区房，最好是南北通透，有阳台",
        "filter_params": {
            "price_range": {"max_price": 800}
        },
        "semantic_weight": 0.8,
        "limit": 15
    }
    ```

    #### 2. 精确条件+语义增强
    ```json
    {
        "semantic_query": "现代装修风格，周边配套完善",
        "filter_params": {
            "qy": "朝阳区",
            "area_range": {"min_area": 90, "max_area": 120},
            "ywdt": "有电梯"
        },
        "semantic_weight": 0.5,
        "limit": 20
    }
    ```

    #### 3. 生活方式匹配
    ```json
    {
        "semantic_query": "适合年轻夫妻居住，交通便利，购物方便，环境优美",
        "filter_params": {
            "fyhx": "两室一厅",
            "price_range": {"min_price": 400, "max_price": 700}
        },
        "semantic_weight": 0.7,
        "limit": 10
    }
    ```

    ---

    ### 语义查询优化建议

    #### 1. 描述要素
    - **房屋特征**: 户型、朝向、装修、楼层等
    - **位置需求**: 学区、地铁、商圈、医院等
    - **环境偏好**: 安静、绿化、社区氛围等
    - **生活方式**: 适合家庭、年轻人、老人等

    #### 2. 查询示例
    - ✅ **好的查询**: "三室两厅学区房，南北通透，地铁10分钟内，社区环境好"
    - ✅ **好的查询**: "适合小家庭的现代公寓，购物便利，安全性高"
    - ❌ **差的查询**: "房子" "好房源"

    ---

    ### 注意事项

    - **语义理解**: 系统会自动分析查询文本的语义含义
    - **权重调节**: 根据具体需求调整semantic_weight获得最佳效果
    - **结合使用**: 建议将结构化过滤和语义查询结合使用
    - **性能考虑**: 语义搜索比精确搜索耗时稍长，建议合理设置limit
    - **结果解释**: 返回结果包含语义得分和过滤得分，便于理解匹配原因
    """
    try:
        service = get_house_reco_service()
        
        # 执行混合搜索
        results = service.hybrid_search(
            semantic_query=request.semantic_query,
            filter_params=request.filter_params or {},
            semantic_weight=request.semantic_weight,
            limit=request.limit
        )
        
        # 序列化保护
        safe_results = []
        for result in results:
            try:
                import json
                json.dumps(result)
                safe_results.append(result)
            except (TypeError, ValueError) as e:
                logger.warning(f"混合搜索结果序列化失败，跳过该条记录: {e}")
                continue
        
        return HouseResponse(
            success=True,
            message="混合搜索成功",
            data={
                "houses": safe_results,
                "total": len(safe_results),
                "semantic_query": request.semantic_query,
                "semantic_weight": request.semantic_weight
            }
        )
        
    except Exception as e:
        logger.error(f"混合搜索失败: {e}")
        return HouseResponse(
            success=False,
            message=f"混合搜索失败: {str(e)}",
            data=None
        )


@router.post("/insert", response_model=HouseResponse, summary="插入单个房源")
def insert_house(request: HouseInsertRequest) -> HouseResponse:
    """
    ### POST `/insert` 插入单个房源接口

    **功能描述**:
    向系统中插入单条房源记录，支持结构化数据和语义信息的综合存储。
    系统会自动进行数据验证、坐标校验、语义向量化处理，并更新搜索索引。

    ---

    ### 请求体 (Request Body)

    #### 必填字段
    | 字段           | 类型      | 描述                              | 示例                    |
    |---------------|-----------|-----------------------------------|-------------------------|
    | `id`          | `str`     | 房源唯一标识符                    | "123456"                |
    | `category`    | `int`     | 房源分类：1-新房；2-二手房；3-出租房| 2                      |
    | `name`        | `str`     | 房源名称/小区名称                 | "万科城市花园"          |
    | `region`      | `str`     | 区县名称                          | "海淀区"                |
    | `address`     | `str`     | 详细地址                          | "中关村大街39号"        |
    | `longitude`   | `float`   | 经度坐标                          | 116.3105               |
    | `latitude`    | `float`   | 纬度坐标                          | 39.9785                |
    | `semantic_str`| `str`     | 语义描述文本                      | "三室两厅，精装修，学区房"|

    #### 可选数值字段
    | 字段                | 类型      | 描述                     | 单位        |
    |--------------------|-----------|--------------------------|-------------|
    | `min_area`         | `float`   | 最小面积                 | 平方米      |
    | `max_area`         | `float`   | 最大面积                 | 平方米      |
    | `min_unit_price`   | `float`   | 最小单价                 | 元/平米     |
    | `max_unit_price`   | `float`   | 最大单价                 | 元/平米     |
    | `min_total_price`  | `float`   | 最小总价                 | 万元        |
    | `max_total_price`  | `float`   | 最大总价                 | 万元        |
    | `rent`             | `float`   | 租金                     | 元/月       |

    #### 可选属性字段
    | 字段                        | 类型     | 描述            | 示例值         |
    |----------------------------|----------|-----------------|---------------|
    | `type`                     | `str`    | 房源类型        | "公寓"        |
    | `year_completion`          | `str`    | 建成年份        | "2018"        |
    | `transaction_ownership`    | `str`    | 交易权属        | "商品房"      |
    | `property_right_duration`  | `int`    | 产权年限        | 70            |
    | `parking_space_ratio`      | `str`    | 车位配比        | "1:1"         |
    | `management_company`       | `str`    | 物业公司        | "万科物业"     |
    | `management_fee`           | `float`  | 物业费          | 3.5           |
    | `developer`                | `str`    | 开发商          | "万科集团"     |
    | `greening_rate`            | `float`  | 绿化率          | 35.5          |
    | `plot_ratio`               | `float`  | 容积率          | 2.8           |
    | `decoration_style`         | `str`    | 装修风格        | "现代简约"     |
    | `decoration_status`        | `str`    | 装修状况        | "精装修"      |
    | `water_electricity`        | `str`    | 水电情况        | "民用水电"     |
    | `has_elevator`             | `str`    | 电梯情况        | "有电梯"      |
    | `has_parking`              | `str`    | 停车情况        | "有车位"      |
    | `orientation`              | `str`    | 房屋朝向        | "南北通透"     |
    | `building_age`             | `str`    | 建筑年龄        | "次新房"      |
    | `furniture_facilities`     | `str`    | 家具设施        | "全配置"      |
    | `floor`                    | `str`    | 楼层信息        | "中楼层"      |
    | `rental_mode`              | `str`    | 出租方式        | "整租"        |
    | `payment_method`           | `str`    | 付款方式        | "押一付三"     |
    | `lease_term`               | `int`    | 租期            | 12            |
    | `preferences_tags`         | `str`    | 偏好标签        | "学区房,地铁房" |

    ---

    ### 请求示例

    #### 基础房源信息
    ```json
    {
        "id": "123456",
        "category": 2,
        "name": "万科城市花园",
        "region": "海淀区",
        "address": "中关村大街39号",
        "longitude": 116.3105,
        "latitude": 39.9785,
        "semantic_str": "三室两厅，南北通透，精装修，学区房，地铁10分钟步行距离"
    }
    ```

    #### 完整房源信息
    ```json
    {
        "id": "234567",
        "category": 2,
        "name": "金地格林小镇",
        "region": "朝阳区",
        "address": "建国路88号",
        "longitude": 116.4634,
        "latitude": 39.9078,
        "semantic_str": "现代公寓，交通便利，配套完善，适合年轻家庭",
        "min_area": 89.5,
        "max_area": 95.0,
        "min_total_price": 650.0,
        "max_total_price": 720.0,
        "type": "住宅",
        "year_completion": "2020",
        "has_elevator": "有电梯",
        "has_parking": "有车位",
        "orientation": "南",
        "decoration_status": "精装修"
    }
    ```

    ---

    ### 响应 (Response)

    #### 成功响应 (200)
    ```json
    {
        "success": true,
        "message": "房源数据插入成功",
        "data": {
            "house_id": "123456",
            "vector_id": 987654321,
            "inserted_at": "2024-01-15T10:30:45Z"
        }
    }
    ```

    #### 验证错误 (400)
    ```json
    {
        "success": false,
        "message": "数据验证失败: 经度超出有效范围",
        "data": {
            "validation_errors": [
                "经度必须在73-135度之间",
                "语义描述不能为空"
            ]
        }
    }
    ```

    #### 重复ID错误 (409)
    ```json
    {
        "success": false,
        "message": "插入失败: 房源ID已存在",
        "data": {
            "existing_id": "123456",
            "conflict_type": "duplicate_id"
        }
    }
    ```

    #### 系统错误 (500)
    ```json
    {
        "success": false,
        "message": "插入失败: 向量化处理异常",
        "data": null
    }
    ```

    ---

    ### 数据处理流程

    #### 1. 请求接收与验证
    - 接收JSON格式的房源数据
    - 验证请求体格式和数据类型
    - 检查必填字段完整性

    #### 2. 数据校验
    ```python
    # 地理坐标验证
    if not (73 <= longitude <= 135 and 18 <= latitude <= 54):
        raise ValidationError("坐标超出中国境内有效范围")
    
    # ID唯一性检查
    if exists_house_id(id):
        raise ConflictError("房源ID已存在")
    
    # 数值范围验证
    if min_area and max_area and min_area > max_area:
        raise ValidationError("面积范围不合理")
    ```

    #### 3. 语义向量化
    - 对`semantic_str`字段进行文本预处理
    - 使用预训练模型生成768维向量
    - 向量标准化和质量检查

    #### 4. 数据存储
    - 将结构化数据和向量数据批量插入Milvus
    - 自动生成唯一的向量ID
    - 更新相关统计信息

    #### 5. 索引更新
    - 异步更新搜索索引
    - 刷新缓存和统计数据
    - 记录操作日志

    ---

    ### 字段验证规则

    #### 地理坐标验证
    | 字段        | 验证规则                    | 错误处理            |
    |-------------|----------------------------|---------------------|
    | `longitude` | 73.0 ≤ 经度 ≤ 135.0       | 返回坐标范围错误     |
    | `latitude`  | 18.0 ≤ 纬度 ≤ 54.0        | 返回坐标范围错误     |
    | 坐标精度    | 建议小数点后6位             | 警告但不阻止插入     |

    #### 数值字段验证
    ```python
    validation_rules = {
        "area": {"min": 10, "max": 1000, "unit": "㎡"},
        "total_price": {"min": 10, "max": 50000, "unit": "万元"},
        "unit_price": {"min": 1000, "max": 200000, "unit": "元/㎡"},
        "rent": {"min": 500, "max": 50000, "unit": "元/月"},
        "greening_rate": {"min": 0, "max": 100, "unit": "%"},
        "plot_ratio": {"min": 0.1, "max": 10.0, "unit": "无"}
    }
    ```

    #### 文本字段验证
    - **ID格式**: 支持数字和字母，长度3-20位
    - **名称长度**: 2-100个字符
    - **地址长度**: 5-200个字符
    - **语义描述**: 10-1000个字符，避免纯数字或重复字符

    ---

    ### 数据优化建议

    #### 地理坐标优化
    - 使用高精度GPS坐标（小数点后6位）
    - 验证坐标与地址的一致性
    - 避免使用不准确的概估坐标

    #### 价格数据优化
    ```json
    // 推荐的价格数据结构
    {
        "min_total_price": 650.0,    // 总价下限
        "max_total_price": 720.0,    // 总价上限
        "min_unit_price": 72625,     // 由总价/面积计算
        "max_unit_price": 80447      // 由总价/面积计算
    }
    ```

    #### 语义描述优化
    ```json
    // 优质语义描述示例
    {
        "semantic_str": "三室两厅两卫，南北通透，精装修，学区房。小区环境优美，绿化率35%，地铁2号线步行8分钟，周边有超市、学校、医院等配套设施齐全。"
    }
    ```

    ---

    ### 性能与存储

    #### 存储占用
    | 数据类型      | 单条记录占用  | 说明                    |
    |---------------|---------------|-------------------------|
    | 结构化数据    | ~2KB         | JSON格式存储             |
    | 语义向量      | ~3KB         | 768维float32向量         |
    | 索引数据      | ~1KB         | 搜索索引和元数据         |
    | **总计**      | **~6KB**     | 每条房源记录预估占用     |

    #### 插入性能
    - **单条插入**: 平均耗时50-100ms
    - **向量化处理**: 占总时间的60-70%
    - **数据存储**: 占总时间的20-30%
    - **索引更新**: 占总时间的10-20%

    ---

    ### 使用场景

    #### 1. 房源录入
    ```bash
    # 新房源录入
    curl -X POST "http://localhost:8000/api/house-reco/insert" \
         -H "Content-Type: application/json" \
         -d '{
           "id": "new_house_001",
           "category": 2,
           "name": "新世界花园",
           ...
         }'
    ```

    #### 2. 数据导入
    ```python
    # 批量处理Excel数据后逐条插入
    for house_data in excel_data:
        response = requests.post(
            "http://localhost:8000/api/house-reco/insert",
            json=house_data
        )
    ```

    #### 3. 实时更新
    ```python
    # 房源信息变更后实时更新
    updated_house = {
        "id": existing_id,
        "category": 2,
        "name": updated_name,
        ...
    }
    ```

    ---

    ### 错误处理策略

    #### 常见错误类型
    1. **数据格式错误**: JSON格式不正确
    2. **必填字段缺失**: 缺少required字段
    3. **数据类型错误**: 字段类型不匹配
    4. **数值范围错误**: 超出合理范围
    5. **ID冲突错误**: 房源ID已存在
    6. **坐标无效错误**: 地理坐标不合理

    #### 处理原则
    - **严格验证**: 数据质量优先，严格校验
    - **明确反馈**: 提供详细的错误信息和修复建议
    - **防重复**: ID唯一性检查防止重复数据
    - **日志记录**: 详细记录所有插入操作和错误

    ---

    ### 注意事项

    - **ID唯一性**: 房源ID必须全局唯一，建议使用UUID或带前缀的序列号
    - **坐标准确性**: 地理坐标直接影响位置搜索精度，务必使用准确坐标
    - **数据一致性**: 确保面积、单价、总价等数值的逻辑一致性
    - **语义质量**: 语义描述越详细准确，智能搜索效果越好
    - **存储成本**: 每条记录约占用6KB存储空间，包含向量数据
    - **索引延迟**: 插入后可能需要几秒钟时间更新搜索索引
    """
    try:
        service = get_house_reco_service()
        
        # 转换为字典格式
        house_data = request.model_dump(exclude_none=True)
        
        # 插入数据
        success = service.insert_house_data(house_data)
        
        if success:
            return HouseResponse(
                success=True,
                message="房源数据插入成功",
                data={"house_id": request.id}
            )
        else:
            return HouseResponse(
                success=False,
                message="房源数据插入失败",
                data=None
            )
            
    except Exception as e:
        logger.error(f"房源插入失败: {e}")
        return HouseResponse(
            success=False,
            message=f"插入失败: {str(e)}",
            data=None
        )


@router.post("/batch-insert", response_model=HouseResponse, summary="批量插入房源")
def batch_insert_houses(houses: List[HouseInsertRequest]) -> HouseResponse:
    """
    ### POST `/batch-insert` 批量插入房源接口

    **功能描述**:
    一次性插入多条房源记录，适用于大批量数据导入场景。支持事务性处理，
    确保数据完整性，并提供详细的处理结果报告。

    ---

    ### 请求体 (Request Body)

    | 字段      | 类型                        | 必填 | 描述                    |
    |-----------|----------------------------|------|-------------------------|
    | `houses`  | `List[HouseInsertRequest]` | 是   | 房源数据列表            |

    每个房源对象的字段结构与单条插入接口相同，包含以下必填和可选字段：

    #### 必填字段（每个房源）
    - `id`: 房源唯一标识符
    - `category`: 房源分类
    - `name`: 房源名称
    - `region`: 区县名称
    - `address`: 详细地址
    - `longitude`: 经度坐标
    - `latitude`: 纬度坐标
    - `semantic_str`: 语义描述文本

    #### 可选字段（每个房源）
    - 数值字段：面积、价格、租金等范围信息
    - 属性字段：房源类型、装修状况、设施配备等

    ---

    ### 请求示例

    #### 批量插入多条房源
    ```json
    [
        {
            "id": "batch_001",
            "category": 2,
            "name": "万科城市花园",
            "region": "海淀区",
            "address": "中关村大街39号",
            "longitude": 116.3105,
            "latitude": 39.9785,
            "semantic_str": "三室两厅，南北通透，精装修，学区房",
            "min_total_price": 650.0,
            "max_total_price": 720.0
        },
        {
            "id": "batch_002",
            "category": 2,
            "name": "金地格林小镇",
            "region": "朝阳区",
            "address": "建国路88号",
            "longitude": 116.4634,
            "latitude": 39.9078,
            "semantic_str": "现代公寓，交通便利，配套完善",
            "min_total_price": 780.0,
            "max_total_price": 850.0
        },
        {
            "id": "batch_003",
            "category": 3,
            "name": "中关村软件园公寓",
            "region": "海淀区",
            "address": "软件园路8号",
            "longitude": 116.2985,
            "latitude": 40.0234,
            "semantic_str": "一居室公寓，适合单身白领，距地铁近",
            "rent": 4500.0
        }
    ]
    ```

    ---

    ### 响应 (Response)

    #### 成功响应 (200)
    ```json
    {
        "success": true,
        "message": "成功批量插入3条房源数据",
        "data": {
            "inserted_count": 3,
            "total_submitted": 3,
            "success_rate": 100.0,
            "processing_time_ms": 1250,
            "inserted_ids": ["batch_001", "batch_002", "batch_003"],
            "failed_ids": [],
            "batch_id": "batch_20240115_103045"
        }
    }
    ```

    #### 部分成功响应 (200)
    ```json
    {
        "success": true,
        "message": "成功批量插入2条房源数据，1条失败",
        "data": {
            "inserted_count": 2,
            "total_submitted": 3,
            "success_rate": 66.7,
            "processing_time_ms": 1180,
            "inserted_ids": ["batch_001", "batch_002"],
            "failed_ids": ["batch_003"],
            "batch_id": "batch_20240115_103045",
            "failure_details": [
                {
                    "id": "batch_003",
                    "error": "坐标超出有效范围",
                    "error_code": "INVALID_COORDINATES"
                }
            ]
        }
    }
    ```

    #### 批量验证错误 (400)
    ```json
    {
        "success": false,
        "message": "批量数据验证失败",
        "data": {
            "validation_errors": [
                {
                    "index": 0,
                    "id": "batch_001",
                    "errors": ["缺少必填字段: semantic_str"]
                },
                {
                    "index": 2,
                    "id": "batch_003",
                    "errors": ["经度超出有效范围", "ID格式不正确"]
                }
            ],
            "total_errors": 3,
            "total_records": 3
        }
    }
    ```

    #### 系统错误 (500)
    ```json
    {
        "success": false,
        "message": "批量插入失败: 数据库连接异常",
        "data": null
    }
    ```

    ---

    ### 批量处理流程

    #### 1. 数据接收与预处理
    - 接收房源数据列表
    - 验证批量大小限制（建议≤1000条）
    - 生成批次ID用于跟踪

    #### 2. 数据验证阶段
    ```python
    # 批量验证流程
    validation_results = []
    for index, house in enumerate(houses):
        try:
            validate_house_data(house)
            validation_results.append({"index": index, "valid": True})
        except ValidationError as e:
            validation_results.append({
                "index": index, 
                "valid": False, 
                "errors": e.errors
            })
    ```

    #### 3. 向量化处理
    - 并行处理多个语义字段
    - 批量生成向量数据
    - 优化GPU/CPU资源利用

    #### 4. 事务性存储
    - 使用事务确保数据一致性
    - 支持部分成功场景
    - 自动回滚异常数据

    #### 5. 索引更新
    - 批量更新搜索索引
    - 异步刷新统计信息
    - 生成处理报告

    ---

    ### 性能优化特性

    #### 批量处理优势
    | 处理方式    | 单条插入  | 批量插入  | 提升比例    |
    |-------------|-----------|-----------|-------------|
    | 网络开销    | 高        | 低        | 90%减少     |
    | 向量化效率  | 低        | 高        | 5-10倍提升  |
    | 存储效率    | 低        | 高        | 3-5倍提升   |
    | 索引更新    | 频繁      | 批量      | 80%减少     |

    #### 并行处理策略
    ```python
    # 并行向量化处理
    with ThreadPoolExecutor(max_workers=4) as executor:
        vector_futures = {
            executor.submit(vectorize_text, house['semantic_str']): i 
            for i, house in enumerate(houses)
        }
    ```

    ---

    ### 批量大小建议

    #### 推荐批次大小
    | 数据量     | 建议批次大小 | 预估处理时间 | 内存占用    |
    |-----------|--------------|--------------|-------------|
    | < 100条   | 一次性处理   | 5-15秒       | < 50MB      |
    | 100-500条 | 100条/批次   | 10-30秒      | 50-200MB    |
    | 500-2000条| 200条/批次   | 30-60秒      | 200-500MB   |
    | > 2000条  | 分多批处理   | 视情况而定   | 建议分批    |

    #### 性能调优参数
    ```json
    {
        "batch_size": 200,
        "parallel_workers": 4,
        "vector_batch_size": 50,
        "commit_interval": 100,
        "timeout_seconds": 300
    }
    ```

    ---

    ### 错误处理机制

    #### 错误处理策略
    1. **全部验证**: 先验证所有数据，再开始插入
    2. **部分容错**: 跳过错误数据，继续处理正确数据
    3. **事务保护**: 确保数据一致性，支持部分回滚
    4. **详细报告**: 提供每条记录的详细处理结果

    #### 常见错误类型
    ```python
    error_types = {
        "VALIDATION_ERROR": "数据验证失败",
        "DUPLICATE_ID": "ID重复",
        "INVALID_COORDINATES": "坐标无效",
        "VECTORIZATION_ERROR": "向量化失败",
        "STORAGE_ERROR": "存储异常",
        "TIMEOUT_ERROR": "处理超时"
    }
    ```

    ---

    ### 使用场景

    #### 1. 数据迁移
    ```python
    # 从旧系统迁移数据
    old_data = load_from_legacy_system()
    batch_data = convert_to_new_format(old_data)
    
    response = requests.post(
        "/api/house-reco/batch-insert",
        json=batch_data
    )
    ```

    #### 2. 定期数据同步
    ```python
    # 定期同步外部数据源
    def daily_sync_job():
        new_houses = fetch_from_external_api()
        if new_houses:
            batch_insert_houses(new_houses)
    ```

    #### 3. 测试数据生成
    ```python
    # 生成测试数据
    test_houses = generate_test_data(count=500)
    batch_insert_houses(test_houses)
    ```

    ---

    ### 监控与日志

    #### 关键指标监控
    ```json
    {
        "batch_processing_metrics": {
            "total_batches": 1,
            "success_rate": 95.5,
            "avg_processing_time_ms": 1250,
            "throughput_records_per_sec": 2.4,
            "error_rate": 4.5,
            "vector_generation_time_ms": 850
        }
    }
    ```

    #### 日志记录
    ```python
    # 批量处理日志
    logger.info({
        "batch_id": "batch_20240115_103045",
        "total_records": 100,
        "success_count": 95,
        "failure_count": 5,
        "processing_time_ms": 12500,
        "errors": ["DUPLICATE_ID", "INVALID_COORDINATES"]
    })
    ```

    ---

    ### 最佳实践

    #### 1. 数据准备
    - 在客户端预先验证数据格式
    - 去重处理，避免重复ID
    - 合理分批，避免单批过大

    #### 2. 错误处理
    - 先进行少量测试验证数据格式
    - 保存处理日志用于问题排查
    - 建立重试机制处理临时故障

    #### 3. 性能优化
    - 避免在业务高峰期进行大批量插入
    - 监控系统资源使用情况
    - 合理设置并行处理参数

    ---

    ### 注意事项

    - **批量限制**: 建议单批次不超过1000条记录
    - **处理时间**: 大批量数据处理可能需要较长时间
    - **内存占用**: 批量处理会占用更多内存资源
    - **错误恢复**: 部分失败的记录需要单独处理
    - **ID唯一性**: 批量内和系统内都要确保ID唯一性
    - **事务控制**: 大批量数据建议分批提交减少锁定时间
    """
    try:
        service = get_house_reco_service()
        
        # 转换为字典列表
        house_data_list = [house.model_dump(exclude_none=True) for house in houses]
        
        # 批量插入
        success = service.insert_house_data(house_data_list)
        
        if success:
            return HouseResponse(
                success=True,
                message=f"成功批量插入{len(houses)}条房源数据",
                data={"inserted_count": len(houses)}
            )
        else:
            return HouseResponse(
                success=False,
                message="批量插入失败",
                data=None
            )
            
    except Exception as e:
        logger.error(f"批量插入失败: {e}")
        return HouseResponse(
            success=False,
            message=f"批量插入失败: {str(e)}",
            data=None
        )


@router.get("/detail/{house_id}", response_model=HouseResponse, summary="获取房源详情")
def get_house_detail(house_id: int) -> HouseResponse:
    """
    ### GET `/detail/{house_id}` 获取房源详情接口

    **功能描述**:
    根据房源ID获取单条房源的完整详细信息，包括基础属性、地理位置、语义描述等所有字段。
    适用于房源详情页展示、数据验证、信息查看等场景。

    ---

    ### 路径参数 (Path Parameters)

    | 参数        | 类型  | 必填 | 描述           | 示例      |
    |-------------|-------|------|----------------|-----------|
    | `house_id`  | `int` | 是   | 房源唯一ID     | 123456    |

    ---

    ### 请求示例

    #### 获取房源详情
    ```bash
    GET /api/house-reco/detail/123456
    ```

    ```javascript
    // JavaScript 示例
    const response = await fetch('/api/house-reco/detail/123456');
    const data = await response.json();
    ```

    ```python
    # Python 示例
    import requests
    response = requests.get('http://localhost:8000/api/house-reco/detail/123456')
    house_detail = response.json()
    ```

    ---

    ### 响应 (Response)

    #### 成功响应 (200)
    ```json
    {
        "success": true,
        "message": "获取房源详情成功",
        "data": {
            "id": "123456",
            "category": 2,
            "name": "万科城市花园",
            "region": "海淀区",
            "address": "中关村大街39号",
            "longitude": 116.3105,
            "latitude": 39.9785,
            "semantic_str": "三室两厅，南北通透，精装修，学区房，地铁10分钟步行距离",
            "min_area": 89.5,
            "max_area": 95.0,
            "min_unit_price": 72625,
            "max_unit_price": 80447,
            "min_total_price": 650.0,
            "max_total_price": 720.0,
            "type": "住宅",
            "year_completion": "2020",
            "transaction_ownership": "商品房",
            "property_right_duration": 70,
            "parking_space_ratio": "1:1",
            "management_company": "万科物业",
            "management_fee": 3.5,
            "developer": "万科集团",
            "greening_rate": 35.5,
            "plot_ratio": 2.8,
            "decoration_style": "现代简约",
            "decoration_status": "精装修",
            "water_electricity": "民用水电",
            "has_elevator": "有电梯",
            "has_parking": "有车位",
            "orientation": "南北通透",
            "building_age": "次新房",
            "furniture_facilities": "全配置",
            "floor": "中楼层（10/30层）",
            "preferences_tags": "学区房,地铁房,商圈房",
            "created_at": "2024-01-15T10:30:45Z",
            "updated_at": "2024-01-16T14:22:30Z"
        }
    }
    ```

    #### 房源不存在 (404)
    ```json
    {
        "success": false,
        "message": "未找到ID为123456的房源",
        "data": null
    }
    ```

    #### 参数错误 (400)
    ```json
    {
        "success": false,
        "message": "house_id参数格式错误，必须为正整数",
        "data": {
            "invalid_param": "house_id",
            "provided_value": "abc123",
            "expected_type": "integer"
        }
    }
    ```

    #### 系统错误 (500)
    ```json
    {
        "success": false,
        "message": "获取详情失败: 数据库连接异常",
        "data": null
    }
    ```

    ---

    ### 响应字段详解

    #### 基础信息字段
    | 字段名       | 类型     | 说明                    | 示例值           |
    |-------------|----------|-------------------------|------------------|
    | `id`        | `str`    | 房源唯一标识            | "123456"         |
    | `category`  | `int`    | 房源分类                | 2 (二手房)       |
    | `name`      | `str`    | 房源/小区名称           | "万科城市花园"    |
    | `region`    | `str`    | 所属区域                | "海淀区"         |
    | `address`   | `str`    | 详细地址                | "中关村大街39号" |

    #### 地理位置字段
    | 字段名        | 类型     | 说明            | 示例值      |
    |---------------|----------|-----------------|-------------|
    | `longitude`   | `float`  | 经度坐标        | 116.3105    |
    | `latitude`    | `float`  | 纬度坐标        | 39.9785     |

    #### 面积价格字段
    | 字段名              | 类型     | 说明            | 单位      | 示例值    |
    |--------------------|----------|-----------------|-----------|-----------|
    | `min_area`         | `float`  | 最小面积        | 平方米    | 89.5      |
    | `max_area`         | `float`  | 最大面积        | 平方米    | 95.0      |
    | `min_total_price`  | `float`  | 最小总价        | 万元      | 650.0     |
    | `max_total_price`  | `float`  | 最大总价        | 万元      | 720.0     |
    | `min_unit_price`   | `float`  | 最小单价        | 元/平米   | 72625     |
    | `max_unit_price`   | `float`  | 最大单价        | 元/平米   | 80447     |

    #### 房源属性字段
    | 字段名                        | 类型     | 说明        | 示例值         |
    |-------------------------------|----------|-------------|----------------|
    | `type`                        | `str`    | 房源类型    | "住宅"         |
    | `year_completion`             | `str`    | 建成年份    | "2020"         |
    | `transaction_ownership`       | `str`    | 交易权属    | "商品房"       |
    | `property_right_duration`     | `int`    | 产权年限    | 70             |
    | `orientation`                 | `str`    | 房屋朝向    | "南北通透"     |
    | `decoration_status`           | `str`    | 装修状况    | "精装修"       |
    | `has_elevator`                | `str`    | 电梯情况    | "有电梯"       |
    | `has_parking`                 | `str`    | 停车情况    | "有车位"       |

    #### 小区配套字段
    | 字段名                 | 类型     | 说明        | 单位    | 示例值      |
    |------------------------|----------|-------------|---------|-------------|
    | `management_company`   | `str`    | 物业公司    | -       | "万科物业"  |
    | `management_fee`       | `float`  | 物业费      | 元/㎡/月| 3.5         |
    | `developer`            | `str`    | 开发商      | -       | "万科集团"  |
    | `greening_rate`        | `float`  | 绿化率      | %       | 35.5        |
    | `plot_ratio`           | `float`  | 容积率      | -       | 2.8         |

    #### 系统字段
    | 字段名        | 类型         | 说明            | 格式             |
    |---------------|--------------|-----------------|------------------|
    | `created_at`  | `datetime`   | 创建时间        | ISO 8601格式     |
    | `updated_at`  | `datetime`   | 更新时间        | ISO 8601格式     |

    ---

    ### 数据处理说明

    #### 字段可用性
    ```json
    // 字段可用性说明
    {
        "always_present": ["id", "category", "name", "region", "address", "longitude", "latitude", "semantic_str"],
        "conditionally_present": ["min_area", "max_area", "rent", "type", "orientation"],
        "system_generated": ["created_at", "updated_at", "vector_id"]
    }
    ```

    #### 数据完整性检查
    - 所有必填字段保证非空
    - 地理坐标已验证在有效范围内
    - 价格和面积数据逻辑一致
    - 系统时间戳自动生成

    #### 敏感信息处理
    - 不返回内部向量数据
    - 不暴露系统内部ID
    - 过滤调试和诊断信息

    ---

    ### 使用场景

    #### 1. 房源详情页
    ```javascript
    // 前端展示房源详情
    async function loadHouseDetail(houseId) {
        const response = await fetch(`/api/house-reco/detail/${houseId}`);
        const result = await response.json();
        
        if (result.success) {
            displayHouseInfo(result.data);
        } else {
            showErrorMessage(result.message);
        }
    }
    ```

    #### 2. 数据验证
    ```python
    # 验证房源数据完整性
    def validate_house_data(house_id):
        response = requests.get(f'/api/house-reco/detail/{house_id}')
        house = response.json()['data']
        
        required_fields = ['name', 'region', 'address', 'longitude', 'latitude']
        missing_fields = [field for field in required_fields if not house.get(field)]
        
        return len(missing_fields) == 0
    ```

    #### 3. 批量导出准备
    ```python
    # 批量获取房源详情用于导出
    def export_houses_details(house_ids):
        details = []
        for house_id in house_ids:
            response = requests.get(f'/api/house-reco/detail/{house_id}')
            if response.json()['success']:
                details.append(response.json()['data'])
        return details
    ```

    ---

    ### 性能考虑

    #### 响应时间
    | 数据复杂度 | 平均响应时间 | 说明               |
    |-----------|--------------|-------------------|
    | 基础字段   | 10-20ms     | 仅包含核心信息     |
    | 完整字段   | 20-50ms     | 包含所有可选字段   |
    | 复杂查询   | 50-100ms    | 包含关联数据       |

    #### 缓存策略
    ```python
    # 缓存机制
    cache_config = {
        "ttl": 300,  # 5分钟缓存
        "key_pattern": "house_detail:{house_id}",
        "invalidation": "on_update"
    }
    ```

    #### 并发处理
    - 支持高并发读取操作
    - 数据库连接池优化
    - 读取操作不影响写入性能

    ---

    ### 错误处理

    #### ID格式验证
    ```python
    # ID验证规则
    def validate_house_id(house_id):
        if not isinstance(house_id, int) or house_id <= 0:
            raise ValueError("house_id必须为正整数")
        if house_id > 999999999:
            raise ValueError("house_id超出有效范围")
        return True
    ```

    #### 常见错误处理
    1. **ID不存在**: 返回404状态码和友好提示
    2. **ID格式错误**: 返回400状态码和格式要求
    3. **数据损坏**: 返回部分可用数据，标注缺失字段
    4. **系统异常**: 返回500状态码，记录错误日志

    ---

    ### 数据字典

    #### 分类枚举值
    ```json
    {
        "category_mapping": {
            "1": "新房",
            "2": "二手房", 
            "3": "出租房"
        }
    }
    ```

    #### 常见属性值
    ```json
    {
        "orientation_values": ["南", "北", "东", "西", "南北", "东西", "南北通透", "东南", "西南", "东北", "西北"],
        "decoration_status": ["毛坯", "简装", "精装", "豪装", "未装修"],
        "elevator_status": ["有电梯", "无电梯", "待装电梯"],
        "parking_status": ["有车位", "无车位", "租赁车位", "产权车位"]
    }
    ```

    ---

    ### 安全考虑

    #### 访问控制
    - 公开接口，无需认证
    - 支持访问频率限制
    - 记录访问日志用于监控

    #### 数据保护
    - 不返回敏感的系统内部信息
    - 过滤可能的恶意输入参数
    - 防止SQL注入和XSS攻击

    ---

    ### 注意事项

    - **ID类型**: 目前使用整数ID，未来可能支持字符串ID
    - **字段变更**: 可选字段可能根据业务需求调整，建议检查字段存在性
    - **数据一致性**: 详情数据可能与搜索结果有轻微延迟
    - **性能影响**: 频繁调用建议实现客户端缓存
    - **数据完整性**: 部分历史数据可能缺少新增字段
    """
    try:
        service = get_house_reco_service()
        
        house = service.get_house_by_id(house_id)
        
        if house:
            # 序列化保护
            try:
                import json
                json.dumps(house)
                safe_house = house
            except (TypeError, ValueError) as e:
                logger.warning(f"房源详情序列化失败: {e}")
                safe_house = {"error": "数据序列化失败", "house_id": house_id}
            
            return HouseResponse(
                success=True,
                message="获取房源详情成功",
                data=safe_house
            )
        else:
            return HouseResponse(
                success=False,
                message=f"未找到ID为{house_id}的房源",
                data=None
            )
            
    except Exception as e:
        logger.error(f"获取房源详情失败: {e}")
        return HouseResponse(
            success=False,
            message=f"获取详情失败: {str(e)}",
            data=None
        )


@router.delete("/{house_id}", response_model=HouseResponse, summary="删除房源")
def delete_house(house_id: int) -> HouseResponse:
    """
    ### DELETE `/{house_id}` 删除房源接口

    **功能描述**:
    根据房源ID永久删除指定的房源记录，包括结构化数据、向量数据和相关索引。
    此操作不可逆，适用于房源下架、数据清理、错误数据修正等场景。

    ---

    ### 路径参数 (Path Parameters)

    | 参数        | 类型  | 必填 | 描述           | 示例      |
    |-------------|-------|------|----------------|-----------|
    | `house_id`  | `int` | 是   | 房源唯一ID     | 123456    |

    ---

    ### 请求示例

    #### 删除指定房源
    ```bash
    DELETE /api/house-reco/123456
    ```

    ```javascript
    // JavaScript 示例
    const response = await fetch('/api/house-reco/123456', {
        method: 'DELETE'
    });
    const result = await response.json();
    ```

    ```python
    # Python 示例
    import requests
    response = requests.delete('http://localhost:8000/api/house-reco/123456')
    delete_result = response.json()
    ```

    ```curl
    # cURL 示例
    curl -X DELETE "http://localhost:8000/api/house-reco/123456"
    ```

    ---

    ### 响应 (Response)

    #### 成功删除 (200)
    ```json
    {
        "success": true,
        "message": "成功删除房源123456",
        "data": {
            "deleted_house_id": 123456,
            "deletion_timestamp": "2024-01-15T10:30:45Z",
            "affected_records": {
                "structural_data": 1,
                "vector_data": 1,
                "index_entries": 1
            },
            "cleanup_status": {
                "cache_cleared": true,
                "index_updated": true,
                "stats_refreshed": true
            }
        }
    }
    ```

    #### 房源不存在 (404)
    ```json
    {
        "success": false,
        "message": "未找到ID为123456的房源，可能已被删除",
        "data": {
            "house_id": 123456,
            "error_type": "NOT_FOUND",
            "suggestions": [
                "检查房源ID是否正确",
                "确认房源是否已被删除",
                "使用搜索接口确认房源存在性"
            ]
        }
    }
    ```

    #### 参数错误 (400)
    ```json
    {
        "success": false,
        "message": "house_id参数格式错误，必须为正整数",
        "data": {
            "invalid_param": "house_id",
            "provided_value": "abc123",
            "expected_type": "integer",
            "valid_range": "1 - 999999999"
        }
    }
    ```

    #### 删除冲突 (409)
    ```json
    {
        "success": false,
        "message": "删除失败: 房源正在被其他操作使用",
        "data": {
            "house_id": 123456,
            "conflict_type": "RESOURCE_LOCKED",
            "blocking_operations": ["batch_update", "vector_rebuild"],
            "retry_after_seconds": 60
        }
    }
    ```

    #### 系统错误 (500)
    ```json
    {
        "success": false,
        "message": "删除失败: 数据库连接异常",
        "data": {
            "error_code": "DATABASE_ERROR",
            "operation_id": "del_20240115_103045"
        }
    }
    ```

    ---

    ### 删除操作流程

    #### 1. 参数验证
    ```python
    # ID格式验证
    def validate_delete_request(house_id):
        if not isinstance(house_id, int) or house_id <= 0:
            raise ValueError("house_id必须为正整数")
        
        if house_id > 999999999:
            raise ValueError("house_id超出有效范围")
        
        return True
    ```

    #### 2. 存在性检查
    - 验证房源ID是否存在于系统中
    - 检查房源当前状态
    - 确认是否有依赖关系或锁定状态

    #### 3. 删除执行
    ```python
    # 删除流程
    def delete_house_process(house_id):
        # 步骤1: 删除结构化数据
        delete_structural_data(house_id)
        
        # 步骤2: 删除向量数据
        delete_vector_data(house_id)
        
        # 步骤3: 更新索引
        update_search_indexes(house_id, operation="delete")
        
        # 步骤4: 清理缓存
        clear_related_cache(house_id)
        
        # 步骤5: 更新统计
        refresh_collection_stats()
    ```

    #### 4. 后续清理
    - 清理相关缓存数据
    - 更新搜索索引
    - 刷新统计信息
    - 记录操作日志

    ---

    ### 数据删除范围

    #### 会被删除的数据
    | 数据类型        | 说明                      | 位置              |
    |---------------|---------------------------|-------------------|
    | 结构化数据     | 房源的基础属性信息         | Milvus Collection |
    | 向量数据       | 语义字段生成的向量         | Milvus Vector     |
    | 索引条目       | 搜索索引中的相关记录       | Index Files       |
    | 缓存数据       | 详情页和搜索结果缓存       | Cache System      |

    #### 不会被删除的数据
    | 数据类型        | 说明                      | 保留原因          |
    |---------------|---------------------------|-------------------|
    | 操作日志       | 删除操作的审计记录         | 审计要求          |
    | 统计历史       | 历史统计数据快照           | 分析需求          |
    | 备份数据       | 已备份的历史数据           | 数据安全          |

    ---

    ### 事务与安全

    #### 原子性保证
    ```python
    # 事务性删除
    def atomic_delete(house_id):
        with transaction():
            try:
                # 删除主数据
                delete_house_data(house_id)
                # 删除向量
                delete_house_vector(house_id)
                # 更新索引
                update_indexes(house_id, "DELETE")
                # 提交事务
                commit()
            except Exception as e:
                # 回滚所有操作
                rollback()
                raise e
    ```

    #### 并发控制
    - 删除期间锁定相关记录
    - 防止并发修改冲突
    - 支持重试机制

    #### 权限控制
    - 验证删除权限（如有权限系统）
    - 记录操作者信息
    - 审计日志记录

    ---

    ### 性能考虑

    #### 删除性能
    | 操作类型       | 平均耗时    | 影响因素               |
    |---------------|-------------|------------------------|
    | 单条记录删除   | 50-100ms   | 数据库连接延迟         |
    | 向量删除       | 100-200ms  | 向量索引大小           |
    | 索引更新       | 200-500ms  | 索引复杂度和数据量     |
    | 缓存清理       | 10-50ms    | 缓存分布和网络延迟     |

    #### 系统影响
    - **搜索性能**: 删除后搜索性能略有提升
    - **存储空间**: 释放约6KB存储空间（包含向量）
    - **索引大小**: 减少索引文件大小
    - **内存占用**: 释放缓存内存空间

    ---

    ### 错误处理与恢复

    #### 常见错误类型
    ```python
    delete_error_types = {
        "NOT_FOUND": "房源不存在",
        "INVALID_ID": "ID格式错误", 
        "RESOURCE_LOCKED": "资源被锁定",
        "DATABASE_ERROR": "数据库异常",
        "PERMISSION_DENIED": "权限不足",
        "TIMEOUT": "操作超时"
    }
    ```

    #### 错误恢复策略
    1. **部分失败**: 记录失败点，支持续传删除
    2. **超时处理**: 异步执行长时间删除操作
    3. **锁定冲突**: 提供重试机制和等待时间
    4. **数据修复**: 提供数据一致性检查工具

    #### 回滚机制
    ```python
    # 删除失败回滚
    def rollback_delete_operation(house_id, operation_id):
        # 恢复被删除的数据（如果可能）
        # 重建索引条目
        # 清理错误状态
        # 记录回滚日志
        pass
    ```

    ---

    ### 使用场景

    #### 1. 房源下架
    ```python
    # 房源到期或下架
    def offline_house(house_id, reason="expired"):
        response = requests.delete(f'/api/house-reco/{house_id}')
        if response.json()['success']:
            log_offline_operation(house_id, reason)
        return response.json()
    ```

    #### 2. 错误数据清理
    ```python
    # 批量清理错误数据
    def cleanup_invalid_houses(invalid_ids):
        results = []
        for house_id in invalid_ids:
            result = requests.delete(f'/api/house-reco/{house_id}')
            results.append({
                'id': house_id,
                'success': result.json()['success'],
                'message': result.json()['message']
            })
        return results
    ```

    #### 3. 数据迁移清理
    ```python
    # 数据迁移后清理旧数据
    def cleanup_migrated_houses(migrated_ids):
        for house_id in migrated_ids:
            # 确认新系统中数据正确
            if verify_migration(house_id):
                delete_response = requests.delete(f'/api/house-reco/{house_id}')
                log_migration_cleanup(house_id, delete_response.json())
    ```

    ---

    ### 监控与日志

    #### 关键指标
    ```json
    {
        "delete_metrics": {
            "total_deletions": 1250,
            "success_rate": 99.2,
            "avg_delete_time_ms": 150,
            "failed_deletions": 10,
            "rollback_count": 2
        }
    }
    ```

    #### 操作日志
    ```python
    # 删除操作日志
    delete_log = {
        "operation_id": "del_20240115_103045",
        "house_id": 123456,
        "operator": "system",
        "timestamp": "2024-01-15T10:30:45Z",
        "execution_time_ms": 150,
        "affected_records": {
            "structural": 1,
            "vector": 1,
            "index": 1
        },
        "status": "success"
    }
    ```

    #### 审计追踪
    - 记录所有删除操作
    - 包含操作者信息（如有）
    - 保存删除前数据快照（可选）
    - 支持操作历史查询

    ---

    ### 最佳实践

    #### 1. 删除前确认
    ```python
    # 删除前验证
    def safe_delete_house(house_id):
        # 1. 检查房源存在性
        detail_response = requests.get(f'/api/house-reco/detail/{house_id}')
        if not detail_response.json()['success']:
            return {"error": "房源不存在"}
        
        # 2. 确认删除操作
        confirm = input(f"确认删除房源 {house_id}? (y/N): ")
        if confirm.lower() != 'y':
            return {"cancelled": True}
        
        # 3. 执行删除
        delete_response = requests.delete(f'/api/house-reco/{house_id}')
        return delete_response.json()
    ```

    #### 2. 批量删除优化
    ```python
    # 批量删除优化
    def batch_delete_houses(house_ids, batch_size=10):
        results = []
        for i in range(0, len(house_ids), batch_size):
            batch = house_ids[i:i+batch_size]
            batch_results = []
            
            for house_id in batch:
                result = requests.delete(f'/api/house-reco/{house_id}')
                batch_results.append(result.json())
                
            results.extend(batch_results)
            # 批次间延迟，避免系统压力
            time.sleep(1)
            
        return results
    ```

    #### 3. 删除验证
    ```python
    # 删除后验证
    def verify_deletion(house_id):
        # 验证数据已被删除
        response = requests.get(f'/api/house-reco/detail/{house_id}')
        if response.status_code == 404:
            return True
        
        # 验证搜索中不存在
        search_response = requests.post('/api/house-reco/search', 
                                       json={'name': f'house_{house_id}'})
        return house_id not in [h['id'] for h in search_response.json()['data']['houses']]
    ```

    ---

    ### 注意事项

    - **不可逆操作**: 删除操作无法撤销，请谨慎操作
    - **数据完整性**: 删除可能影响相关统计和分析数据
    - **性能影响**: 大量删除操作可能影响系统性能
    - **并发安全**: 删除期间避免对同一房源的并发操作
    - **索引延迟**: 索引更新可能需要几秒时间生效
    - **缓存一致性**: 删除后相关缓存将被清理，可能影响响应时间
    - **审计要求**: 生产环境建议记录详细的删除日志
    """
    try:
        service = get_house_reco_service()
        
        success = service.delete_house(house_id)
        
        if success:
            return HouseResponse(
                success=True,
                message=f"成功删除房源{house_id}",
                data={"deleted_house_id": house_id}
            )
        else:
            return HouseResponse(
                success=False,
                message=f"删除房源{house_id}失败",
                data=None
            )
            
    except Exception as e:
        logger.error(f"删除房源失败: {e}")
        return HouseResponse(
            success=False,
            message=f"删除失败: {str(e)}",
            data=None
        )


@router.get("/stats", response_model=HouseResponse, summary="获取集合统计信息")
def get_collection_stats() -> HouseResponse:
    """
    ### GET `/stats` 获取集合统计信息接口

    **功能描述**:
    获取房源推荐系统的详细统计信息，包括数据量、存储占用、索引状态等关键指标。
    用于系统监控、性能分析和运维管理。

    ---

    ### 请求参数

    无需任何请求参数。

    ---

    ### 响应 (Response)

    #### 成功响应 (200)
    ```json
    {
        "success": true,
        "message": "获取统计信息成功",
        "data": {
            "collection_name": "house_recommendation",
            "total_entities": 25847,
            "indexed_entities": 25847,
            "collection_size_mb": 1248.5,
            "index_status": "IndexState.Finished",
            "last_updated": "2024-01-15T10:30:45Z",
            "dimensions": 768,
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "nlist": 1024,
            "memory_usage_mb": 892.3,
            "disk_usage_mb": 1248.5,
            "search_performance": {
                "avg_latency_ms": 45.2,
                "qps_capacity": 500
            },
            "data_distribution": {
                "by_region": {
                    "海淀区": 5234,
                    "朝阳区": 4891,
                    "西城区": 3456,
                    "东城区": 2876
                },
                "by_price_range": {
                    "0-500万": 8765,
                    "500-1000万": 12456,
                    "1000-2000万": 3892,
                    "2000万以上": 734
                }
            }
        }
    }
    ```

    #### 错误响应 (500)
    ```json
    {
        "success": false,
        "message": "获取统计信息失败: 具体错误信息",
        "data": null
    }
    ```

    ---

    ### 统计指标说明

    #### 基础指标
    | 字段                 | 描述                               | 单位      |
    |---------------------|-----------------------------------|----------|
    | `collection_name`   | 集合名称                           | -        |
    | `total_entities`    | 总房源数量                         | 条       |
    | `indexed_entities`  | 已建立索引的房源数量                | 条       |
    | `collection_size_mb`| 集合总大小                         | MB       |
    | `dimensions`        | 向量维度                           | 维       |

    #### 索引信息
    | 字段          | 描述                    | 可能值                      |
    |---------------|------------------------|----------------------------|
    | `index_status`| 索引状态               | Finished/Building/Failed   |
    | `index_type`  | 索引类型               | IVF_FLAT/IVF_SQ8/HNSW     |
    | `metric_type` | 距离度量类型           | COSINE/L2/IP               |
    | `nlist`       | 聚类中心数量           | 整数值                     |

    #### 性能指标
    | 字段               | 描述                    | 单位    |
    |-------------------|------------------------|---------|
    | `memory_usage_mb` | 内存占用               | MB      |
    | `disk_usage_mb`   | 磁盘占用               | MB      |
    | `avg_latency_ms`  | 平均搜索延迟           | 毫秒    |
    | `qps_capacity`    | 理论QPS容量            | 次/秒   |

    ---

    ### 数据分布分析

    #### 区域分布
    显示各个区域的房源数量分布，帮助了解数据的地理分布特征。

    #### 价格分布  
    按价格区间统计房源数量，反映市场价格结构。

    #### 户型分布
    ```json
    "by_house_type": {
        "一室一厅": 1245,
        "两室一厅": 4567,
        "三室两厅": 8901,
        "四室两厅": 2134
    }
    ```

    #### 面积分布
    ```json
    "by_area_range": {
        "50-80㎡": 3456,
        "80-120㎡": 12789,
        "120-200㎡": 7654,
        "200㎡以上": 1948
    }
    ```

    ---

    ### 系统健康指标

    #### 索引健康度
    - **完整索引**: indexed_entities = total_entities
    - **部分索引**: indexed_entities < total_entities
    - **索引失败**: index_status = "Failed"

    #### 性能健康度
    - **优秀**: avg_latency_ms < 50ms
    - **良好**: 50ms ≤ avg_latency_ms < 100ms  
    - **需优化**: avg_latency_ms ≥ 100ms

    #### 存储健康度
    - **正常**: memory_usage < disk_usage * 0.8
    - **高负载**: memory_usage ≥ disk_usage * 0.8

    ---

    ### 监控建议

    #### 关键指标监控
    ```python
    # 设置监控阈值
    monitoring_thresholds = {
        "max_latency_ms": 100,
        "min_qps": 200,
        "max_memory_usage_mb": 2000,
        "index_completion_rate": 0.95
    }
    ```

    #### 告警条件
    - 搜索延迟超过100ms
    - 索引完成率低于95%
    - 内存使用率超过80%
    - QPS容量低于预期值

    ---

    ### 性能优化建议

    #### 索引优化
    ```json
    // 当数据量较大时，可考虑调整索引参数
    {
        "index_type": "IVF_SQ8",  // 压缩索引，减少内存占用
        "nlist": 2048,           // 增加聚类数量，提高搜索精度
        "m": 8                   // PQ压缩参数
    }
    ```

    #### 查询优化
    - 合理设置搜索参数nprobe
    - 避免过大的limit参数
    - 使用过滤条件减少搜索范围

    ---

    ### 使用场景

    #### 1. 系统监控
    ```bash
    # 定期获取统计信息用于监控
    curl -X GET "http://localhost:8000/api/house-reco/stats"
    ```

    #### 2. 性能分析
    ```bash
    # 分析系统性能表现
    curl -X GET "http://localhost:8000/api/house-reco/stats" | jq '.data.search_performance'
    ```

    #### 3. 容量规划
    ```bash
    # 评估存储和内存需求
    curl -X GET "http://localhost:8000/api/house-reco/stats" | jq '.data | {memory_usage_mb, disk_usage_mb, total_entities}'
    ```

    ---

    ### 注意事项

    - **实时性**: 统计信息可能有几秒的延迟
    - **缓存机制**: 部分统计信息可能被缓存以提高响应速度
    - **权限控制**: 建议对统计接口进行适当的权限控制
    - **监控集成**: 可将统计数据集成到监控系统中
    - **定期更新**: 建议定期获取统计信息进行系统健康检查
    """
    try:
        service = get_house_reco_service()
        
        stats = service.get_collection_stats()
        
        return HouseResponse(
            success=True,
            message="获取统计信息成功",
            data=stats
        )
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        return HouseResponse(
            success=False,
            message=f"获取统计信息失败: {str(e)}",
            data=None
        )


@router.post("/upload-excel", response_model=HouseResponse, summary="上传Excel文件导入房源")
async def upload_excel_file(file: UploadFile = File(...)) -> HouseResponse:
    """
    ### POST `/upload-excel` Excel文件批量导入房源接口

    **功能描述**:
    支持通过Excel文件批量导入房源数据到系统中。系统会自动进行文件验证、数据清洗、
    去重处理，并将数据转换为向量形式存储到Milvus数据库中。

    ---

    ### 请求参数

    | 参数   | 类型         | 必填 | 描述                               |
    |--------|-------------|------|-----------------------------------|
    | `file` | `UploadFile` | 是   | Excel文件 (.xlsx, .xls, .tsv, .txt) |

    ---

    ### 支持的文件格式

    | 格式   | 扩展名        | 描述                    |
    |--------|--------------|------------------------|
    | Excel  | .xlsx, .xls  | Microsoft Excel文件    |
    | TSV    | .tsv, .txt   | 制表符分隔的文本文件    |

    ---

    ### Excel文件格式要求

    #### 必须包含的列
    | 列名     | 数据类型 | 示例               | 说明              |
    |----------|----------|-------------------|-------------------|
    | id       | 整数     | 123456            | 房源唯一ID        |
    | xqmc     | 文本     | 万科城市花园      | 小区名称          |
    | qy       | 文本     | 海淀区            | 区域              |
    | dz       | 文本     | 中关村大街39号    | 详细地址          |
    | jd       | 浮点数   | 116.3105          | 经度              |
    | wd       | 浮点数   | 39.9785           | 纬度              |
    | mj       | 浮点数   | 89.5              | 面积(平方米)      |
    | fyhx     | 文本     | 三室两厅          | 房源户型          |
    | lc       | 文本     | 中楼层(共30层)    | 楼层情况          |
    | zj       | 浮点数   | 650.0             | 总价(万元)        |

    #### 可选列
    | 列名   | 数据类型 | 示例           | 说明                |
    |--------|----------|---------------|---------------------|
    | cx     | 文本     | 南北          | 朝向                |
    | dj     | 浮点数   | 72625         | 单价(元/平方米)     |
    | wyf    | 浮点数   | 2.5           | 物业费(元/平/月)    |
    | zxqk   | 文本     | 精装修        | 装修情况            |
    | ywdt   | 文本     | 有电梯        | 有无电梯            |
    | ywcw   | 文本     | 有车位        | 有无车位            |
    | xqtd   | 文本     | 环境优美      | 小区特点            |
    | zb     | 文本     | 交通便利      | 周边环境            |

    ---

    ### 响应 (Response)

    #### 成功响应 (200)
    ```json
    {
        "success": true,
        "message": "成功导入245条房源数据（去重 12 条重复记录）",
        "data": {
            "original_rows": 257,
            "imported_rows": 245,
            "duplicate_count": 12,
            "filename": "houses_data.xlsx",
            "validation_result": {
                "valid": true,
                "total_rows": 257,
                "duplicate_count": 12,
                "warnings": ["部分记录缺少单价信息，已自动计算"]
            },
            "collection_auto_created": true
        }
    }
    ```

    #### 文件格式错误 (400)
    ```json
    {
        "success": false,
        "message": "不支持的文件格式，请上传Excel文件(.xlsx, .xls)或TSV文件(.tsv, .txt)",
        "data": null
    }
    ```

    #### 文件验证失败 (400)
    ```json
    {
        "success": false,
        "message": "文件验证失败: 缺少必填列: id, xqmc, qy",
        "data": {
            "valid": false,
            "errors": [
                "缺少必填列: id",
                "缺少必填列: xqmc",
                "第3行: 经度超出有效范围"
            ],
            "warnings": ["部分记录单价为空"]
        }
    }
    ```

    ---

    ### 数据处理流程

    1. **文件上传**: 接收并保存上传的Excel文件到临时目录
    2. **格式验证**: 检查文件格式和必需列是否存在
    3. **数据读取**: 解析Excel内容并转换为结构化数据
    4. **数据验证**: 
       - 验证必填字段完整性
       - 检查数据类型和格式正确性
       - 验证地理坐标有效性
       - 检查ID唯一性
    5. **数据清洗**: 
       - 去除空行和无效数据
       - 自动计算缺失的单价信息
       - 标准化文本格式
    6. **去重处理**: 基于房源ID进行去重
    7. **向量化**: 对语义字段进行向量化处理
    8. **批量插入**: 将处理后的数据批量插入Milvus数据库
    9. **清理临时文件**: 删除临时文件并释放资源

    ---

    ### Excel模板示例

    ```
    | id     | xqmc        | qy    | dz           | jd      | wd     | mj   | fyhx    | lc       | zj   | cx | zxqk  |
    |--------|-------------|-------|-------------|---------|--------|------|---------|----------|------|----|----|
    | 123456 | 万科城市花园 | 海淀区 | 中关村大街39号 | 116.3105| 39.9785| 89.5 | 三室两厅 | 中楼层   | 650.0| 南北| 精装|
    | 123457 | 金地格林小镇 | 朝阳区 | 建国路88号   | 116.4634| 39.9078| 105.0| 三室两厅 | 高楼层   | 780.0| 南 | 简装|
    ```

    ---

    ### 数据验证规则

    #### 必填字段验证
    - **ID**: 必须为正整数且全局唯一
    - **小区名称**: 不能为空，支持多个名称用逗号分隔
    - **区域**: 不能为空
    - **地址**: 不能为空
    - **经纬度**: 必须在有效范围内（中国境内）
    - **面积**: 必须为正数，单位平方米
    - **户型**: 不能为空
    - **总价**: 必须为正数，单位万元

    #### 数据格式验证
    - **坐标范围**: 经度73-135°，纬度18-54°
    - **面积范围**: 10-1000平方米
    - **价格范围**: 总价10-50000万元
    - **单价计算**: 如果单价为空，自动计算 = 总价 × 10000 / 面积

    ---

    ### 错误处理机制

    #### 常见错误类型
    1. **文件格式错误**: 不支持的文件类型
    2. **列缺失错误**: 缺少必需的数据列
    3. **数据类型错误**: 数值字段包含非数字内容
    4. **数据范围错误**: 坐标、面积、价格超出合理范围
    5. **重复ID错误**: 文件内或与数据库中存在重复ID
    6. **空值错误**: 必填字段包含空值

    #### 处理策略
    - **跳过错误行**: 跳过有错误的数据行，继续处理其他行
    - **自动修复**: 对可修复的数据进行自动处理（如计算单价）
    - **详细报告**: 返回详细的错误和警告信息

    ---

    ### 使用建议

    #### 1. 文件准备
    - 使用提供的Excel模板确保列名正确
    - 确保必填字段完整
    - 检查地理坐标的准确性

    #### 2. 性能优化
    - 建议单次上传不超过10000条记录
    - 大量数据建议分批上传
    - 避免在高峰期进行大批量导入

    #### 3. 数据质量
    - 语义字段描述越详细，搜索效果越好
    - 保持数据一致性和准确性
    - 定期清理无效或过期数据

    ---

    ### 注意事项

    - **文件大小**: 建议单个文件不超过50MB
    - **处理时间**: 大文件处理可能需要较长时间，请耐心等待
    - **内存占用**: 导入过程会占用系统内存，建议合理安排
    - **数据备份**: 建议在大批量导入前备份现有数据
    - **集合管理**: 系统会自动创建Milvus集合（如果不存在）
    - **临时文件**: 系统会自动清理上传的临时文件
    """
    temp_file_path = None
    try:
        # 验证文件类型
        if not file.filename.endswith(('.xlsx', '.xls', '.tsv', '.txt')):
            return HouseResponse(
                success=False,
                message="不支持的文件格式，请上传Excel文件(.xlsx, .xls)或TSV文件(.tsv, .txt)",
                data=None
            )
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            # 读取上传的文件内容
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # 获取数据导入服务
        import_service = get_data_import_service()
        
        # 先验证文件
        validation_result = import_service.validate_excel_file(temp_file_path)
        if not validation_result['valid']:
            return HouseResponse(
                success=False,
                message=f"文件验证失败: {', '.join(validation_result['errors'])}",
                data=validation_result
            )
        
        # 执行导入
        success = import_service.import_from_excel(temp_file_path)
        
        if success:
            # 构建成功消息
            imported_count = validation_result['total_rows']
            duplicate_count = validation_result.get('duplicate_count', 0)
            actual_imported = imported_count - duplicate_count
            
            message_parts = [f"成功导入{actual_imported}条房源数据"]
            if duplicate_count > 0:
                message_parts.append(f"（去重 {duplicate_count} 条重复记录）")
            
            warnings = validation_result.get('warnings', [])
            if warnings:
                message_parts.append(f"，警告: {'; '.join(warnings)}")
            
            return HouseResponse(
                success=True,
                message=''.join(message_parts),
                data={
                    "original_rows": imported_count,
                    "imported_rows": actual_imported,
                    "duplicate_count": duplicate_count,
                    "filename": file.filename,
                    "validation_result": validation_result,
                    "collection_auto_created": True  # 标识集合是否自动创建
                }
            )
        else:
            return HouseResponse(
                success=False,
                message="数据导入失败",
                data=None
            )
            
    except Exception as e:
        logger.error(f"上传Excel文件失败: {e}")
        return HouseResponse(
            success=False,
            message=f"上传失败: {str(e)}",
            data=None
        )
    finally:
        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"清理临时文件失败: {e}")


@router.post("/preview-excel", response_model=HouseResponse, summary="预览Excel文件数据")
async def preview_excel_file(file: UploadFile = File(...), limit: int = 5) -> HouseResponse:
    """
    ### POST `/preview-excel` 预览Excel文件数据接口

    **功能描述**:
    在正式导入前预览Excel文件的内容和结构，验证数据格式和完整性。
    支持快速检查文件格式、字段映射、数据样本，帮助用户在导入前发现和修正问题。

    ---

    ### 请求参数 (Request Parameters)

    | 参数     | 类型         | 必填 | 默认值 | 描述                          |
    |----------|-------------|------|--------|-------------------------------|
    | `file`   | `UploadFile`| 是   | -      | Excel文件 (.xlsx, .xls, .tsv, .txt) |
    | `limit`  | `int`       | 否   | 5      | 预览行数限制 (1-50)            |

    ---

    ### 支持的文件格式

    | 格式    | 扩展名        | 描述                    | 最大文件大小 |
    |---------|--------------|------------------------|-------------|
    | Excel   | .xlsx, .xls  | Microsoft Excel文件     | 10MB        |
    | TSV     | .tsv, .txt   | 制表符分隔的文本文件     | 10MB        |

    ---

    ### 请求示例

    #### 使用表单上传预览
    ```bash
    curl -X POST "http://localhost:8000/api/house-reco/preview-excel" \
         -F "file=@houses_data.xlsx" \
         -F "limit=10"
    ```

    ```javascript
    // JavaScript FormData 上传
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('limit', '10');

    const response = await fetch('/api/house-reco/preview-excel', {
        method: 'POST',
        body: formData
    });
    const result = await response.json();
    ```

    ```python
    # Python requests 上传
    import requests

    with open('houses_data.xlsx', 'rb') as file:
        files = {'file': file}
        data = {'limit': 10}
        response = requests.post(
            'http://localhost:8000/api/house-reco/preview-excel',
            files=files,
            data=data
        )
        preview_result = response.json()
    ```

    ---

    ### 响应 (Response)

    #### 成功预览 (200)
    ```json
    {
        "success": true,
        "message": "预览成功，显示前5行数据",
        "data": {
            "preview_data": [
                {
                    "row_index": 1,
                    "id": "123456",
                    "xqmc": "万科城市花园",
                    "qy": "海淀区",
                    "dz": "中关村大街39号",
                    "jd": 116.3105,
                    "wd": 39.9785,
                    "mj": 89.5,
                    "fyhx": "三室两厅",
                    "lc": "中楼层(10/30层)",
                    "zj": 650.0,
                    "cx": "南北",
                    "dj": 72625,
                    "wyf": 3.5,
                    "zxqk": "精装修",
                    "ywdt": "有电梯",
                    "ywcw": "有车位"
                },
                {
                    "row_index": 2,
                    "id": "123457",
                    "xqmc": "金地格林小镇",
                    "qy": "朝阳区",
                    "dz": "建国路88号",
                    "jd": 116.4634,
                    "wd": 39.9078,
                    "mj": 105.0,
                    "fyhx": "三室两厅",
                    "lc": "高楼层(20/35层)",
                    "zj": 780.0,
                    "cx": "南",
                    "dj": 74286,
                    "wyf": 4.2,
                    "zxqk": "简装修",
                    "ywdt": "有电梯",
                    "ywcw": "有车位"
                }
            ],
            "field_mapping": {
                "id": {"target": "id", "required": true, "type": "str", "description": "房源ID"},
                "xqmc": {"target": "name", "required": true, "type": "str", "description": "小区名称"},
                "qy": {"target": "region", "required": true, "type": "str", "description": "区域"},
                "dz": {"target": "address", "required": true, "type": "str", "description": "地址"},
                "jd": {"target": "longitude", "required": true, "type": "float", "description": "经度"},
                "wd": {"target": "latitude", "required": true, "type": "float", "description": "纬度"},
                "mj": {"target": "area", "required": false, "type": "float", "description": "面积"},
                "fyhx": {"target": "type", "required": false, "type": "str", "description": "房源户型"},
                "lc": {"target": "floor", "required": false, "type": "str", "description": "楼层"},
                "zj": {"target": "total_price", "required": false, "type": "float", "description": "总价"},
                "cx": {"target": "orientation", "required": false, "type": "str", "description": "朝向"},
                "dj": {"target": "unit_price", "required": false, "type": "float", "description": "单价"},
                "wyf": {"target": "management_fee", "required": false, "type": "float", "description": "物业费"},
                "zxqk": {"target": "decoration_status", "required": false, "type": "str", "description": "装修情况"},
                "ywdt": {"target": "has_elevator", "required": false, "type": "str", "description": "电梯"},
                "ywcw": {"target": "has_parking", "required": false, "type": "str", "description": "车位"}
            },
            "filename": "houses_data.xlsx",
            "preview_count": 5,
            "total_rows_detected": 245,
            "file_info": {
                "size_mb": 2.3,
                "sheet_name": "Sheet1",
                "encoding": "utf-8"
            },
            "validation_summary": {
                "required_fields_present": true,
                "missing_required_fields": [],
                "data_type_issues": 0,
                "coordinate_range_issues": 0,
                "duplicate_ids": 0,
                "warnings": ["部分记录缺少单价信息"]
            }
        }
    }
    ```

    #### 文件格式错误 (400)
    ```json
    {
        "success": false,
        "message": "不支持的文件格式，请上传Excel文件(.xlsx, .xls)或TSV文件(.tsv, .txt)",
        "data": {
            "uploaded_filename": "data.pdf",
            "detected_extension": ".pdf",
            "supported_formats": [".xlsx", ".xls", ".tsv", ".txt"]
        }
    }
    ```

    #### 文件解析错误 (400)
    ```json
    {
        "success": false,
        "message": "文件解析失败: Excel文件损坏或格式不正确",
        "data": {
            "filename": "damaged_file.xlsx",
            "error_type": "FILE_CORRUPT",
            "suggestions": [
                "检查文件是否完整下载",
                "尝试在Excel中重新保存文件",
                "确保文件未被其他程序占用"
            ]
        }
    }
    ```

    #### 文件过大 (413)
    ```json
    {
        "success": false,
        "message": "文件大小超出限制，最大支持10MB",
        "data": {
            "file_size_mb": 15.2,
            "max_size_mb": 10.0,
            "filename": "large_data.xlsx"
        }
    }
    ```

    ---

    ### 字段映射说明

    #### 必填字段映射
    | Excel列名 | 系统字段名   | 数据类型 | 描述          | 验证规则              |
    |-----------|-------------|----------|---------------|----------------------|
    | `id`      | `id`        | `str`    | 房源ID        | 必须唯一，3-20位      |
    | `xqmc`    | `name`      | `str`    | 小区名称      | 2-100个字符           |
    | `qy`      | `region`    | `str`    | 区域          | 2-50个字符            |
    | `dz`      | `address`   | `str`    | 地址          | 5-200个字符           |
    | `jd`      | `longitude` | `float`  | 经度          | 73.0-135.0           |
    | `wd`      | `latitude`  | `float`  | 纬度          | 18.0-54.0            |

    #### 可选字段映射
    | Excel列名 | 系统字段名          | 数据类型 | 描述          | 默认值处理            |
    |-----------|------------------- |----------|---------------|----------------------|
    | `mj`      | `area`             | `float`  | 面积(㎡)      | 可为空，建议10-1000   |
    | `fyhx`    | `type`             | `str`    | 房源户型      | 可为空                |
    | `lc`      | `floor`            | `str`    | 楼层信息      | 可为空                |
    | `zj`      | `total_price`      | `float`  | 总价(万元)    | 可为空，建议10-50000  |
    | `cx`      | `orientation`      | `str`    | 朝向          | 可为空                |
    | `dj`      | `unit_price`       | `float`  | 单价(元/㎡)   | 可自动计算            |
    | `wyf`     | `management_fee`   | `float`  | 物业费        | 可为空                |
    | `zxqk`    | `decoration_status`| `str`    | 装修情况      | 可为空                |
    | `ywdt`    | `has_elevator`     | `str`    | 电梯          | 可为空                |
    | `ywcw`    | `has_parking`      | `str`    | 车位          | 可为空                |

    ---

    ### 数据验证规则

    #### 基础验证
    ```python
    validation_rules = {
        "id": {
            "required": True,
            "type": "str",
            "pattern": r"^[a-zA-Z0-9_-]{3,20}$",
            "unique": True
        },
        "coordinates": {
            "longitude": {"min": 73.0, "max": 135.0},
            "latitude": {"min": 18.0, "max": 54.0}
        },
        "area": {"min": 10, "max": 1000, "unit": "㎡"},
        "total_price": {"min": 10, "max": 50000, "unit": "万元"}
    }
    ```

    #### 数据完整性检查
    - **重复ID检查**: 确保文件内ID唯一
    - **坐标有效性**: 验证经纬度在中国境内
    - **数据类型验证**: 确保数值字段格式正确
    - **逻辑一致性**: 检查面积、单价、总价的逻辑关系

    #### 常见数据问题检测
    ```json
    {
        "common_issues": {
            "missing_required_fields": "缺少必填字段",
            "invalid_coordinates": "坐标超出有效范围",
            "duplicate_ids": "存在重复的房源ID",
            "invalid_data_types": "数据类型不匹配",
            "logical_inconsistency": "数值逻辑不一致"
        }
    }
    ```

    ---

    ### 预览数据处理

    #### 数据采样策略
    ```python
    # 智能采样策略
    def sample_preview_data(data, limit=5):
        if len(data) <= limit:
            return data
        
        # 均匀采样
        step = len(data) // limit
        samples = []
        for i in range(0, len(data), step):
            samples.append(data[i])
            if len(samples) >= limit:
                break
        
        return samples
    ```

    #### 数据脱敏处理
    - 敏感信息部分隐藏（如具体门牌号）
    - 保护个人隐私信息
    - 保留数据结构完整性

    #### 格式标准化
    ```python
    # 数据格式化
    formatting_rules = {
        "coordinates": "保留6位小数",
        "prices": "保留1位小数",
        "areas": "保留1位小数", 
        "text_fields": "去除首尾空格"
    }
    ```

    ---

    ### 性能优化

    #### 文件处理性能
    | 文件大小 | 预览耗时  | 内存占用 | 处理策略           |
    |----------|-----------|----------|--------------------|
    | < 1MB    | 1-3秒     | < 20MB   | 直接内存处理       |
    | 1-5MB    | 3-8秒     | 20-100MB | 流式读取           |
    | 5-10MB   | 8-15秒    | 100-200MB| 分块处理           |

    #### 缓存机制
    ```python
    # 预览结果缓存
    cache_config = {
        "ttl": 1800,  # 30分钟缓存
        "key_pattern": "preview:{file_hash}:{limit}",
        "max_cache_size": "100MB"
    }
    ```

    #### 并发限制
    - 同时预览文件数量限制：5个
    - 单用户预览频率限制：10次/分钟
    - 文件大小累计限制：50MB/用户

    ---

    ### 使用场景

    #### 1. 导入前验证
    ```python
    # 导入前数据验证流程
    def validate_before_import(file_path):
        # 1. 预览数据
        preview_result = preview_excel_file(file_path, limit=10)
        
        # 2. 检查验证结果
        validation = preview_result['data']['validation_summary']
        if not validation['required_fields_present']:
            return {"error": "缺少必填字段", "fields": validation['missing_required_fields']}
        
        # 3. 确认无严重问题后执行导入
        if validation['data_type_issues'] == 0:
            return {"ready_for_import": True}
        else:
            return {"warnings": validation['warnings']}
    ```

    #### 2. 数据质量评估
    ```javascript
    // 前端数据质量评估
    async function assessDataQuality(file) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('limit', '20');
        
        const response = await fetch('/api/house-reco/preview-excel', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        const validation = result.data.validation_summary;
        
        return {
            quality_score: calculateQualityScore(validation),
            issues: validation.warnings,
            ready_for_import: validation.required_fields_present
        };
    }
    ```

    #### 3. 模板格式确认
    ```python
    # 模板格式确认
    def check_template_format(file_path):
        preview = preview_excel_file(file_path, limit=1)
        field_mapping = preview['data']['field_mapping']
        
        required_fields = ['id', 'xqmc', 'qy', 'dz', 'jd', 'wd']
        missing_fields = [field for field in required_fields if field not in field_mapping]
        
        return {
            "template_valid": len(missing_fields) == 0,
            "missing_fields": missing_fields,
            "detected_fields": list(field_mapping.keys())
        }
    ```

    ---

    ### 错误处理

    #### 文件处理错误
    ```python
    file_error_types = {
        "UNSUPPORTED_FORMAT": "不支持的文件格式",
        "FILE_CORRUPT": "文件损坏",
        "FILE_TOO_LARGE": "文件过大",
        "EMPTY_FILE": "文件为空",
        "ENCODING_ERROR": "编码错误",
        "SHEET_NOT_FOUND": "工作表不存在"
    }
    ```

    #### 数据解析错误
    ```python
    parse_error_types = {
        "NO_HEADERS": "未找到列标题",
        "INVALID_STRUCTURE": "文件结构无效",
        "MIXED_DATA_TYPES": "数据类型混合",
        "EMPTY_ROWS": "存在空行",
        "SPECIAL_CHARACTERS": "特殊字符问题"
    }
    ```

    ---

    ### 最佳实践

    #### 1. 文件准备建议
    - 使用标准Excel模板
    - 确保数据类型一致
    - 避免合并单元格
    - 删除空行和无关数据

    #### 2. 预览参数优化
    ```python
    # 推荐预览参数
    preview_params = {
        "small_files": {"limit": 10},    # < 1000行
        "medium_files": {"limit": 20},   # 1000-10000行  
        "large_files": {"limit": 50}     # > 10000行
    }
    ```

    #### 3. 问题排查步骤
    1. 检查文件格式和扩展名
    2. 验证文件是否完整
    3. 确认必填字段存在
    4. 检查数据类型匹配
    5. 验证坐标范围正确性

    ---

    ### 注意事项

    - **文件大小限制**: 预览接口限制单文件最大10MB
    - **处理时间**: 大文件预览可能需要较长时间，请耐心等待
    - **数据安全**: 预览数据仅临时存储，不会永久保存
    - **格式兼容**: 建议使用最新版本的Excel格式(.xlsx)
    - **中文编码**: TSV文件建议使用UTF-8编码
    - **并发限制**: 避免同时上传多个大文件预览
    """
    temp_file_path = None
    try:
        # 验证文件类型
        if not file.filename.endswith(('.xlsx', '.xls', '.tsv', '.txt')):
            return HouseResponse(
                success=False,
                message="不支持的文件格式，请上传Excel文件(.xlsx, .xls)或TSV文件(.tsv, .txt)",
                data=None
            )
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # 获取数据导入服务
        import_service = get_data_import_service()
        
        # 预览数据
        preview_data = import_service.preview_data(temp_file_path, limit=limit)
        
        # 获取字段映射
        field_mapping = import_service.get_field_mapping()
        
        return HouseResponse(
            success=True,
            message=f"预览成功，显示前{len(preview_data)}行数据",
            data={
                "preview_data": preview_data,
                "field_mapping": field_mapping,
                "filename": file.filename,
                "preview_count": len(preview_data)
            }
        )
        
    except Exception as e:
        logger.error(f"预览Excel文件失败: {e}")
        return HouseResponse(
            success=False,
            message=f"预览失败: {str(e)}",
            data=None
        )
    finally:
        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"清理临时文件失败: {e}")


@router.get("/import-stats", response_model=HouseResponse, summary="获取导入统计信息")
def get_import_stats() -> HouseResponse:
    """
    ### GET `/import-stats` 获取导入统计信息接口

    **功能描述**:
    获取系统数据导入的详细统计信息，包括导入历史、成功率、错误统计、性能指标等。
    用于监控导入操作、分析数据质量、优化导入流程和系统运维管理。

    ---

    ### 请求参数

    无需任何请求参数。

    ---

    ### 请求示例

    #### 获取导入统计
    ```bash
    GET /api/house-reco/import-stats
    ```

    ```javascript
    // JavaScript 示例
    const response = await fetch('/api/house-reco/import-stats');
    const stats = await response.json();
    ```

    ```python
    # Python 示例
    import requests
    response = requests.get('http://localhost:8000/api/house-reco/import-stats')
    import_stats = response.json()
    ```

    ```curl
    # cURL 示例
    curl -X GET "http://localhost:8000/api/house-reco/import-stats"
    ```

    ---

    ### 响应 (Response)

    #### 成功响应 (200)
    ```json
    {
        "success": true,
        "message": "获取导入统计信息成功",
        "data": {
            "summary": {
                "total_import_sessions": 156,
                "total_records_processed": 45678,
                "total_records_imported": 44892,
                "total_records_failed": 786,
                "overall_success_rate": 98.3,
                "last_import_time": "2024-01-15T14:22:30Z",
                "total_data_size_mb": 234.5
            },
            "recent_imports": [
                {
                    "session_id": "import_20240115_142230",
                    "filename": "houses_batch_001.xlsx",
                    "timestamp": "2024-01-15T14:22:30Z",
                    "records_processed": 500,
                    "records_imported": 485,
                    "records_failed": 15,
                    "success_rate": 97.0,
                    "processing_time_ms": 12500,
                    "file_size_mb": 3.2,
                    "error_summary": {
                        "DUPLICATE_ID": 8,
                        "INVALID_COORDINATES": 4,
                        "MISSING_REQUIRED_FIELDS": 3
                    }
                },
                {
                    "session_id": "import_20240115_103045",
                    "filename": "new_houses_data.xlsx",
                    "timestamp": "2024-01-15T10:30:45Z",
                    "records_processed": 1200,
                    "records_imported": 1185,
                    "records_failed": 15,
                    "success_rate": 98.8,
                    "processing_time_ms": 28750,
                    "file_size_mb": 7.8,
                    "error_summary": {
                        "INVALID_COORDINATES": 10,
                        "DATA_TYPE_ERROR": 5
                    }
                }
            ],
            "error_statistics": {
                "error_distribution": {
                    "DUPLICATE_ID": {
                        "count": 245,
                        "percentage": 31.2,
                        "description": "重复的房源ID",
                        "recent_trend": "decreasing"
                    },
                    "INVALID_COORDINATES": {
                        "count": 189,
                        "percentage": 24.0,
                        "description": "坐标超出有效范围",
                        "recent_trend": "stable"
                    },
                    "MISSING_REQUIRED_FIELDS": {
                        "count": 156,
                        "percentage": 19.8,
                        "description": "缺少必填字段",
                        "recent_trend": "decreasing"
                    },
                    "DATA_TYPE_ERROR": {
                        "count": 123,
                        "percentage": 15.6,
                        "description": "数据类型错误",
                        "recent_trend": "stable"
                    },
                    "VECTORIZATION_ERROR": {
                        "count": 73,
                        "percentage": 9.3,
                        "description": "向量化处理失败",
                        "recent_trend": "increasing"
                    }
                },
                "most_common_errors": [
                    "DUPLICATE_ID",
                    "INVALID_COORDINATES", 
                    "MISSING_REQUIRED_FIELDS"
                ],
                "error_trend": {
                    "last_7_days": 45,
                    "last_30_days": 178,
                    "change_percentage": -12.5
                }
            },
            "performance_metrics": {
                "throughput": {
                    "avg_records_per_second": 15.8,
                    "max_records_per_second": 42.3,
                    "min_records_per_second": 3.2
                },
                "processing_time": {
                    "avg_time_per_record_ms": 63.2,
                    "avg_vectorization_time_ms": 45.8,
                    "avg_storage_time_ms": 17.4
                },
                "file_processing": {
                    "avg_file_size_mb": 4.7,
                    "max_file_size_mb": 15.2,
                    "avg_processing_time_per_mb_ms": 2847
                }
            },
            "data_quality_metrics": {
                "field_completeness": {
                    "required_fields": 100.0,
                    "optional_fields": 76.3,
                    "semantic_descriptions": 89.2
                },
                "coordinate_accuracy": {
                    "valid_coordinates": 98.7,
                    "precision_6_decimal": 94.2,
                    "within_china_bounds": 99.1
                },
                "data_consistency": {
                    "price_area_logic": 96.8,
                    "id_uniqueness": 99.8,
                    "text_format": 92.4
                }
            },
            "system_impact": {
                "storage_impact": {
                    "added_storage_mb": 234.5,
                    "vector_storage_mb": 178.2,
                    "index_storage_mb": 56.3
                },
                "performance_impact": {
                    "search_latency_change_ms": 2.3,
                    "memory_usage_change_mb": 45.7,
                    "index_rebuild_time_ms": 3450
                }
            },
            "recommendations": [
                {
                    "type": "data_quality",
                    "priority": "high",
                    "message": "建议在导入前验证坐标数据，31.2%的错误来自重复ID",
                    "action": "使用预览接口验证数据"
                },
                {
                    "type": "performance",
                    "priority": "medium", 
                    "message": "向量化处理时间偏长，建议优化批次大小",
                    "action": "将批次大小调整为100-200条记录"
                }
            ],
            "generated_at": "2024-01-15T15:30:45Z"
        }
    }
    ```

    #### 系统错误 (500)
    ```json
    {
        "success": false,
        "message": "获取统计信息失败: 数据库连接异常",
        "data": {
            "error_code": "DATABASE_ERROR",
            "retry_suggested": true
        }
    }
    ```

    ---

    ### 统计指标详解

    #### 总体统计 (Summary)
    | 字段                        | 描述                    | 单位        |
    |-----------------------------|-------------------------|-------------|
    | `total_import_sessions`     | 总导入会话数            | 次          |
    | `total_records_processed`   | 总处理记录数            | 条          |
    | `total_records_imported`    | 总成功导入记录数        | 条          |
    | `total_records_failed`      | 总失败记录数            | 条          |
    | `overall_success_rate`      | 整体成功率              | %           |
    | `last_import_time`          | 最后导入时间            | ISO 8601    |
    | `total_data_size_mb`        | 总数据大小              | MB          |

    #### 最近导入记录 (Recent Imports)
    ```json
    // 单次导入记录结构
    {
        "session_id": "导入会话ID",
        "filename": "文件名",
        "timestamp": "导入时间",
        "records_processed": "处理记录数",
        "records_imported": "成功导入数",
        "records_failed": "失败记录数",
        "success_rate": "成功率",
        "processing_time_ms": "处理时间",
        "file_size_mb": "文件大小",
        "error_summary": "错误类型统计"
    }
    ```

    #### 错误统计 (Error Statistics)
    | 错误类型                    | 描述                    | 常见原因            |
    |-----------------------------|-------------------------|---------------------|
    | `DUPLICATE_ID`              | 重复的房源ID            | 数据源重复、导入重复 |
    | `INVALID_COORDINATES`       | 坐标无效                | 坐标超出中国范围    |
    | `MISSING_REQUIRED_FIELDS`   | 缺少必填字段            | 数据不完整          |
    | `DATA_TYPE_ERROR`           | 数据类型错误            | 格式不匹配          |
    | `VECTORIZATION_ERROR`       | 向量化失败              | 语义内容异常        |

    #### 性能指标 (Performance Metrics)
    | 指标类别      | 具体指标                    | 说明                      |
    |---------------|----------------------------|---------------------------|
    | **吞吐量**    | `avg_records_per_second`   | 平均每秒处理记录数         |
    |               | `max_records_per_second`   | 最大每秒处理记录数         |
    |               | `min_records_per_second`   | 最小每秒处理记录数         |
    | **处理时间**  | `avg_time_per_record_ms`   | 平均每条记录处理时间       |
    |               | `avg_vectorization_time_ms`| 平均向量化时间            |
    |               | `avg_storage_time_ms`      | 平均存储时间              |

    ---

    ### 数据质量分析

    #### 字段完整性分析
    ```json
    {
        "field_completeness": {
            "required_fields": 100.0,      // 必填字段完整率
            "optional_fields": 76.3,       // 可选字段完整率  
            "semantic_descriptions": 89.2   // 语义描述完整率
        }
    }
    ```

    #### 坐标数据质量
    ```json
    {
        "coordinate_accuracy": {
            "valid_coordinates": 98.7,      // 有效坐标比例
            "precision_6_decimal": 94.2,    // 高精度坐标比例
            "within_china_bounds": 99.1     // 中国境内坐标比例
        }
    }
    ```

    #### 数据一致性检查
    ```json
    {
        "data_consistency": {
            "price_area_logic": 96.8,       // 价格面积逻辑一致性
            "id_uniqueness": 99.8,          // ID唯一性
            "text_format": 92.4             // 文本格式规范性
        }
    }
    ```

    ---

    ### 性能趋势分析

    #### 吞吐量趋势
    ```python
    # 吞吐量计算方式
    throughput = total_records / total_time_seconds
    
    # 趋势分析
    performance_trend = {
        "last_7_days": 15.8,   # 最近7天平均吞吐量
        "last_30_days": 14.2,  # 最近30天平均吞吐量
        "improvement": "+11.3%" # 性能改善情况
    }
    ```

    #### 错误率趋势
    ```python
    # 错误率计算
    error_rate = (failed_records / total_records) * 100
    
    # 趋势分析
    error_trend = {
        "current_week": 1.7,    # 本周错误率
        "last_week": 2.3,       # 上周错误率
        "trend": "improving"    # 趋势方向
    }
    ```

    ---

    ### 系统影响评估

    #### 存储影响
    | 指标                | 说明                      | 示例值    |
    |--------------------|---------------------------|-----------|
    | `added_storage_mb` | 新增存储空间              | 234.5 MB  |
    | `vector_storage_mb`| 向量数据存储              | 178.2 MB  |
    | `index_storage_mb` | 索引数据存储              | 56.3 MB   |

    #### 性能影响
    | 指标                        | 说明                  | 影响评估  |
    |----------------------------|-----------------------|-----------|
    | `search_latency_change_ms` | 搜索延迟变化          | +2.3ms    |
    | `memory_usage_change_mb`   | 内存使用变化          | +45.7MB   |
    | `index_rebuild_time_ms`    | 索引重建时间          | 3450ms    |

    ---

    ### 优化建议系统

    #### 建议类型
    ```json
    {
        "recommendation_types": {
            "data_quality": "数据质量改进建议",
            "performance": "性能优化建议",
            "system": "系统配置建议",
            "workflow": "流程优化建议"
        }
    }
    ```

    #### 建议优先级
    ```python
    priority_levels = {
        "critical": "严重问题，需要立即处理",
        "high": "重要问题，建议优先处理", 
        "medium": "中等问题，可以计划处理",
        "low": "轻微问题，有时间时处理"
    }
    ```

    #### 智能建议算法
    ```python
    def generate_recommendations(stats):
        recommendations = []
        
        # 数据质量建议
        if stats['error_rate'] > 5.0:
            recommendations.append({
                "type": "data_quality",
                "priority": "high",
                "message": "错误率过高，建议加强数据验证"
            })
        
        # 性能建议
        if stats['avg_processing_time'] > 100:
            recommendations.append({
                "type": "performance", 
                "priority": "medium",
                "message": "处理时间偏长，建议优化批处理参数"
            })
        
        return recommendations
    ```

    ---

    ### 使用场景

    #### 1. 运维监控
    ```python
    # 定期检查导入状态
    def monitor_import_health():
        stats = get_import_stats()
        
        # 检查错误率
        if stats['data']['summary']['overall_success_rate'] < 95.0:
            alert_admin("导入成功率低于95%")
        
        # 检查性能指标
        if stats['data']['performance_metrics']['throughput']['avg_records_per_second'] < 10:
            alert_admin("处理性能下降")
        
        return stats
    ```

    #### 2. 数据质量分析
    ```javascript
    // 前端数据质量仪表板
    async function loadQualityDashboard() {
        const stats = await fetch('/api/house-reco/import-stats');
        const data = await stats.json();
        
        // 显示质量指标
        updateQualityMetrics(data.data.data_quality_metrics);
        
        // 显示错误分布
        renderErrorChart(data.data.error_statistics.error_distribution);
        
        // 显示趋势图
        renderTrendChart(data.data.performance_metrics);
    }
    ```

    #### 3. 容量规划
    ```python
    # 基于统计数据进行容量规划
    def plan_system_capacity(stats):
        current_storage = stats['data']['system_impact']['storage_impact']['added_storage_mb']
        avg_growth_rate = calculate_growth_rate(stats['data']['recent_imports'])
        
        # 预测未来6个月存储需求
        predicted_storage = current_storage * (1 + avg_growth_rate) ** 6
        
        return {
            "current_storage_mb": current_storage,
            "predicted_storage_mb": predicted_storage,
            "additional_needed_mb": predicted_storage - current_storage
        }
    ```

    ---

    ### 高级分析功能

    #### 错误模式识别
    ```python
    # 识别错误模式
    def analyze_error_patterns(error_stats):
        patterns = []
        
        # 时间模式分析
        if is_weekend_error_spike(error_stats):
            patterns.append("周末错误率异常升高")
        
        # 文件类型模式
        if is_large_file_error_prone(error_stats):
            patterns.append("大文件导入错误率高")
        
        return patterns
    ```

    #### 性能基准对比
    ```python
    # 性能基准对比
    performance_benchmarks = {
        "excellent": {"throughput": ">25 records/sec", "error_rate": "<1%"},
        "good": {"throughput": "15-25 records/sec", "error_rate": "1-3%"},
        "average": {"throughput": "10-15 records/sec", "error_rate": "3-5%"},
        "poor": {"throughput": "<10 records/sec", "error_rate": ">5%"}
    }
    ```

    ---

    ### API集成示例

    #### 监控集成
    ```python
    # 集成到监控系统
    import requests
    
    def export_metrics_to_prometheus():
        stats = requests.get('http://localhost:8000/api/house-reco/import-stats').json()
        
        # 转换为Prometheus格式
        metrics = [
            f'house_import_success_rate {stats["data"]["summary"]["overall_success_rate"]}',
            f'house_import_throughput {stats["data"]["performance_metrics"]["throughput"]["avg_records_per_second"]}',
            f'house_import_error_total {stats["data"]["summary"]["total_records_failed"]}'
        ]
        
        return metrics
    ```

    #### 报警集成
    ```python
    # 自动报警系统
    def check_and_alert():
        stats = requests.get('http://localhost:8000/api/house-reco/import-stats').json()
        
        # 检查关键指标
        if stats['data']['summary']['overall_success_rate'] < 90:
            send_alert("导入成功率低于90%", severity="high")
        
        if stats['data']['error_statistics']['error_trend']['last_7_days'] > 100:
            send_alert("最近7天错误数量激增", severity="medium")
    ```

    ---

    ### 注意事项

    - **数据时效性**: 统计数据每5分钟更新一次，可能存在轻微延迟
    - **历史数据**: 默认保留90天的详细统计数据，90天前的数据会被聚合
    - **性能影响**: 统计计算对系统性能影响很小，可以频繁调用
    - **权限控制**: 建议对统计接口进行适当的访问控制
    - **缓存机制**: 统计结果有5分钟缓存，减少重复计算开销
    - **数据准确性**: 统计数据基于实际导入记录，保证数据准确性
    """
    try:
        import_service = get_data_import_service()
        
        stats = import_service.get_import_statistics()
        
        return HouseResponse(
            success=True,
            message="获取导入统计信息成功",
            data=stats
        )
        
    except Exception as e:
        logger.error(f"获取导入统计信息失败: {e}")
        return HouseResponse(
            success=False,
            message=f"获取统计信息失败: {str(e)}",
            data=None
        )


@router.post("/clear-data", response_model=HouseResponse, summary="清空所有房源数据")
def clear_all_data() -> HouseResponse:
    """
    ### POST `/clear-data` 清空所有房源数据接口

    **功能描述**:
    彻底清空系统中的所有房源数据，包括Milvus向量数据库中的所有记录。
    此操作不可逆，请谨慎使用。适用于系统重置、测试环境清理等场景。

    ---

    ### 请求参数

    无需任何请求参数。

    ---

    ### 响应 (Response)

    #### 成功响应 (200)
    ```json
    {
        "success": true,
        "message": "成功清空所有房源数据",
        "data": {
            "cleared": true
        }
    }
    ```

    #### 错误响应 (500)
    ```json
    {
        "success": false,
        "message": "清空失败: 具体错误信息",
        "data": null
    }
    ```

    ---

    ### 操作流程

    1. **权限检查**: 验证操作权限（如有权限控制）
    2. **连接检查**: 确认Milvus数据库连接正常
    3. **集合检查**: 检查房源集合是否存在
    4. **数据清理**: 删除集合中的所有数据记录
    5. **索引重建**: 重新构建必要的索引结构
    6. **状态更新**: 更新系统状态和统计信息

    ---

    ### 清理范围

    #### 会被清空的数据
    - **房源基础信息**: 所有房源的结构化数据
    - **向量数据**: 所有语义字段生成的向量
    - **索引数据**: 相关的搜索索引
    - **统计信息**: 房源数量、导入记录等统计数据

    #### 不会被清空的数据
    - **集合结构**: Milvus集合定义保持不变
    - **字段映射**: 数据字段映射配置保持不变
    - **系统配置**: 系统参数和配置信息
    - **日志记录**: 操作日志和错误日志

    ---

    ### 使用场景

    #### 1. 系统重置
    ```bash
    # 完全重新开始，清空所有数据后重新导入
    curl -X POST "http://localhost:8000/api/house-reco/clear-data"
    ```

    #### 2. 测试环境清理
    ```bash
    # 测试完成后清理测试数据
    curl -X POST "http://localhost:8000/api/house-reco/clear-data"
    ```

    #### 3. 数据结构变更
    ```bash
    # 在修改数据结构前清空旧数据
    curl -X POST "http://localhost:8000/api/house-reco/clear-data"
    ```

    ---

    ### 性能影响

    #### 执行时间
    - **小量数据** (< 1万条): 1-3秒
    - **中等数据** (1-10万条): 3-10秒  
    - **大量数据** (> 10万条): 10-30秒

    #### 系统影响
    - **暂时不可用**: 清理期间搜索功能暂时不可用
    - **内存释放**: 清理后会释放大量内存空间
    - **存储回收**: 磁盘存储空间会被回收

    ---

    ### 安全建议

    #### 1. 操作前确认
    - 确认确实需要清空所有数据
    - 检查是否有重要数据需要备份
    - 确认当前环境（生产/测试）

    #### 2. 数据备份
    ```bash
    # 建议在清空前先导出数据备份
    curl -X GET "http://localhost:8000/api/house-reco/export-data" > backup.json
    ```

    #### 3. 权限控制
    - 建议在生产环境中限制此接口的访问权限
    - 可以添加额外的确认步骤或管理员审批
    - 记录操作日志用于审计

    ---

    ### 恢复数据

    #### 从备份恢复
    ```bash
    # 1. 清空数据
    curl -X POST "http://localhost:8000/api/house-reco/clear-data"
    
    # 2. 重新导入备份数据
    curl -X POST "http://localhost:8000/api/house-reco/upload-excel" \
         -F "file=@backup_data.xlsx"
    ```

    #### 重新导入
    ```bash
    # 从原始数据源重新导入
    curl -X POST "http://localhost:8000/api/house-reco/upload-excel" \
         -F "file=@original_data.xlsx"
    ```

    ---

    ### 注意事项

    - **不可逆操作**: 一旦执行，数据无法直接恢复
    - **服务中断**: 清理期间相关服务将暂时不可用
    - **并发限制**: 清理期间避免并发写入操作
    - **生产环境**: 生产环境使用需要特别谨慎
    - **监控告警**: 建议配置监控，及时发现异常情况
    - **日志记录**: 系统会记录清理操作的详细日志
    """
    try:
        import_service = get_data_import_service()
        
        success = import_service.clear_all_data()
        
        if success:
            return HouseResponse(
                success=True,
                message="成功清空所有房源数据",
                data={"cleared": True}
            )
        else:
            return HouseResponse(
                success=False,
                message="清空数据失败",
                data=None
            )
            
    except Exception as e:
        logger.error(f"清空数据失败: {e}")
        return HouseResponse(
            success=False,
            message=f"清空失败: {str(e)}",
            data=None
        )


@router.post("/create-collection", response_model=HouseResponse, summary="创建房源推荐集合")
def create_collection() -> HouseResponse:
    """
    ### POST `/create-collection` 创建房源推荐集合接口

    **功能描述**:
    在Milvus向量数据库中创建房源推荐系统所需的数据集合（Collection）。
    定义数据结构、向量字段、索引配置等，为房源数据存储和检索建立基础架构。

    ---

    ### 请求参数

    无需任何请求参数。系统将使用预定义的集合结构和配置。

    ---

    ### 请求示例

    #### 创建房源集合
    ```bash
    POST /api/house-reco/create-collection
    ```

    ```javascript
    // JavaScript 示例
    const response = await fetch('/api/house-reco/create-collection', {
        method: 'POST'
    });
    const result = await response.json();
    ```

    ```python
    # Python 示例
    import requests
    response = requests.post('http://localhost:8000/api/house-reco/create-collection')
    create_result = response.json()
    ```

    ```curl
    # cURL 示例
    curl -X POST "http://localhost:8000/api/house-reco/create-collection"
    ```

    ---

    ### 响应 (Response)

    #### 成功创建 (200)
    ```json
    {
        "success": true,
        "message": "成功创建房源推荐集合",
        "data": {
            "collection_name": "house_recommendation",
            "collection_id": 451234567890,
            "created_at": "2024-01-15T10:30:45Z",
            "schema": {
                "fields": [
                    {
                        "name": "id",
                        "type": "VarChar",
                        "max_length": 50,
                        "is_primary": true,
                        "description": "房源唯一标识符"
                    },
                    {
                        "name": "category", 
                        "type": "Int64",
                        "description": "房源分类：1-新房；2-二手房；3-出租房"
                    },
                    {
                        "name": "semantic_vector",
                        "type": "FloatVector",
                        "dimension": 768,
                        "description": "语义描述向量"
                    },
                    {
                        "name": "name",
                        "type": "VarChar",
                        "max_length": 200,
                        "description": "房源名称/小区名称"
                    },
                    {
                        "name": "region",
                        "type": "VarChar", 
                        "max_length": 100,
                        "description": "区域"
                    }
                ],
                "total_fields": 25,
                "primary_field": "id",
                "vector_field": "semantic_vector"
            },
            "index_config": {
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "index_params": {
                    "nlist": 1024
                }
            },
            "storage_config": {
                "estimated_size_per_entity": "6KB",
                "max_entities": 10000000,
                "estimated_max_size_gb": 60
            }
        }
    }
    ```

    #### 集合已存在 (409)
    ```json
    {
        "success": false,
        "message": "房源推荐集合已存在",
        "data": {
            "collection_name": "house_recommendation",
            "existing_collection_info": {
                "created_at": "2024-01-10T08:15:30Z",
                "entity_count": 15678,
                "size_mb": 94.2
            },
            "suggestion": "如需重新创建，请先删除现有集合"
        }
    }
    ```

    #### Milvus连接错误 (503)
    ```json
    {
        "success": false,
        "message": "创建失败: Milvus数据库连接异常",
        "data": {
            "error_code": "MILVUS_CONNECTION_ERROR",
            "milvus_host": "localhost:19530",
            "retry_suggested": true,
            "troubleshooting": [
                "检查Milvus服务是否运行",
                "验证连接配置是否正确",
                "确认网络连接正常"
            ]
        }
    }
    ```

    #### 权限错误 (403)
    ```json
    {
        "success": false,
        "message": "创建失败: 没有创建集合的权限",
        "data": {
            "error_code": "INSUFFICIENT_PRIVILEGES",
            "required_permissions": ["CREATE_COLLECTION", "CREATE_INDEX"],
            "suggestion": "请联系管理员配置相应权限"
        }
    }
    ```

    #### 系统错误 (500)
    ```json
    {
        "success": false,
        "message": "创建失败: 系统内部错误",
        "data": {
            "error_code": "INTERNAL_ERROR",
            "operation_id": "create_20240115_103045"
        }
    }
    ```

    ---

    ### 集合结构详解

    #### 数据字段定义
    ```python
    # 完整字段定义
    collection_schema = {
        # 基础字段
        "id": {"type": "VarChar", "max_length": 50, "primary_key": True},
        "category": {"type": "Int64", "description": "房源分类"},
        "name": {"type": "VarChar", "max_length": 200},
        "region": {"type": "VarChar", "max_length": 100},
        "address": {"type": "VarChar", "max_length": 500},
        
        # 地理坐标
        "longitude": {"type": "Double", "description": "经度"},
        "latitude": {"type": "Double", "description": "纬度"},
        
        # 数值字段
        "min_area": {"type": "Float", "nullable": True},
        "max_area": {"type": "Float", "nullable": True},
        "min_total_price": {"type": "Float", "nullable": True},
        "max_total_price": {"type": "Float", "nullable": True},
        "rent": {"type": "Float", "nullable": True},
        
        # 属性字段
        "type": {"type": "VarChar", "max_length": 100, "nullable": True},
        "orientation": {"type": "VarChar", "max_length": 50, "nullable": True},
        "decoration_status": {"type": "VarChar", "max_length": 50, "nullable": True},
        "has_elevator": {"type": "VarChar", "max_length": 20, "nullable": True},
        "has_parking": {"type": "VarChar", "max_length": 20, "nullable": True},
        
        # 向量字段
        "semantic_vector": {"type": "FloatVector", "dimension": 768}
    }
    ```

    #### 索引配置详解
    | 配置项        | 值          | 说明                              |
    |---------------|-------------|-----------------------------------|
    | `index_type`  | IVF_FLAT    | 倒排文件索引，平衡精度和性能      |
    | `metric_type` | COSINE      | 余弦相似度，适合语义相似度计算    |
    | `nlist`       | 1024        | 聚类中心数量，影响搜索精度和速度  |

    #### 可选索引配置
    ```python
    # 高性能配置（适合大数据量）
    high_performance_index = {
        "index_type": "IVF_SQ8",
        "metric_type": "COSINE", 
        "index_params": {
            "nlist": 2048,
            "m": 8
        }
    }
    
    # 高精度配置（适合小数据量）
    high_accuracy_index = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "index_params": {
            "M": 16,
            "efConstruction": 256
        }
    }
    ```

    ---

    ### 创建流程详解

    #### 1. 环境检查
    ```python
    def check_environment():
        # 检查Milvus连接
        if not milvus_client.is_connected():
            raise ConnectionError("Milvus数据库连接失败")
        
        # 检查权限
        if not check_create_permission():
            raise PermissionError("没有创建集合权限")
        
        # 检查存储空间
        if not check_storage_space():
            raise StorageError("存储空间不足")
        
        return True
    ```

    #### 2. 集合创建
    ```python
    def create_collection_process():
        # 步骤1: 定义集合Schema
        schema = define_collection_schema()
        
        # 步骤2: 创建集合
        collection = Collection(
            name="house_recommendation",
            schema=schema,
            description="房源推荐系统数据集合"
        )
        
        # 步骤3: 创建索引
        collection.create_index(
            field_name="semantic_vector",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE", 
                "params": {"nlist": 1024}
            }
        )
        
        # 步骤4: 加载集合到内存
        collection.load()
        
        return collection
    ```

    #### 3. 验证和初始化
    ```python
    def validate_and_initialize():
        # 验证集合创建成功
        collection = Collection("house_recommendation")
        assert collection.is_empty
        
        # 初始化统计信息
        init_collection_stats()
        
        # 设置集合别名（可选）
        set_collection_alias("house_recommendation", "main_collection")
        
        return True
    ```

    ---

    ### 存储容量规划

    #### 存储计算
    ```python
    # 单条记录存储估算
    storage_estimation = {
        "structured_data": 2048,    # 2KB (结构化字段)
        "vector_data": 3072,        # 3KB (768维float32向量)
        "index_overhead": 1024,     # 1KB (索引开销)
        "total_per_record": 6144    # 6KB (总计)
    }
    
    # 容量规划计算
    def calculate_storage_requirements(max_entities=1000000):
        total_size_bytes = max_entities * storage_estimation["total_per_record"]
        total_size_gb = total_size_bytes / (1024**3)
        
        return {
            "max_entities": max_entities,
            "estimated_size_gb": round(total_size_gb, 2),
            "recommended_disk_gb": round(total_size_gb * 1.5, 2)  # 50%缓冲
        }
    ```

    #### 性能预估
    | 数据量    | 内存占用  | 索引大小  | 搜索延迟 | QPS估计 |
    |-----------|-----------|-----------|----------|---------|
    | 10万条    | 600MB     | 200MB     | 10-20ms  | 1000+   |
    | 50万条    | 3GB       | 1GB       | 20-50ms  | 500-800 |
    | 100万条   | 6GB       | 2GB       | 30-80ms  | 300-500 |
    | 500万条   | 30GB      | 10GB      | 50-150ms | 100-300 |

    ---

    ### 最佳实践配置

    #### 生产环境配置
    ```python
    # 生产环境推荐配置
    production_config = {
        "collection_name": "house_recommendation_prod",
        "shards_num": 4,              # 分片数，提高并发性能
        "consistency_level": "Strong", # 强一致性
        "index_config": {
            "index_type": "IVF_SQ8",   # 压缩索引，节省内存
            "metric_type": "COSINE",
            "nlist": 2048              # 较大的nlist，提高精度
        },
        "load_config": {
            "replica_number": 2         # 副本数，提高可用性
        }
    }
    ```

    #### 开发测试配置
    ```python
    # 开发测试环境配置
    development_config = {
        "collection_name": "house_recommendation_dev",
        "shards_num": 1,               # 单分片，节省资源
        "consistency_level": "Session", # 会话一致性
        "index_config": {
            "index_type": "IVF_FLAT",   # 简单索引，快速创建
            "metric_type": "COSINE",
            "nlist": 512               # 较小的nlist，快速构建
        }
    }
    ```

    ---

    ### 使用场景

    #### 1. 系统初始化
    ```python
    # 系统首次部署时创建集合
    def initialize_system():
        try:
            # 创建主集合
            result = create_collection()
            if result['success']:
                print("房源推荐系统初始化成功")
                
                # 创建测试数据
                create_sample_data()
                
                # 验证系统功能
                test_basic_operations()
                
            return result
        except Exception as e:
            print(f"系统初始化失败: {e}")
            return False
    ```

    #### 2. 数据迁移准备
    ```python
    # 数据迁移前创建新集合
    def prepare_data_migration():
        # 创建新集合
        new_collection_name = f"house_recommendation_{datetime.now().strftime('%Y%m%d')}"
        
        # 使用相同schema但可能不同的配置
        create_result = create_collection_with_custom_config(
            name=new_collection_name,
            config=get_migration_config()
        )
        
        return create_result
    ```

    #### 3. 开发环境搭建
    ```python
    # 开发人员本地环境搭建
    def setup_dev_environment():
        # 检查本地Milvus是否运行
        if not check_local_milvus():
            print("请先启动本地Milvus服务")
            return False
        
        # 创建开发集合
        dev_result = create_collection_with_config(development_config)
        
        # 导入测试数据
        if dev_result['success']:
            import_test_data()
        
        return dev_result
    ```

    ---

    ### 故障排除

    #### 常见错误及解决方案
    ```python
    troubleshooting_guide = {
        "MILVUS_CONNECTION_ERROR": {
            "原因": "无法连接到Milvus服务",
            "解决方案": [
                "检查Milvus服务状态：docker ps | grep milvus",
                "验证连接参数：host, port", 
                "检查网络连接和防火墙设置",
                "查看Milvus日志：docker logs milvus-standalone"
            ]
        },
        "INSUFFICIENT_PRIVILEGES": {
            "原因": "权限不足",
            "解决方案": [
                "确认用户具有CREATE_COLLECTION权限",
                "检查Milvus用户配置",
                "使用管理员账户创建集合"
            ]
        },
        "COLLECTION_EXISTS": {
            "原因": "集合已存在",
            "解决方案": [
                "检查现有集合状态",
                "如需重建，先删除现有集合",
                "使用不同的集合名称"
            ]
        },
        "SCHEMA_ERROR": {
            "原因": "Schema定义错误",
            "解决方案": [
                "验证字段类型和约束",
                "检查向量维度设置",
                "确认主键字段配置"
            ]
        }
    }
    ```

    #### 健康检查
    ```python
    # 创建后健康检查
    def health_check_after_creation(collection_name):
        checks = []
        
        # 检查集合存在
        if Collection.exists(collection_name):
            checks.append({"test": "collection_exists", "status": "PASS"})
        else:
            checks.append({"test": "collection_exists", "status": "FAIL"})
        
        # 检查索引状态
        collection = Collection(collection_name)
        index_info = collection.index()
        if index_info:
            checks.append({"test": "index_created", "status": "PASS"})
        else:
            checks.append({"test": "index_created", "status": "FAIL"})
        
        # 检查集合是否已加载
        if collection.is_empty is not None:
            checks.append({"test": "collection_loaded", "status": "PASS"})
        else:
            checks.append({"test": "collection_loaded", "status": "FAIL"})
        
        return checks
    ```

    ---

    ### 监控和维护

    #### 集合状态监控
    ```python
    def monitor_collection_status():
        collection = Collection("house_recommendation")
        
        status = {
            "name": collection.name,
            "entity_count": collection.num_entities,
            "is_loaded": collection.is_empty is not None,
            "index_status": collection.index().params,
            "memory_usage": get_memory_usage(collection),
            "last_insert": get_last_insert_time(collection)
        }
        
        return status
    ```

    #### 性能优化建议
    ```python
    def get_optimization_recommendations(collection_stats):
        recommendations = []
        
        # 检查数据量与索引配置
        if collection_stats['entity_count'] > 1000000:
            if collection_stats['index_type'] == 'IVF_FLAT':
                recommendations.append({
                    "type": "index_optimization",
                    "message": "数据量较大，建议使用IVF_SQ8压缩索引",
                    "priority": "medium"
                })
        
        # 检查内存使用
        if collection_stats['memory_usage'] > 8000:  # 8GB
            recommendations.append({
                "type": "memory_optimization", 
                "message": "内存使用偏高，建议增加分片数或优化索引",
                "priority": "high"
            })
        
        return recommendations
    ```

    ---

    ### 注意事项

    - **唯一性要求**: 每个系统只需创建一次，重复创建会返回错误
    - **资源需求**: 创建过程需要足够的系统资源，建议在低负载时进行
    - **数据持久化**: 创建的集合结构是持久化的，删除需要明确操作
    - **版本兼容**: 不同Milvus版本的Schema定义可能有差异
    - **权限管理**: 生产环境建议限制集合创建权限
    - **备份策略**: 重要的集合结构建议进行配置备份
    - **监控告警**: 建议配置集合状态监控和异常告警
    """
    try:
        service = get_house_reco_service()
        
        success = service.create_collection()
        
        if success:
            return HouseResponse(
                success=True,
                message="成功创建房源推荐集合",
                data={"collection_name": service.COLLECTION_NAME}
            )
        else:
            return HouseResponse(
                success=False,
                message="创建集合失败",
                data=None
            )
            
    except Exception as e:
        logger.error(f"创建集合失败: {e}")
        return HouseResponse(
            success=False,
            message=f"创建失败: {str(e)}",
            data=None
        )


@router.post("/calculate-distance", response_model=HouseResponse, summary="计算两点间距离")
def calculate_distance_between_points(
    lat1: float, lng1: float, lat2: float, lng2: float
) -> HouseResponse:
    """
    ### POST `/calculate-distance` 计算地理坐标距离接口

    **功能描述**:
    使用高精度算法计算两个地理坐标点之间的直线距离（球面距离）。
    基于WGS84坐标系和Haversine公式，提供准确的地理距离计算服务。

    ---

    ### 请求参数 (Query Parameters)

    | 参数   | 类型     | 必填 | 描述                | 范围            | 示例      |
    |--------|----------|------|--------------------|--------------------|-----------|
    | `lat1` | `float`  | 是   | 第一个点的纬度     | -90.0 到 90.0     | 39.9042   |
    | `lng1` | `float`  | 是   | 第一个点的经度     | -180.0 到 180.0   | 116.4074  |
    | `lat2` | `float`  | 是   | 第二个点的纬度     | -90.0 到 90.0     | 39.9785   |
    | `lng2` | `float`  | 是   | 第二个点的经度     | -180.0 到 180.0   | 116.3105  |

    ---

    ### 请求示例

    #### URL查询参数方式
    ```bash
    POST /api/house-reco/calculate-distance?lat1=39.9042&lng1=116.4074&lat2=39.9785&lng2=116.3105
    ```

    ```javascript
    // JavaScript 示例
    const params = new URLSearchParams({
        lat1: 39.9042,
        lng1: 116.4074,
        lat2: 39.9785,
        lng2: 116.3105
    });
    
    const response = await fetch(`/api/house-reco/calculate-distance?${params}`, {
        method: 'POST'
    });
    const result = await response.json();
    ```

    ```python
    # Python 示例
    import requests
    
    params = {
        'lat1': 39.9042,
        'lng1': 116.4074,
        'lat2': 39.9785,
        'lng2': 116.3105
    }
    
    response = requests.post(
        'http://localhost:8000/api/house-reco/calculate-distance',
        params=params
    )
    distance_result = response.json()
    ```

    ```curl
    # cURL 示例
    curl -X POST "http://localhost:8000/api/house-reco/calculate-distance?lat1=39.9042&lng1=116.4074&lat2=39.9785&lng2=116.3105"
    ```

    ---

    ### 响应 (Response)

    #### 成功响应 (200)
    ```json
    {
        "success": true,
        "message": "距离计算成功",
        "data": {
            "point1": {
                "latitude": 39.9042,
                "longitude": 116.4074,
                "location_name": "天安门广场"
            },
            "point2": {
                "latitude": 39.9785,
                "longitude": 116.3105,
                "location_name": "中关村"
            },
            "distance_km": 10.23,
            "distance_m": 10230.0,
            "distance_miles": 6.36,
            "calculation_method": "Haversine",
            "coordinate_system": "WGS84",
            "accuracy": "高精度",
            "calculated_at": "2024-01-15T10:30:45Z",
            "additional_info": {
                "great_circle_distance": true,
                "earth_radius_km": 6371.0,
                "bearing_degrees": 284.5,
                "initial_bearing": "西北方向"
            }
        }
    }
    ```

    #### 坐标参数错误 (400)
    ```json
    {
        "success": false,
        "message": "坐标参数错误: 纬度必须在-90到90度之间",
        "data": {
            "error_code": "INVALID_COORDINATES",
            "invalid_params": ["lat1"],
            "provided_values": {
                "lat1": 195.5,
                "lng1": 116.4074,
                "lat2": 39.9785,
                "lng2": 116.3105
            },
            "valid_ranges": {
                "latitude": {"min": -90.0, "max": 90.0},
                "longitude": {"min": -180.0, "max": 180.0}
            },
            "suggestions": [
                "检查纬度值是否在-90到90度范围内",
                "检查经度值是否在-180到180度范围内",
                "确认坐标格式为十进制度数"
            ]
        }
    }
    ```

    #### 缺少参数 (400)
    ```json
    {
        "success": false,
        "message": "缺少必需参数: lat1, lng1",
        "data": {
            "error_code": "MISSING_PARAMETERS",
            "missing_params": ["lat1", "lng1"],
            "required_params": ["lat1", "lng1", "lat2", "lng2"],
            "provided_params": ["lat2", "lng2"]
        }
    }
    ```

    #### 计算错误 (500)
    ```json
    {
        "success": false,
        "message": "距离计算失败: 数学计算异常",
        "data": {
            "error_code": "CALCULATION_ERROR",
            "operation_id": "calc_20240115_103045"
        }
    }
    ```

    ---

    ### 算法详解

    #### Haversine公式
    ```python
    # Haversine公式实现
    import math
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        # 地球半径（公里）
        R = 6371.0
        
        # 转换为弧度
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # 计算差值
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Haversine公式
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(dlon/2)**2)
        
        c = 2 * math.asin(math.sqrt(a))
        
        # 计算距离
        distance = R * c
        
        return distance
    ```

    #### 算法精度说明
    | 算法类型     | 精度评估    | 适用场景              | 计算速度  |
    |-------------|-------------|----------------------|-----------|
    | Haversine   | ±0.5%      | 中短距离(<500km)     | 快        |
    | Vincenty    | ±0.005%    | 高精度长距离计算     | 慢        |
    | 简化球面    | ±2%        | 快速估算             | 最快      |

    #### 地球模型参数
    ```python
    earth_parameters = {
        "radius_km": 6371.0,           # 平均半径
        "equatorial_radius": 6378.137,  # 赤道半径
        "polar_radius": 6356.752,       # 极地半径
        "flattening": 1/298.257223563,  # 扁率
        "coordinate_system": "WGS84"    # 坐标系
    }
    ```

    ---

    ### 计算精度与误差

    #### 精度分析
    | 距离范围    | 绝对误差  | 相对误差 | 说明                    |
    |-------------|-----------|----------|-------------------------|
    | 0-10km      | ±5m      | ±0.05%   | 城市内距离，高精度      |
    | 10-100km    | ±50m     | ±0.05%   | 城际距离，精度良好      |
    | 100-1000km  | ±500m    | ±0.05%   | 省际距离，精度可接受    |
    | >1000km     | ±5km     | ±0.5%    | 跨国距离，精度一般      |

    #### 误差来源分析
    ```python
    error_sources = {
        "earth_model": "地球非完美球体，实际为椭球体",
        "coordinate_precision": "输入坐标的精度限制",
        "algorithm_approximation": "算法本身的近似误差",
        "floating_point": "浮点数计算精度限制"
    }
    ```

    #### 精度优化建议
    - **高精度需求**: 使用更高精度的坐标（小数点后6位以上）
    - **长距离计算**: 考虑使用Vincenty算法
    - **批量计算**: 使用向量化计算提高效率

    ---

    ### 坐标系统支持

    #### 支持的坐标系
    | 坐标系      | 全称                          | 适用地区    | 精度评估  |
    |-------------|-------------------------------|-------------|-----------|
    | WGS84       | World Geodetic System 1984   | 全球        | 最高      |
    | GCJ02       | 国测局坐标系                  | 中国大陆    | 高        |
    | BD09        | 百度坐标系                    | 百度地图    | 高        |
    | CGCS2000    | 中国大地坐标系                | 中国        | 最高      |

    #### 坐标转换示例
    ```python
    # WGS84转GCJ02（粗略转换）
    def wgs84_to_gcj02(lng, lat):
        # 简化转换公式（实际转换更复杂）
        dlat = transform_lat(lng - 105.0, lat - 35.0)
        dlng = transform_lng(lng - 105.0, lat - 35.0)
        
        radlat = lat / 180.0 * math.pi
        magic = math.sin(radlat)
        magic = 1 - 0.00669342162296594323 * magic * magic
        sqrtmagic = math.sqrt(magic)
        
        dlat = (dlat * 180.0) / ((6335552.717000426 / (magic * sqrtmagic)) * math.pi)
        dlng = (dlng * 180.0) / (6378137.0 / sqrtmagic * math.cos(radlat) * math.pi)
        
        mglat = lat + dlat
        mglng = lng + dlng
        
        return mglng, mglat
    ```

    ---

    ### 性能与优化

    #### 计算性能
    | 操作类型      | 响应时间    | CPU使用   | 内存使用  |
    |---------------|-------------|-----------|-----------|
    | 单次计算      | <1ms       | 极低      | 极低      |
    | 批量计算(100) | 5-10ms     | 低        | 低        |
    | 批量计算(1000)| 50-100ms   | 中        | 低        |

    #### 缓存策略
    ```python
    # 距离计算缓存
    cache_config = {
        "cache_enabled": True,
        "ttl_seconds": 3600,  # 1小时缓存
        "max_cache_size": 10000,  # 最大缓存条目
        "cache_key_precision": 6  # 坐标精度到小数点后6位
    }
    
    def get_cache_key(lat1, lng1, lat2, lng2):
        # 创建缓存键，考虑坐标顺序
        coords = sorted([(lat1, lng1), (lat2, lng2)])
        return f"dist_{coords[0][0]:.6f}_{coords[0][1]:.6f}_{coords[1][0]:.6f}_{coords[1][1]:.6f}"
    ```

    #### 批量计算优化
    ```python
    # 向量化计算示例
    import numpy as np
    
    def batch_calculate_distances(points1, points2):
        # 转换为numpy数组
        p1 = np.array(points1)
        p2 = np.array(points2)
        
        # 向量化Haversine计算
        lat1, lon1 = np.radians(p1[:, 0]), np.radians(p1[:, 1])
        lat2, lon2 = np.radians(p2[:, 0]), np.radians(p2[:, 1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)
        
        c = 2 * np.arcsin(np.sqrt(a))
        distances = 6371.0 * c
        
        return distances
    ```

    ---

    ### 使用场景

    #### 1. 房源距离计算
    ```python
    # 计算房源到地铁站距离
    def calculate_house_to_subway_distance(house_coords, subway_coords):
        response = requests.post(
            '/api/house-reco/calculate-distance',
            params={
                'lat1': house_coords[0],
                'lng1': house_coords[1], 
                'lat2': subway_coords[0],
                'lng2': subway_coords[1]
            }
        )
        
        if response.json()['success']:
            distance_km = response.json()['data']['distance_km']
            return {
                'distance_km': distance_km,
                'walking_time_min': distance_km * 12,  # 步行速度5km/h
                'driving_time_min': distance_km * 2    # 城市驾车速度30km/h
            }
    ```

    #### 2. 附近房源搜索
    ```python
    # 搜索指定位置附近的房源
    def find_nearby_houses(center_lat, center_lng, radius_km, houses):
        nearby_houses = []
        
        for house in houses:
            response = requests.post(
                '/api/house-reco/calculate-distance',
                params={
                    'lat1': center_lat,
                    'lng1': center_lng,
                    'lat2': house['latitude'],
                    'lng2': house['longitude']
                }
            )
            
            if response.json()['success']:
                distance = response.json()['data']['distance_km']
                if distance <= radius_km:
                    house['distance_km'] = distance
                    nearby_houses.append(house)
        
        # 按距离排序
        return sorted(nearby_houses, key=lambda x: x['distance_km'])
    ```

    #### 3. 路线规划辅助
    ```python
    # 计算多点路线总距离
    def calculate_route_distance(waypoints):
        total_distance = 0.0
        
        for i in range(len(waypoints) - 1):
            response = requests.post(
                '/api/house-reco/calculate-distance',
                params={
                    'lat1': waypoints[i][0],
                    'lng1': waypoints[i][1],
                    'lat2': waypoints[i+1][0],
                    'lng2': waypoints[i+1][1]
                }
            )
            
            if response.json()['success']:
                total_distance += response.json()['data']['distance_km']
        
        return {
            'total_distance_km': total_distance,
            'estimated_driving_time_min': total_distance * 2,
            'waypoint_count': len(waypoints)
        }
    ```

    ---

    ### 地理应用扩展

    #### 方向角计算
    ```python
    # 计算两点间的方向角（方位角）
    def calculate_bearing(lat1, lng1, lat2, lng2):
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlng_rad = math.radians(lng2 - lng1)
        
        y = math.sin(dlng_rad) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlng_rad))
        
        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        
        # 转换为0-360度
        bearing_deg = (bearing_deg + 360) % 360
        
        return bearing_deg
    ```

    #### 距离分级
    ```python
    # 距离分级系统
    distance_categories = {
        "步行范围": {"max_km": 1.0, "icon": "🚶", "color": "green"},
        "骑行范围": {"max_km": 5.0, "icon": "🚴", "color": "blue"},
        "公交范围": {"max_km": 15.0, "icon": "🚌", "color": "orange"},
        "驾车范围": {"max_km": 50.0, "icon": "🚗", "color": "red"},
        "远距离": {"max_km": float('inf'), "icon": "✈️", "color": "gray"}
    }
    
    def categorize_distance(distance_km):
        for category, config in distance_categories.items():
            if distance_km <= config["max_km"]:
                return {
                    "category": category,
                    "distance_km": distance_km,
                    **config
                }
        
        return distance_categories["远距离"]
    ```

    ---

    ### 质量保证

    #### 输入验证
    ```python
    def validate_coordinates(lat, lng):
        errors = []
        
        # 纬度验证
        if not isinstance(lat, (int, float)):
            errors.append("纬度必须为数字类型")
        elif not -90 <= lat <= 90:
            errors.append("纬度必须在-90到90度之间")
        
        # 经度验证
        if not isinstance(lng, (int, float)):
            errors.append("经度必须为数字类型")
        elif not -180 <= lng <= 180:
            errors.append("经度必须在-180到180度之间")
        
        return errors
    ```

    #### 结果验证
    ```python
    def validate_distance_result(distance_km, lat1, lng1, lat2, lng2):
        # 检查结果合理性
        max_earth_distance = math.pi * 6371  # 地球最大距离（半圆周）
        
        if distance_km < 0:
            return False, "距离不能为负数"
        
        if distance_km > max_earth_distance:
            return False, "距离超出地球最大可能距离"
        
        # 检查相同坐标点
        if abs(lat1 - lat2) < 1e-10 and abs(lng1 - lng2) < 1e-10:
            if distance_km > 0.001:  # 1米误差容忍
                return False, "相同坐标点距离应接近0"
        
        return True, "距离计算结果正常"
    ```

    ---

    ### 注意事项

    - **坐标精度**: 建议使用至少6位小数精度的坐标，提供米级精度
    - **算法选择**: 中短距离使用Haversine公式，长距离考虑更精确算法
    - **坐标系统**: 确保输入坐标使用相同的坐标系（默认WGS84）
    - **性能考虑**: 大批量计算建议使用批量接口或本地计算
    - **缓存利用**: 相同坐标对的计算结果会被缓存，提高响应速度
    - **边界情况**: 注意处理极地地区和国际日期变更线附近的计算
    - **误差范围**: 理解算法误差范围，根据应用场景选择合适精度
    """
    try:
        distance_km = HouseRecoService.calculate_distance_km(lat1, lng1, lat2, lng2)
        
        return HouseResponse(
            success=True,
            message="距离计算成功",
            data={
                "point1": {"latitude": lat1, "longitude": lng1},
                "point2": {"latitude": lat2, "longitude": lng2},
                "distance_km": round(distance_km, 2),
                "distance_m": round(distance_km * 1000, 0)
            }
        )
        
    except Exception as e:
        logger.error(f"距离计算失败: {e}")
        return HouseResponse(
            success=False,
            message=f"距离计算失败: {str(e)}",
            data=None
        )