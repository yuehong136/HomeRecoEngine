from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
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
    """房源搜索请求模型"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    # 过滤字段
    xqmc: Optional[List[str]] = Field(None, description="小区名称列表")
    qy: Optional[str] = Field(None, description="区域")
    
    # 价格范围
    price_range: Optional[Dict[str, float]] = Field(
        None, 
        description="价格范围 {'min_price': 最低价, 'max_price': 最高价}"
    )
    
    # 面积范围
    area_range: Optional[Dict[str, float]] = Field(
        None,
        description="面积范围 {'min_area': 最小面积, 'max_area': 最大面积}"
    )
    
    # 地理位置搜索 (支持圆形区域和矩形区域)
    location: Optional[Dict[str, float]] = Field(
        None,
        description="""地理位置搜索，支持两种模式：
        1. 圆形区域: {'center_longitude': 中心经度, 'center_latitude': 中心纬度, 'radius_km': 半径公里数}
        2. 矩形区域: {'min_longitude': 最小经度, 'max_longitude': 最大经度, 'min_latitude': 最小纬度, 'max_latitude': 最大纬度}"""
    )
    
    # 房屋特征
    lc: Optional[str] = Field(None, description="楼层 (高楼层/低楼层)")
    zxqk: Optional[str] = Field(None, description="装修情况")
    cx: Optional[str] = Field(None, description="朝向")
    ywdt: Optional[str] = Field(None, description="有无电梯")
    ywcw: Optional[str] = Field(None, description="有无车位")
    
    # 产权年限范围
    cqnx_range: Optional[Dict[str, int]] = Field(
        None,
        description="产权年限范围 {'min_years': 最小年限, 'max_years': 最大年限}"
    )
    
    # 语义查询字段 (用户的自然语言需求描述)
    user_query_text: Optional[str] = Field(
        None, 
        description="用户查询描述，如：'我想要一个有学区的三室两厅，交通便利，周边配套齐全的房子'"
    )
    
    # 搜索参数
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
    """房源插入请求模型"""
    model_config = {"extra": "forbid", "validate_assignment": True}
    
    # 必填字段
    id: int = Field(..., description="房源主键ID")
    
    # 小区信息
    xqmc: List[str] = Field(..., description="小区名称列表")
    qy: str = Field(..., description="区域")
    dz: str = Field(..., description="详细地址")
    jd: float = Field(..., description="经度")
    wd: float = Field(..., description="纬度")
    
    # 房屋基本信息
    mj: float = Field(..., description="面积(平方米)")
    fyhx: str = Field(..., description="房源户型")
    lc: str = Field(..., description="楼层情况")
    cx: Optional[str] = Field(None, description="朝向")
    
    # 价格信息
    zj: float = Field(..., description="总价(万元)")
    dj: Optional[float] = Field(None, description="单价(元/平方米)")
    wyf: Optional[float] = Field(None, description="物业费(元/平方米/月)")
    
    # 房屋特征
    cqnx: Optional[int] = Field(None, description="产权年限")
    zxfg: Optional[str] = Field(None, description="装修风格")
    zxqk: Optional[str] = Field(None, description="装修情况")
    sd: Optional[str] = Field(None, description="水电情况")
    ywdt: Optional[str] = Field(None, description="有无电梯")
    ywcw: Optional[str] = Field(None, description="有无车位")
    fwnx: Optional[int] = Field(None, description="房屋年限")
    
    # 小区特征
    lhl: Optional[float] = Field(None, description="绿化率(%)")
    rjl: Optional[float] = Field(None, description="容积率")
    
    # 语义字段
    xqtd: Optional[str] = Field(None, description="小区特点")
    xqmd: Optional[str] = Field(None, description="小区卖点")
    xq: Optional[str] = Field(None, description="学区")
    xqph: Optional[str] = Field(None, description="小区偏好")
    zb: Optional[str] = Field(None, description="周边环境")
    fyb: Optional[str] = Field(None, description="房源标签")
    
    # 用户查询文本 (可选，用于个性化推荐)
    user_query_text: Optional[str] = Field(None, description="相关查询文本")


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
    基于多种条件进行房源搜索，支持小区名称、区域、价格、面积、地理位置、房屋特征等多维度过滤。
    支持语义搜索功能，能够理解用户的自然语言需求描述。

    ---

    ### 请求体 (Request Body)

    | 字段           | 类型                  | 必填 | 描述                                                                    |
    |----------------|----------------------|------|-------------------------------------------------------------------------|
    | `xqmc`         | `List[str]`          | 否   | 小区名称列表，支持多个小区同时搜索                                       |
    | `qy`           | `str`                | 否   | 区域名称                                                               |
    | `price_range`  | `Dict[str, float]`   | 否   | 价格范围: {"min_price": 最低价, "max_price": 最高价}（单位：万元）        |
    | `area_range`   | `Dict[str, float]`   | 否   | 面积范围: {"min_area": 最小面积, "max_area": 最大面积}（单位：平方米）    |
    | `location`     | `Dict[str, float]`   | 否   | 地理位置搜索，支持圆形和矩形区域                                         |
    | `lc`           | `str`                | 否   | 楼层情况（如：高楼层、低楼层）                                           |
    | `zxqk`         | `str`                | 否   | 装修情况                                                               |
    | `cx`           | `str`                | 否   | 房屋朝向                                                               |
    | `ywdt`         | `str`                | 否   | 有无电梯                                                               |
    | `ywcw`         | `str`                | 否   | 有无车位                                                               |
    | `cqnx_range`   | `Dict[str, int]`     | 否   | 产权年限范围: {"min_years": 最小年限, "max_years": 最大年限}             |
    | `user_query_text` | `str`             | 否   | 用户自然语言查询描述                                                    |
    | `limit`        | `int`                | 否   | 返回结果数量限制（1-100，默认10）                                       |
    | `offset`       | `int`                | 否   | 结果偏移量（默认0）                                                     |

    ---

    ### 地理位置搜索详细说明

    #### 圆形区域搜索
    ```json
    {
        "location": {
            "center_longitude": 116.3974,
            "center_latitude": 39.9093,
            "radius_km": 2.5
        }
    }
    ```

    #### 矩形区域搜索
    ```json
    {
        "location": {
            "min_longitude": 116.3,
            "max_longitude": 116.5,
            "min_latitude": 39.8,
            "max_latitude": 40.0
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
                    "id": 123,
                    "xqmc": ["金地格林小镇"],
                    "qy": "朝阳区",
                    "mj": 89.5,
                    "fyhx": "三室两厅",
                    "zj": 650.0,
                    "dj": 72600,
                    "similarity_score": 0.95
                }
            ],
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

    ### 搜索算法说明

    1. **精确过滤**: 首先根据结构化字段进行精确匹配过滤
    2. **地理位置计算**: 使用Haversine公式计算地理距离
    3. **语义搜索**: 对user_query_text进行向量化并计算相似度
    4. **结果排序**: 综合考虑匹配度和相似度进行排序
    5. **分页返回**: 根据limit和offset参数返回指定范围的结果

    ---

    ### 使用场景

    #### 1. 基础条件搜索
    ```json
    {
        "qy": "朝阳区",
        "price_range": {"min_price": 500, "max_price": 800},
        "area_range": {"min_area": 80, "max_area": 120},
        "limit": 20
    }
    ```

    #### 2. 地理位置搜索
    ```json
    {
        "location": {
            "center_longitude": 116.4074,
            "center_latitude": 39.9042,
            "radius_km": 3.0
        },
        "limit": 15
    }
    ```

    #### 3. 语义搜索
    ```json
    {
        "user_query_text": "我想要一个有学区的三室两厅，交通便利，周边配套齐全的房子",
        "price_range": {"max_price": 1000},
        "limit": 10
    }
    ```

    ---

    ### 注意事项

    - **性能优化**: 建议合理设置limit参数，避免一次性获取过多数据
    - **地理搜索**: 圆形搜索适合以某点为中心的需求，矩形搜索适合特定区域范围
    - **语义理解**: user_query_text会进行向量化处理，支持自然语言描述
    - **结果排序**: 搜索结果按相关性得分降序排列
    - **数据一致性**: 所有过滤条件采用AND逻辑组合
    """
    try:
        # 获取服务实例
        service = get_house_reco_service()
        
        # 构建搜索参数
        search_params = {}
        
        # 添加过滤条件
        if request.xqmc:
            search_params["xqmc"] = request.xqmc
        if request.qy:
            search_params["qy"] = request.qy
        if request.price_range:
            search_params["price_range"] = request.price_range
        if request.area_range:
            search_params["area_range"] = request.area_range
        if request.location:
            search_params["location"] = request.location
        if request.lc:
            search_params["lc"] = request.lc
        if request.zxqk:
            search_params["zxqk"] = request.zxqk
        if request.cx:
            search_params["cx"] = request.cx
        if request.ywdt:
            search_params["ywdt"] = request.ywdt
        if request.ywcw:
            search_params["ywcw"] = request.ywcw
        if request.cqnx_range:
            search_params["cqnx_range"] = request.cqnx_range
        if request.user_query_text:
            search_params["user_query_text"] = request.user_query_text
        
        # 执行搜索
        results = service.search_houses(
            search_params=search_params,
            limit=request.limit,
            offset=request.offset
        )
        
        # 额外的序列化保护
        safe_results = []
        for result in results:
            try:
                # 确保每个结果都是可序列化的
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
    向系统中插入单个房源数据，支持完整的房源信息录入。
    系统会自动进行数据验证、向量化处理，并存储到Milvus向量数据库中。

    ---

    ### 请求体 (Request Body)

    #### 必填字段
    | 字段    | 类型          | 描述                           |
    |---------|---------------|-------------------------------|
    | `id`    | `int`         | 房源主键ID（全局唯一）         |
    | `xqmc`  | `List[str]`   | 小区名称列表                   |
    | `qy`    | `str`         | 区域名称                       |
    | `dz`    | `str`         | 详细地址                       |
    | `jd`    | `float`       | 经度坐标                       |
    | `wd`    | `float`       | 纬度坐标                       |
    | `mj`    | `float`       | 面积（平方米）                 |
    | `fyhx`  | `str`         | 房源户型                       |
    | `lc`    | `str`         | 楼层情况                       |
    | `zj`    | `float`       | 总价（万元）                   |

    #### 可选字段
    | 字段    | 类型     | 描述                          |
    |---------|----------|-------------------------------|
    | `cx`    | `str`    | 朝向                          |
    | `dj`    | `float`  | 单价（元/平方米）             |
    | `wyf`   | `float`  | 物业费（元/平方米/月）        |
    | `cqnx`  | `int`    | 产权年限                      |
    | `zxfg`  | `str`    | 装修风格                      |
    | `zxqk`  | `str`    | 装修情况                      |
    | `sd`    | `str`    | 水电情况                      |
    | `ywdt`  | `str`    | 有无电梯                      |
    | `ywcw`  | `str`    | 有无车位                      |
    | `fwnx`  | `int`    | 房屋年限                      |
    | `lhl`   | `float`  | 绿化率（%）                   |
    | `rjl`   | `float`  | 容积率                        |

    #### 语义字段（用于智能搜索）
    | 字段      | 类型  | 描述                         |
    |-----------|-------|------------------------------|
    | `xqtd`    | `str` | 小区特点                     |
    | `xqmd`    | `str` | 小区卖点                     |
    | `xq`      | `str` | 学区信息                     |
    | `xqph`    | `str` | 小区偏好                     |
    | `zb`      | `str` | 周边环境                     |
    | `fyb`     | `str` | 房源标签                     |
    | `user_query_text` | `str` | 相关查询文本        |

    ---

    ### 响应 (Response)

    #### 成功响应 (200)
    ```json
    {
        "success": true,
        "message": "房源数据插入成功",
        "data": {
            "house_id": 123456
        }
    }
    ```

    #### 错误响应 (500)
    ```json
    {
        "success": false,
        "message": "插入失败: 具体错误信息",
        "data": null
    }
    ```

    ---

    ### 请求示例

    #### 基础房源信息插入
    ```json
    {
        "id": 123456,
        "xqmc": ["万科城市花园"],
        "qy": "海淀区",
        "dz": "海淀区中关村大街39号",
        "jd": 116.3105,
        "wd": 39.9785,
        "mj": 89.5,
        "fyhx": "三室两厅",
        "lc": "中楼层(共30层)",
        "zj": 650.0,
        "dj": 72625,
        "cx": "南北",
        "zxqk": "精装修",
        "ywdt": "有电梯",
        "ywcw": "有车位"
    }
    ```

    #### 包含语义信息的房源插入
    ```json
    {
        "id": 123457,
        "xqmc": ["金地格林小镇"],
        "qy": "朝阳区",
        "dz": "朝阳区建国路88号",
        "jd": 116.4634,
        "wd": 39.9078,
        "mj": 105.0,
        "fyhx": "三室两厅一卫",
        "lc": "高楼层(共25层)",
        "zj": 780.0,
        "xqtd": "高端社区，环境优美，绿化率高",
        "xqmd": "地铁1号线直达，周边配套完善",
        "xq": "朝阳区重点小学学区",
        "zb": "邻近国贸商圈，购物便利，交通发达"
    }
    ```

    ---

    ### 数据处理流程

    1. **数据验证**: 检查必填字段完整性和数据格式正确性
    2. **坐标验证**: 验证经纬度坐标的有效性
    3. **语义处理**: 对语义字段进行向量化处理
    4. **数据入库**: 将结构化数据和向量数据存储到Milvus
    5. **索引更新**: 更新相关索引以支持快速搜索

    ---

    ### 字段说明与建议

    #### 地理坐标
    - 经度(jd): 中国境内一般在73-135度之间
    - 纬度(wd): 中国境内一般在18-54度之间
    - 建议使用高精度坐标（小数点后6位）

    #### 价格信息
    - 总价(zj): 单位为万元，建议保留1位小数
    - 单价(dj): 单位为元/平方米，可通过总价和面积自动计算

    #### 语义字段优化
    - 详细描述有助于提高搜索匹配准确性
    - 建议包含周边配套、交通、教育、环境等信息
    - 避免过于简单或重复的描述

    ---

    ### 注意事项

    - **ID唯一性**: 房源ID必须全局唯一，重复ID会导致插入失败
    - **坐标准确性**: 地理坐标直接影响位置搜索的准确性
    - **数据一致性**: 建议单价与总价面积保持一致
    - **语义优化**: 丰富的语义描述能显著提升智能搜索效果
    - **存储空间**: 系统会自动为语义字段生成向量，占用额外存储空间
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
    批量插入房源数据
    
    Args:
        houses: 房源数据列表
        
    Returns:
        批量插入结果响应
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
    获取房源详情
    
    Args:
        house_id: 房源ID
        
    Returns:
        房源详情响应
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
    删除房源
    
    Args:
        house_id: 房源ID
        
    Returns:
        删除结果响应
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
    预览Excel文件数据
    
    Args:
        file: 上传的Excel文件
        limit: 预览行数限制
        
    Returns:
        预览数据响应
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
    获取导入统计信息
    
    Returns:
        导入统计信息响应
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
    创建房源推荐集合
    
    Returns:
        创建结果响应
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
    计算两个地理坐标点之间的距离
    
    Args:
        lat1, lng1: 第一个点的纬度和经度
        lat2, lng2: 第二个点的纬度和经度
        
    Returns:
        距离计算结果
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