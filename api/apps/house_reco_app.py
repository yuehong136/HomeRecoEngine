"""
房源推荐API接口
"""
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
    搜索房源
    
    Args:
        request: 房源搜索请求
        
    Returns:
        搜索结果响应
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
    混合搜索房源 (语义搜索 + 过滤)
    
    Args:
        request: 混合搜索请求
        
    Returns:
        搜索结果响应
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
    插入房源数据
    
    Args:
        request: 房源插入请求
        
    Returns:
        插入结果响应
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
    获取集合统计信息
    
    Returns:
        统计信息响应
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
    上传Excel文件并导入房源数据
    
    Args:
        file: 上传的Excel文件
        
    Returns:
        导入结果响应
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
    清空所有房源数据
    
    Returns:
        清空结果响应
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