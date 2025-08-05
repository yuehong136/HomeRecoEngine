"""
房源推荐系统使用示例
演示如何使用HouseRecoService和DataImportService
"""
from pymilvus import MilvusClient
from .house_reco_service import HouseRecoService
from .data_import_service import DataImportService
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_create_collection():
    """示例：创建集合"""
    # 初始化Milvus客户端
    client = MilvusClient(
        uri="http://localhost:19530",  # Milvus服务地址
        user="",                       # 用户名 (可选)
        password="",                   # 密码 (可选)
        db_name=""                     # 数据库名 (可选)
    )
    
    # 创建房源推荐服务
    house_service = HouseRecoService(client)
    
    # 创建集合
    success = house_service.create_collection()
    
    if success:
        logger.info("集合创建成功！")
    else:
        logger.error("集合创建失败！")
    
    return house_service


def example_import_data(house_service: HouseRecoService):
    """示例：导入样例数据"""
    # 创建数据导入服务
    import_service = DataImportService(house_service)
    
    # 预览数据
    excel_path = "reference/样例数据.xlsx"
    preview_data = import_service.preview_data(excel_path, limit=3)
    
    logger.info("预览导入数据:")
    for i, house in enumerate(preview_data, 1):
        logger.info(f"房源{i}: {house['xqmc']} - {house['qy']} - {house['mj']}平方米")
    
    # 导入全部数据
    success = import_service.import_from_excel(excel_path)
    
    if success:
        logger.info("数据导入成功！")
    else:
        logger.error("数据导入失败！")


def example_search_houses(house_service: HouseRecoService):
    """示例：搜索房源"""
    
    # 示例1：基本过滤搜索
    logger.info("\n=== 示例1：基本过滤搜索 ===")
    search_params = {
        "qy": "亭湖区",  # 区域过滤
        "area_range": {"min_area": 50, "max_area": 100},  # 面积范围
        "lc": "低楼层"   # 楼层过滤
    }
    
    results = house_service.search_houses(search_params, limit=5)
    logger.info(f"找到{len(results)}条符合条件的房源")
    
    # 示例2：小区名称多选
    logger.info("\n=== 示例2：小区名称搜索 ===")
    search_params = {
        "xqmc": ["新河湾花园", "琥珀湾"]  # 小区名称多选
    }
    
    results = house_service.search_houses(search_params, limit=5)
    logger.info(f"找到{len(results)}条符合条件的房源")
    
    # 示例3：价格范围搜索
    logger.info("\n=== 示例3：价格范围搜索 ===")
    search_params = {
        "price_range": {"min_price": 0, "max_price": 100},  # 总价范围 (万元)
        "area_range": {"min_area": 30, "max_area": 80}      # 面积范围
    }
    
    results = house_service.search_houses(search_params, limit=5)
    logger.info(f"找到{len(results)}条符合条件的房源")
    
    # 示例4：语义搜索
    logger.info("\n=== 示例4：语义搜索 ===")
    search_params = {
        "user_query_text": "我想要一个学区房，交通便利，周边配套齐全"
    }
    
    results = house_service.search_houses(search_params, limit=5)
    logger.info(f"语义搜索找到{len(results)}条相关房源")


def example_hybrid_search(house_service: HouseRecoService):
    """示例：混合搜索"""
    logger.info("\n=== 混合搜索示例 ===")
    
    # 组合语义查询和过滤条件
    semantic_query = "安静的小区，适合居住，有好的学区"
    filter_params = {
        "qy": "亭湖区",
        "area_range": {"min_area": 40, "max_area": 120}
    }
    
    results = house_service.hybrid_search(
        semantic_query=semantic_query,
        filter_params=filter_params,
        semantic_weight=0.7,  # 语义权重70%
        limit=10
    )
    
    logger.info(f"混合搜索找到{len(results)}条匹配房源")


def example_insert_single_house(house_service: HouseRecoService):
    """示例：插入单条房源数据"""
    logger.info("\n=== 插入单条房源示例 ===")
    
    house_data = {
        "id": 9999,
        "xqmc": ["测试小区"],
        "qy": "测试区域",
        "dz": "测试地址123号",
        "jd": 120.15,
        "wd": 33.40,
        "mj": 89.5,
        "fyhx": "三室两厅",
        "lc": "中楼层",
        "zj": 150.0,
        "dj": 16760.0,
        "xqtd": "环境优美，配套齐全",
        "xqmd": "地段优越，投资首选",
        "zb": "靠近地铁站，交通便利",
        "user_query_text": ""
    }
    
    success = house_service.insert_house_data(house_data)
    
    if success:
        logger.info("单条房源插入成功！")
    else:
        logger.error("单条房源插入失败！")


def example_get_house_detail(house_service: HouseRecoService):
    """示例：获取房源详情"""
    logger.info("\n=== 获取房源详情示例 ===")
    
    house_id = 1  # 假设查询ID为1的房源
    house = house_service.get_house_by_id(house_id)
    
    if house:
        logger.info(f"房源详情: {house['xqmc']} - {house['qy']} - {house['mj']}平方米")
    else:
        logger.warning(f"未找到ID为{house_id}的房源")


def example_collection_stats(house_service: HouseRecoService):
    """示例：获取集合统计信息"""
    logger.info("\n=== 集合统计信息示例 ===")
    
    stats = house_service.get_collection_stats()
    logger.info(f"集合统计信息: {stats}")


def run_complete_example():
    """运行完整示例"""
    logger.info("开始房源推荐系统示例...")
    
    try:
        # 1. 创建集合
        house_service = example_create_collection()
        
        # 2. 导入数据
        example_import_data(house_service)
        
        # 3. 搜索示例
        example_search_houses(house_service)
        
        # 4. 混合搜索
        example_hybrid_search(house_service)
        
        # 5. 插入单条数据
        example_insert_single_house(house_service)
        
        # 6. 获取详情
        example_get_house_detail(house_service)
        
        # 7. 统计信息
        example_collection_stats(house_service)
        
        logger.info("示例运行完成！")
        
    except Exception as e:
        logger.error(f"示例运行失败: {e}")


# 参数字典示例说明
def example_search_parameter_dictionary():
    """
    搜索参数字典示例和说明
    这个函数展示了所有支持的搜索参数格式
    """
    
    # 完整的搜索参数示例
    complete_search_params = {
        # === 过滤字段 ===
        
        # 小区名称 (支持多选)
        "xqmc": ["新河湾花园", "琥珀湾", "其他小区"],
        
        # 区域
        "qy": "亭湖区",
        
        # 价格范围 (万元)
        "price_range": {
            "min_price": 50.0,   # 最低价格
            "max_price": 200.0   # 最高价格
        },
        
        # 面积范围 (平方米)
        "area_range": {
            "min_area": 60.0,    # 最小面积
            "max_area": 120.0    # 最大面积
        },
        
        # 地理位置范围
        "location_range": {
            "min_jd": 120.0,     # 最小经度
            "max_jd": 121.0,     # 最大经度
            "min_wd": 33.0,      # 最小纬度
            "max_wd": 34.0       # 最大纬度
        },
        
        # 楼层
        "lc": "低楼层",  # 或 "高楼层"、"中楼层"
        
        # 装修情况
        "zxqk": "毛坯",  # 或 "精装修"、"简装修"
        
        # 朝向
        "cx": "南",      # 或 "北"、"东"、"西"、"南北"等
        
        # 有无电梯
        "ywdt": "无",    # 或 "有"
        
        # 有无车位
        "ywcw": "有",    # 或 "无"
        
        # 产权年限范围
        "cqnx_range": {
            "min_years": 50,     # 最小产权年限
            "max_years": 70      # 最大产权年限
        },
        
        # === 语义字段 ===
        
        # 用户查询文本 (自然语言描述)
        "user_query_text": "我想要一个安静的小区，有好的学区，交通便利，周边配套齐全"
    }
    
    # 不同场景的搜索参数示例
    scenarios = {
        
        # 场景1: 刚需购房 - 关注价格和面积
        "just_need": {
            "price_range": {"min_price": 0, "max_price": 100},
            "area_range": {"min_area": 60, "max_area": 90},
            "user_query_text": "经济实用的房子，适合小家庭居住"
        },
        
        # 场景2: 改善型购房 - 关注品质和环境
        "improvement": {
            "price_range": {"min_price": 150, "max_price": 300},
            "area_range": {"min_area": 100, "max_area": 150},
            "zxqk": "精装修",
            "ywdt": "有",
            "user_query_text": "高品质小区，环境优美，配套完善"
        },
        
        # 场景3: 学区房 - 关注教育资源
        "school_district": {
            "user_query_text": "优质学区房，靠近好学校，教育资源丰富"
        },
        
        # 场景4: 投资购房 - 关注地段和升值潜力
        "investment": {
            "user_query_text": "地段优越，交通便利，有升值潜力的房产"
        },
        
        # 场景5: 地理位置搜索 - 在特定区域内
        "location_based": {
            "qy": "亭湖区",
            "location_range": {
                "min_jd": 120.0, "max_jd": 120.5,
                "min_wd": 33.3, "max_wd": 33.5
            },
            "user_query_text": "在指定区域内的优质房源"
        }
    }
    
    return complete_search_params, scenarios


if __name__ == "__main__":
    # 运行完整示例
    run_complete_example()
    
    # 显示参数字典说明
    complete_params, scenarios = example_search_parameter_dictionary()
    logger.info("\n=== 搜索参数字典格式说明 ===")
    logger.info("支持的所有参数类型和格式请参考代码中的示例")