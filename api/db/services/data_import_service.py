"""
数据导入服务
用于将Excel样例数据导入到Milvus集合中
"""
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
import re

logger = logging.getLogger(__name__)


class DataImportService:
    """数据导入服务类"""
    
    # 字段映射关系（Excel中文字段名 -> Milvus英文字段名）
    FIELD_MAPPING = {
        '主键': 'id',
        '名称': 'xqmc',      # 小区名称
        '区县': 'qy',        # 区域  
        '地址': 'dz',        # 地址
        '经度': 'jd',        # 经度
        '纬度': 'wd',        # 纬度
        '产权年限': 'cqnx',  # 产权年限
        '绿化率': 'lhl',      # 绿化率
        '容积率': 'rjl',      # 容积率
        '装修风格': 'zxfg',   # 装修风格
        '装修情况': 'zxqk',   # 装修情况
        '水电': 'sd',         # 水电
        '有无电梯': 'ywdt',   # 有无电梯
        '面积': 'mj',         # 面积
        '朝向': 'cx',         # 朝向
        '单价': 'dj',         # 单价
        '总价': 'zj',         # 总价
        '物业费': 'wyf',       # 物业费
        '楼层': 'lc',         # 楼层
        '小区特点': 'xqtd',   # 小区特点
        '学区': 'xq',         # 学区
        '小区卖点': 'xqmd',   # 小区卖点
        '偏好': 'xqph',       # 小区偏好
        '房源标签': 'fyb',    # 房源标签
        '房源户型': 'fyhx',   # 房源户型
        '周边': 'zb'          # 周边环境
    }
    
    def __init__(self, house_reco_service):
        """
        初始化数据导入服务
        
        Args:
            house_reco_service: 房源推荐服务实例
        """
        self.house_service = house_reco_service
        
    def import_from_excel(self, excel_path: str) -> bool:
        """
        从 Excel文件导入数据
        
        Args:
            excel_path: Excel文件路径
            
        Returns:
            导入是否成功
        """
        try:
            # 读取TSV文件（实际上是制表符分隔的文本文件）
            df = pd.read_csv(excel_path, sep='\t', encoding='utf-8')
            
            logger.info(f"读取到 {len(df)} 条原始数据")
            
            # 处理重复主键：保留最后一条记录
            if '主键' in df.columns:
                duplicate_count = df['主键'].duplicated().sum()
                if duplicate_count > 0:
                    logger.info(f"发现 {duplicate_count} 个重复主键，保留最后出现的记录")
                    df = df.drop_duplicates(subset=['主键'], keep='last')
                    logger.info(f"去重后剩余 {len(df)} 条数据")
            
            # 数据清洗和转换
            house_data_list = self._transform_excel_data(df)
            
            if not house_data_list:
                logger.error("数据转换后为空")
                return False
            
            # 确保集合存在，如果不存在则自动创建
            if not self._ensure_collection_exists():
                logger.error("集合创建或检查失败")
                return False
            
            # 批量插入数据
            success = self.house_service.insert_house_data(house_data_list)
            
            if success:
                logger.info(f"成功导入{len(house_data_list)}条房源数据")
                return True
            else:
                logger.error("数据插入失败")
                return False
                
        except Exception as e:
            logger.error(f"导入Excel数据失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def preview_data(self, excel_path: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        预览Excel数据
        
        Args:
            excel_path: Excel文件路径
            limit: 预览数量限制
            
        Returns:
            预览数据列表
        """
        try:
            # 读取TSV文件
            df = pd.read_csv(excel_path, sep='\t', encoding='utf-8')
            
            # 处理重复主键：保留最后一条记录
            if '主键' in df.columns:
                duplicate_count = df['主键'].duplicated().sum()
                if duplicate_count > 0:
                    logger.info(f"预览时发现 {duplicate_count} 个重复主键，保留最后出现的记录")
                    df = df.drop_duplicates(subset=['主键'], keep='last')
            
            # 只取前 limit 条数据
            preview_df = df.head(limit)
            
            # 转换为字典列表
            house_data_list = self._transform_excel_data(preview_df)
            
            return house_data_list
            
        except Exception as e:
            logger.error(f"预览Excel数据失败: {e}")
            return []
    
    def _transform_excel_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        转换Excel数据为Milvus格式
        
        Args:
            df: pandas DataFrame
            
        Returns:
            转换后的房源数据列表
        """
        house_data_list = []
        
        for index, row in df.iterrows():
            try:
                house_data = {}
                
                # 基本字段映射
                for chinese_field, english_field in self.FIELD_MAPPING.items():
                    if chinese_field in df.columns:
                        value = row.get(chinese_field)
                        
                        # 处理空值
                        if pd.isna(value) or value == '' or str(value).strip() == '':
                            house_data[english_field] = '' if english_field in ['xqtd', 'xqmd', 'xq', 'xqph', 'zb', 'fyb', 'fyhx', 'user_query_text'] else None
                        else:
                            house_data[english_field] = value
                
                # 添加用户查询文本字段（空值）
                house_data['user_query_text'] = ''
                
                # 添加房屋年限字段（默认值）
                house_data['fwnx'] = 0
                
                # 添加有无车位字段（默认值）
                house_data['ywcw'] = ''
                
                # 特殊处理：小区名称可能有多个，用逗号分隔
                if 'xqmc' in house_data and house_data['xqmc']:
                    xqmc_value = str(house_data['xqmc'])
                    if ',' in xqmc_value:
                        # 已经是用逗号分隔的格式
                        house_data['xqmc'] = xqmc_value
                    else:
                        # 单个小区名称
                        house_data['xqmc'] = xqmc_value
                
                # 数据验证：确保主键存在
                if not house_data.get('id'):
                    logger.warning(f"第{index+1}行数据缺少主键，跳过")
                    continue
                
                house_data_list.append(house_data)
                
            except Exception as e:
                logger.error(f"转换第{index+1}行数据失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        logger.info(f"成功转换 {len(house_data_list)} 条数据")
        return house_data_list
    
    def get_field_mapping(self) -> Dict[str, str]:
        """
        获取字段映射关系
        
        Returns:
            字段映射字典
        """
        return self.FIELD_MAPPING.copy()
    
    def validate_excel_file(self, excel_path: str) -> Dict[str, Any]:
        """
        验证Excel文件格式和数据
        
        Args:
            excel_path: Excel文件路径
            
        Returns:
            验证结果字典
        """
        try:
            # 读取文件
            df = pd.read_csv(excel_path, sep='\t', encoding='utf-8')
            
            validation_result = {
                'valid': True,
                'total_rows': len(df),
                'columns': df.columns.tolist(),
                'missing_fields': [],
                'extra_fields': [],
                'errors': []
            }
            
            # 检查必需字段
            required_fields = ['主键', '名称', '区县']
            for field in required_fields:
                if field not in df.columns:
                    validation_result['missing_fields'].append(field)
                    validation_result['valid'] = False
            
            # 检查额外字段
            expected_fields = set(self.FIELD_MAPPING.keys())
            actual_fields = set(df.columns)
            extra_fields = actual_fields - expected_fields
            if extra_fields:
                validation_result['extra_fields'] = list(extra_fields)
            
            # 检查数据质量
            if validation_result['valid']:
                # 检查主键重复情况（改为警告而非错误）
                if '主键' in df.columns:
                    duplicate_ids = df['主键'].duplicated().sum()
                    if duplicate_ids > 0:
                        validation_result['warnings'] = validation_result.get('warnings', [])
                        validation_result['warnings'].append(f"发现 {duplicate_ids} 个重复的主键，将使用最后出现的记录")
                        validation_result['duplicate_count'] = duplicate_ids
            
            return validation_result
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'total_rows': 0,
                'columns': [],
                'missing_fields': [],
                'extra_fields': [],
                'errors': [str(e)]
            }
    
    def get_sample_data_structure(self) -> Dict[str, str]:
        """
        获取样例数据结构说明
        
        Returns:
            数据结构说明字典
        """
        return {
            '主键': '房源唯一标识符（数字）',
            '名称': '小区名称，多个名称用逗号分隔',
            '区县': '所在区县',
            '地址': '详细地址',
            '经度': '地理经度（数字）',
            '纬度': '地理纬度（数字）',
            '产权年限': '产权年限（数字）',
            '绿化率': '绿化率百分比，如"41%"',
            '容积率': '容积率（数字）',
            '装修风格': '装修风格描述',
            '装修情况': '装修情况描述',
            '水电': '水电情况描述',
            '有无电梯': '是否有电梯（有/无）',
            '面积': '房屋面积，如"89.5平方米"',
            '朝向': '房屋朝向',
            '单价': '单价，如"15000元/平方米"',
            '总价': '总价，如"150万元"',
            '物业费': '物业费，如"2.5元/m²/月"',
            '楼层': '楼层情况，如"低楼层"、"高楼层"',
            '小区特点': '小区特点描述',
            '学区': '学区信息',
            '小区卖点': '小区主要卖点',
            '偏好': '小区偏好描述',
            '房源标签': '房源相关标签',
            '房源户型': '房屋户型，如"三室两厅一卫"',
            '周边': '周边环境和配套设施描述'
        }
    
    def get_import_statistics(self) -> Dict[str, Any]:
        """
        获取导入统计信息
        
        Returns:
            统计信息字典
        """
        try:
            # 获取集合统计信息
            stats = self.house_service.get_collection_stats()
            
            return {
                'total_houses': stats.get('row_count', 0),
                'collection_name': stats.get('collection_name', ''),
                'import_status': 'ready' if stats.get('created', False) else 'not_ready',
                'field_mapping': self.FIELD_MAPPING
            }
            
        except Exception as e:
            logger.error(f"获取导入统计信息失败: {e}")
            return {
                'total_houses': 0,
                'collection_name': '',
                'import_status': 'error',
                'error': str(e),
                'field_mapping': self.FIELD_MAPPING
            }
    
    def clear_all_data(self) -> bool:
        """
        清空所有房源数据（重新创建集合）
        
        Returns:
            清空是否成功
        """
        try:
            # 删除现有集合
            if self.house_service.client.has_collection(self.house_service.COLLECTION_NAME):
                self.house_service.client.drop_collection(self.house_service.COLLECTION_NAME)
                logger.info("已删除现有集合")
            
            # 重新创建集合
            success = self.house_service.create_collection()
            if success:
                logger.info("成功清空所有房源数据")
                return True
            else:
                logger.error("重新创建集合失败")
                return False
            
        except Exception as e:
            logger.error(f"清空数据失败: {e}")
            return False
    
    def import_from_dict_list(self, house_data_list: List[Dict[str, Any]]) -> bool:
        """
        从字典列表导入数据
        
        Args:
            house_data_list: 房源数据字典列表
            
        Returns:
            导入是否成功
        """
        try:
            if not house_data_list:
                logger.warning("输入的房源数据列表为空")
                return False
            
            # 确保集合存在，如果不存在则自动创建
            if not self._ensure_collection_exists():
                logger.error("集合创建或检查失败")
                return False
            
            # 直接使用房源服务插入数据
            success = self.house_service.insert_house_data(house_data_list)
            
            if success:
                logger.info(f"成功导入{len(house_data_list)}条房源数据")
                return True
            else:
                logger.error("数据插入失败")
                return False
                
        except Exception as e:
            logger.error(f"从字典列表导入数据失败: {e}")
            return False
    
    def update_house_data(self, house_id: int, updated_data: Dict[str, Any]) -> bool:
        """
        更新房源数据
        
        Args:
            house_id: 房源ID
            updated_data: 更新的数据
            
        Returns:
            更新是否成功
        """
        try:
            # Milvus不支持直接更新，需要先删除再插入
            
            # 1. 删除原数据
            delete_success = self.house_service.delete_house(house_id)
            if not delete_success:
                logger.error(f"删除原房源 {house_id} 失败")
                return False
            
            # 2. 插入新数据
            updated_data['id'] = house_id
            insert_success = self.house_service.insert_house_data(updated_data)
            
            if insert_success:
                logger.info(f"成功更新房源 {house_id}")
                return True
            else:
                logger.error(f"更新房源 {house_id} 失败")
                return False
                
        except Exception as e:
            logger.error(f"更新房源数据失败: {e}")
            return False
    
    def _ensure_collection_exists(self) -> bool:
        """
        确保集合存在，如果不存在则自动创建
        
        Returns:
            集合是否存在或创建成功
        """
        try:
            # 检查集合是否已存在
            if self.house_service.client.has_collection(self.house_service.COLLECTION_NAME):
                logger.info(f"集合 {self.house_service.COLLECTION_NAME} 已存在")
                return True
            
            # 集合不存在，自动创建
            logger.info(f"集合 {self.house_service.COLLECTION_NAME} 不存在，正在自动创建...")
            success = self.house_service.create_collection()
            
            if success:
                logger.info(f"成功自动创建集合 {self.house_service.COLLECTION_NAME}")
                return True
            else:
                logger.error(f"自动创建集合 {self.house_service.COLLECTION_NAME} 失败")
                return False
                
        except Exception as e:
            logger.error(f"检查或创建集合失败: {e}")
            return False