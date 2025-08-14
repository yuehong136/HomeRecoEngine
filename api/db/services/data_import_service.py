"""
数据导入服务
用于将Excel样例数据导入到Milvus集合中
"""
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
import re
import os
import hashlib

logger = logging.getLogger(__name__)


class DataImportService:
    """数据导入服务类"""
    
    # 字段映射关系（Excel中文字段名 -> Milvus英文字段名）
    FIELD_MAPPING = {
        '主键': 'id',
        '房源分类': 'category',
        '名称': 'name',
        '区县': 'region',
        '区域': 'region',  # 添加区域字段映射
        '地址': 'address',
        '最小面积': 'min_area',
        '最大面积': 'max_area',
        '最小单价': 'min_unit_price',
        '最大单价': 'max_unit_price',
        '最小总价': 'min_total_price',
        '最大总价': 'max_total_price',
        '租金': 'rent',
        '经度': 'longitude',
        '纬度': 'latitude',
        '房源类型': 'type',
        '建成年代': 'year_completion',
        '交易权属': 'transaction_ownership',
        '产权年限': 'property_right_duration',
        '车位比': 'parking_space_ratio',
        '物业公司': 'management_company',
        '物业费': 'management_fee',
        '开发商': 'developer',
        '绿化率': 'greening_rate',
        '容积率': 'plot_ratio',
        '装修风格': 'decoration_style',
        '装修情况': 'decoration_status',
        '水电': 'water_electricity',
        '有无电梯': 'has_elevator',
        '有无车位': 'has_parking',
        '朝向': 'orientation',
        '房屋年限': 'building_age',
        '家具设施': 'furniture_facilities',
        '楼层': 'floor',
        '租赁模式': 'rental_mode',
        '付款方式': 'payment_method',
        '租期': 'lease_term',
        '偏好与标签': 'preferences_tags',
        '偏好': 'preferences_tags',  # 添加偏好字段映射
        '房源标签': 'property_tags',  # 添加房源标签字段映射
        '房源户型': 'property_type',  # 添加房源户型字段映射
        '封面图': 'cover_url',  # 添加封面图字段映射
        '面积': 'area',  # 添加面积字段映射
        '单价': 'unit_price',  # 添加单价字段映射
        '总价': 'total_price',  # 添加总价字段映射
        '学区': 'school_district',  # 添加学区字段映射
        '小区特点': 'community_features',  # 添加小区特点字段映射
        '小区卖点': 'community_highlights',  # 添加小区卖点字段映射
        '周边': 'surrounding_area',  # 添加周边字段映射
        '语义字符': 'semantic_str',
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
            # 读取文件（支持 xlsx/xls/tsv/txt/csv）
            df = self._read_dataframe(excel_path)
            
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
            # 读取文件
            df = self._read_dataframe(excel_path)
            
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
                
                # 主键：若缺失则按关键字段生成稳定主键
                if not house_data.get('id'):
                    key_parts = [
                        str(house_data.get('category', '')),
                        str(house_data.get('name', '')),
                        str(house_data.get('region', '')),
                        str(house_data.get('address', '')),
                        str(house_data.get('longitude', '')),
                        str(house_data.get('latitude', '')),
                    ]
                    raw_key = '|'.join(key_parts)
                    house_data['id'] = hashlib.md5(raw_key.encode('utf-8')).hexdigest()

                # 核心必填字段校验（除 id 外）
                core_required_keys = ['name', 'region', 'address', 'longitude', 'latitude']
                missing_core = [k for k in core_required_keys if pd.isna(house_data.get(k)) or house_data.get(k) in [None, '']]
                if missing_core:
                    logger.warning(f"第{index+1}行数据缺少核心必填字段{missing_core}，已跳过")
                    continue
                
                # 为可选字段生成默认值
                if pd.isna(house_data.get('category')) or house_data.get('category') in [None, '']:
                    # 根据其他信息推断房源分类，默认为"住宅"
                    house_data['category'] = '住宅'
                    logger.debug(f"第{index+1}行数据缺失房源分类，设为默认值: 住宅")
                
                if pd.isna(house_data.get('semantic_str')) or house_data.get('semantic_str') in [None, '']:
                    # 生成语义字符串，组合主要特征
                    semantic_parts = []
                    if house_data.get('name'): semantic_parts.append(str(house_data['name']))
                    if house_data.get('region'): semantic_parts.append(str(house_data['region']))
                    if house_data.get('address'): semantic_parts.append(str(house_data['address']))
                    if house_data.get('area'): semantic_parts.append(f"面积{house_data['area']}")
                    if house_data.get('unit_price'): semantic_parts.append(f"单价{house_data['unit_price']}")
                    
                    house_data['semantic_str'] = ' '.join(semantic_parts) if semantic_parts else f"{house_data.get('name', '房源')} {house_data.get('region', '')} {house_data.get('address', '')}"
                    logger.debug(f"第{index+1}行数据缺失语义字符，已生成: {house_data['semantic_str'][:50]}...")
                
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

    def _detect_file_format(self, file_path: str) -> str:
        """通过文件头检测真实的文件格式"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)
            
            # Excel 2007+ (.xlsx) - ZIP 文件头
            if header.startswith(b'PK\x03\x04'):
                return 'xlsx'
            # Excel 97-2003 (.xls) - OLE 文件头
            elif header.startswith(b'\xd0\xcf\x11\xe0'):
                return 'xls'
            # CSV/TSV 文本文件 (尝试解码为文本)
            else:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        first_line = f.readline()
                    if '\t' in first_line:
                        return 'tsv'
                    elif ',' in first_line:
                        return 'csv'
                    else:
                        return 'txt'
                except:
                    return 'unknown'
        except Exception:
            return 'unknown'

    def _read_dataframe(self, file_path: str) -> pd.DataFrame:
        """根据扩展名和文件内容读取 DataFrame，支持 xlsx/xls/tsv/txt/csv。"""
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 检查文件大小
        if os.path.getsize(file_path) == 0:
            raise ValueError("文件为空")
        
        # 获取文件扩展名
        _, ext = os.path.splitext(file_path.lower())
        
        # 检测真实文件格式
        detected_format = self._detect_file_format(file_path)
        
        # 如果扩展名和实际格式不匹配，给出警告并使用检测到的格式
        if ext == '.xlsx' and detected_format != 'xlsx':
            logger.warning(f"文件 {file_path} 扩展名为 .xlsx 但实际格式为 {detected_format}")
        elif ext == '.xls' and detected_format != 'xls':
            logger.warning(f"文件 {file_path} 扩展名为 .xls 但实际格式为 {detected_format}")
        
        # 根据检测到的格式读取文件
        if detected_format == 'xlsx':
            try:
                return pd.read_excel(file_path, engine='openpyxl')
            except Exception as e:
                raise ValueError(f"无法读取 .xlsx 文件 '{file_path}': {str(e)}. 请确保文件格式正确且未损坏。")
        
        elif detected_format == 'xls':
            try:
                return pd.read_excel(file_path, engine='xlrd')
            except Exception as e:
                raise ValueError(f"无法读取 .xls 文件 '{file_path}': {str(e)}. 请确保文件格式正确且未损坏。")
        
        elif detected_format == 'tsv':
            try:
                return pd.read_csv(file_path, sep='\t', encoding='utf-8')
            except Exception as e:
                raise ValueError(f"无法读取 TSV 文件 '{file_path}': {str(e)}")
        
        elif detected_format == 'csv':
            try:
                return pd.read_csv(file_path, encoding='utf-8')
            except Exception as e:
                raise ValueError(f"无法读取 CSV 文件 '{file_path}': {str(e)}")
        
        else:
            # 最后的尝试 - 按原扩展名处理
            if ext in ['.xlsx', '.xls']:
                try:
                    engine = 'openpyxl' if ext == '.xlsx' else 'xlrd'
                    return pd.read_excel(file_path, engine=engine)
                except Exception as e:
                    raise ValueError(f"文件 '{file_path}' 格式无法识别或已损坏: {str(e)}. 请检查文件是否为有效的 Excel 文件。")
            
            # 尝试作为文本文件读取
            try:
                return pd.read_csv(file_path, encoding='utf-8')
            except Exception:
                try:
                    return pd.read_csv(file_path, sep='\t', encoding='utf-8')
                except Exception as e:
                    raise ValueError(f"无法识别文件格式: {file_path}. 支持的格式: .xlsx, .xls, .csv, .tsv, .txt. 错误: {str(e)}")
    
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
            df = self._read_dataframe(excel_path)
            
            validation_result = {
                'valid': True,
                'total_rows': len(df),
                'columns': df.columns.tolist(),
                'missing_fields': [],
                'extra_fields': [],
                'errors': []
            }
            
            # 检查必需字段（主键可选，缺失自动生成）
            # 对于某些字段，可以有替代字段名
            required_field_alternatives = {
                '名称': ['名称', 'name'],
                '区域': ['区域', '区县'],  # 区域和区县可以互相替代
                '地址': ['地址', 'address'],
                '经度': ['经度', 'longitude'],
                '纬度': ['纬度', 'latitude']
            }
            
            # 可选字段（如果缺失会自动生成或使用默认值）
            optional_field_alternatives = {
                '房源分类': ['房源分类', 'category'],  # 可以从其他信息推断
                '语义字符': ['语义字符', 'semantic_str']  # 可以从其他字段生成
            }
            
            for required_field, alternatives in required_field_alternatives.items():
                # 检查是否有任一替代字段存在
                if not any(alt in df.columns for alt in alternatives):
                    validation_result['missing_fields'].append(required_field)
                    validation_result['valid'] = False
            
            # 检查可选字段的缺失情况（仅记录警告，不影响valid状态）
            missing_optional_fields = []
            for optional_field, alternatives in optional_field_alternatives.items():
                if not any(alt in df.columns for alt in alternatives):
                    missing_optional_fields.append(optional_field)
            
            if missing_optional_fields:
                validation_result['warnings'] = [f"缺失可选字段: {', '.join(missing_optional_fields)}，将自动生成默认值"]
            
            # 检查额外字段（现在更宽松，只记录但不报错）
            expected_fields = set(self.FIELD_MAPPING.keys())
            actual_fields = set(df.columns)
            extra_fields = actual_fields - expected_fields
            if extra_fields:
                validation_result['extra_fields'] = list(extra_fields)
                # 不再因为额外字段而标记为无效
                # validation_result['valid'] = False
            
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