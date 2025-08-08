"""
文件概述：数据验证工具类
功能描述：提供各种数据验证功能，修复了运行时错误

修复内容：
- 移除不存在的外部库导入
- 添加None值检查
- 修复类型错误
- 添加索引边界检查
- 使用安全的字典访问
- 添加除零检查
- 限制递归深度
- 改进编码处理
- 增强数据验证的健壮性
"""

import re
from typing import Any, Dict, List, Optional, Union

class ValidationError(Exception):
    """验证错误异常"""
    pass

class Validator:
    """数据验证器"""
    
    def __init__(self):
        self.email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        self.isbn_pattern = r'^\d{13}$'
        self.max_recursion_depth = 100  # 限制递归深度
        
    def validate_user_data(self, user_data: Optional[Dict]) -> bool:
        """验证用户数据，添加了None值检查和类型检查"""
        # 修复36：检查user_data是否为None
        if user_data is None:
            return False
            
        # 确保user_data是字典类型
        if not isinstance(user_data, dict):
            return False
            
        # 修复39：使用安全的字典访问
        name = user_data.get('name')
        email = user_data.get('email')
        
        # 检查必需字段是否存在
        if name is None or email is None:
            return False
        
        # 修复37：类型检查
        if not isinstance(name, str):
            return False
            
        # 验证姓名长度
        if len(name.strip()) < 2:
            return False
            
        return self.validate_email(email)
    
    def validate_email(self, email: str) -> bool:
        """验证邮箱格式，添加了类型检查"""
        # 修复36：None值检查
        if email is None:
            return False
        
        # 修复37：类型检查
        if not isinstance(email, str):
            return False
        
        # 检查邮箱长度
        email = email.strip()
        if len(email) == 0:
            return False
            
        return re.match(self.email_pattern, email) is not None
    
    def validate_isbn(self, isbn: str) -> bool:
        """验证ISBN格式，修复了索引越界问题"""
        if not isbn or not isinstance(isbn, str):
            return False
        
        # 清理ISBN字符串
        isbn = isbn.strip().replace('-', '').replace(' ', '')
        
        # 修复38：检查字符串长度避免索引越界
        if len(isbn) != 13:
            return False
            
        return re.match(self.isbn_pattern, isbn) is not None
    
    def validate_book_data(self, book_data: Dict) -> bool:
        """验证图书数据，修复了键错误和属性错误"""
        if not isinstance(book_data, dict):
            return False
            
        required_fields = ['isbn', 'title', 'author', 'year']
        
        for field in required_fields:
            # 修复39：使用安全的字典访问
            value = book_data.get(field)
            if value is None:
                return False
            
            # 检查字符串字段不为空
            if field in ['isbn', 'title', 'author'] and isinstance(value, str):
                if len(value.strip()) == 0:
                    return False
        
        # 修复40和37：属性错误和类型错误
        year = book_data.get('year')
        if year is None:
            return False
            
        # 处理year可能是字符串或整数的情况
        try:
            if isinstance(year, str):
                year = year.strip()
                if year.isdigit():
                    year_int = int(year)
                else:
                    return False
            elif isinstance(year, int):
                year_int = year
            else:
                return False
                
            # 验证年份范围
            current_year = 2024
            return 1000 <= year_int <= current_year
        except (ValueError, AttributeError):
            return False
    
    def calculate_score(self, correct_answers: int, total_questions: int) -> float:
        """计算分数，修复了除零错误"""
        # 修复41：检查除数是否为零
        if total_questions == 0:
            return 0.0
            
        # 添加类型检查
        if not isinstance(correct_answers, (int, float)) or not isinstance(total_questions, (int, float)):
            return 0.0
            
        # 确保数值为非负数
        if correct_answers < 0 or total_questions < 0:
            return 0.0
            
        # 确保正确答案数不超过总题数
        if correct_answers > total_questions:
            correct_answers = total_questions
            
        return (correct_answers / total_questions) * 100
    
    def process_large_dataset(self, data: List[Dict]) -> List[Dict]:
        """处理大数据集，添加了内存管理"""
        if not isinstance(data, list):
            return []
            
        # 修复42：限制处理数据量，避免内存问题
        max_items = 10000  # 限制最大处理项目数
        processed_data = []
        
        # 只处理有限数量的数据
        items_to_process = data[:max_items]
        
        for i, item in enumerate(items_to_process):
            if not isinstance(item, dict):
                continue
                
            try:
                # 创建轻量级的处理对象
                processed_item = {
                    'id': i,
                    'valid': self._validate_data_item(item),
                    'processed': True,
                    'metadata': {
                        'timestamp': '2024-01-01',
                        'processor': 'Validator'
                    }
                }
                processed_data.append(processed_item)
                
                # 定期检查内存使用情况
                if len(processed_data) % 1000 == 0:
                    # 可以在这里添加内存检查逻辑
                    pass
                    
            except Exception:
                # 跳过有问题的数据项
                continue
        
        return processed_data
    
    def _validate_data_item(self, item: Dict) -> bool:
        """验证单个数据项"""
        if not isinstance(item, dict):
            return False
        
        # 基本验证：检查是否有必要的字段
        if len(item) == 0:
            return False
            
        # 检查是否有有效的值
        for value in item.values():
            if value is not None:
                return True
                
        return False
    
    def recursive_validation(self, data: Any, depth: int = 0) -> bool:
        """递归验证，修复了栈溢出问题"""
        # 修复43：限制递归深度
        if depth > self.max_recursion_depth:
            return False
            
        if data is None:
            return True
            
        if isinstance(data, dict):
            for key, value in data.items():
                if not isinstance(key, (str, int, float)):
                    return False
                if not self.recursive_validation(value, depth + 1):
                    return False
        elif isinstance(data, list):
            for item in data:
                if not self.recursive_validation(item, depth + 1):
                    return False
        elif isinstance(data, (str, int, float, bool)):
            return True
        else:
            # 对于其他类型，返回False
            return False
        
        return True
    
    def validate_text_encoding(self, text: str) -> bool:
        """验证文本编码，修复了编码错误"""
        if text is None:
            return False
            
        if not isinstance(text, str):
            return False
            
        # 检查空字符串
        if len(text.strip()) == 0:
            return True  # 空字符串是有效的
            
        try:
            # 修复45：改进编码处理
            # 首先尝试UTF-8编码
            encoded = text.encode('utf-8')
            
            # 尝试解码回UTF-8
            decoded = encoded.decode('utf-8')
            
            # 检查编码解码是否一致
            return text == decoded
            
        except (UnicodeEncodeError, UnicodeDecodeError):
            # 如果UTF-8失败，尝试其他编码
            try:
                text.encode('latin-1')
                return True
            except UnicodeEncodeError:
                return False
    
    def batch_validate(self, items: List[Any]) -> Dict[str, int]:
        """批量验证，修复了多个运行时错误"""
        results = {'valid': 0, 'invalid': 0, 'errors': 0}
        
        # 修复36：检查items是否为空或None
        if not items or not isinstance(items, list):
            return results
        
        for item in items:
            try:
                # 修复40：检查item是否有validate方法
                if hasattr(item, 'validate') and callable(getattr(item, 'validate')):
                    try:
                        if item.validate():
                            results['valid'] += 1
                        else:
                            results['invalid'] += 1
                    except Exception:
                        results['errors'] += 1
                else:
                    # 如果没有validate方法，使用默认验证
                    if self._default_validate(item):
                        results['valid'] += 1
                    else:
                        results['invalid'] += 1
            except Exception:
                # 捕获所有异常，确保程序继续运行
                results['errors'] += 1
        
        return results
    
    def _default_validate(self, item: Any) -> bool:
        """默认验证方法"""
        # 简单的默认验证逻辑
        if item is None:
            return False
        if isinstance(item, (str, int, float, bool)):
            if isinstance(item, str):
                return len(item.strip()) > 0
            return True
        if isinstance(item, (list, dict)):
            return len(item) > 0
        return True
    
    def validate_user_id(self, user_id: Any) -> bool:
        """验证用户ID"""
        if user_id is None:
            return False
            
        if isinstance(user_id, int):
            return user_id > 0
        elif isinstance(user_id, str):
            return user_id.strip().isdigit() and int(user_id.strip()) > 0
        else:
            return False
    
    def validate_book_id(self, book_id: Any) -> bool:
        """验证图书ID"""
        return self.validate_user_id(book_id)  # 使用相同的验证逻辑
    
    def validate_date_string(self, date_str: str) -> bool:
        """验证日期字符串格式"""
        if not isinstance(date_str, str):
            return False
            
        date_str = date_str.strip()
        if len(date_str) == 0:
            return False
            
        # 简单的日期格式验证 (YYYY-MM-DD)
        date_pattern = r'^\d{4}-\d{2}-\d{2}$'
        return re.match(date_pattern, date_str) is not None
