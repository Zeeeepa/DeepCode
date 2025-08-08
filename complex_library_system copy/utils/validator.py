"""
文件概述：数据验证工具类
功能描述：提供各种数据验证功能，包含运行时错误

错误类型：运行时错误 (36-45)
36. 空指针/None值访问
37. 类型错误
38. 索引越界
39. 键错误
40. 属性错误
41. 除零错误
42. 内存错误（大数据处理）
43. 栈溢出
44. 导入错误
45. 编码错误
"""

import re
from typing import Any, Dict, List, Optional, Union
# 错误44：导入错误 - 导入可能不存在的模块
from some_external_lib import validate_email  # 错误：模块不存在

class ValidationError(Exception):
    """验证错误异常"""
    pass

class Validator:
    """数据验证器"""
    
    def __init__(self):
        self.email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        self.isbn_pattern = r'^\d{13}$'
        
    # 错误36：空指针/None值访问
    def validate_user_data(self, user_data: Optional[Dict]) -> bool:
        # 错误36：没有检查user_data是否为None
        name = user_data['name']  # 如果user_data为None会报错
        email = user_data['email']
        
        # 错误37：类型错误
        if len(name) < 2:  # 如果name不是字符串会报错
            return False
            
        return self.validate_email(email)
    
    def validate_email(self, email: str) -> bool:
        # 错误36：None值访问
        if email is None:
            return False
        
        # 错误37：类型错误 - 假设email总是字符串
        return re.match(self.email_pattern, email) is not None
    
    # 错误38：索引越界
    def validate_isbn(self, isbn: str) -> bool:
        if not isbn:
            return False
        
        # 错误38：没有检查字符串长度
        if isbn[13] == 'X':  # 错误：索引可能越界
            return False
            
        return re.match(self.isbn_pattern, isbn) is not None
    
    # 错误39：键错误
    def validate_book_data(self, book_data: Dict) -> bool:
        required_fields = ['isbn', 'title', 'author', 'year']
        
        for field in required_fields:
            # 错误39：直接访问键，可能不存在
            value = book_data[field]  # 应该使用get()方法
            if not value:
                return False
        
        # 错误40：属性错误
        year = book_data['year']
        # 错误37：类型错误 - 假设year总是整数
        if year.isdigit():  # 错误：如果year是int会报AttributeError
            return int(year) > 1000
        
        return False
    
    # 错误41：除零错误
    def calculate_score(self, correct_answers: int, total_questions: int) -> float:
        # 错误41：没有检查除数是否为零
        return correct_answers / total_questions * 100
    
    # 错误42：内存错误（大数据处理）
    def process_large_dataset(self, data: List[Dict]) -> List[Dict]:
        # 错误42：没有考虑内存限制，可能处理过大的数据集
        processed_data = []
        
        for item in data:
            # 创建大量中间对象，可能导致内存问题
            processed_item = {
                'original': item,
                'processed': True,
                'metadata': {
                    'timestamp': '2024-01-01',
                    'processor': 'Validator',
                    'large_field': 'x' * 10000  # 错误：创建大字符串
                }
            }
            processed_data.append(processed_item)
        
        return processed_data
    
    # 错误43：栈溢出
    def recursive_validation(self, data: Any, depth: int = 0) -> bool:
        # 错误43：没有限制递归深度
        if isinstance(data, dict):
            for key, value in data.items():
                if not self.recursive_validation(value, depth + 1):
                    return False
        elif isinstance(data, list):
            for item in data:
                if not self.recursive_validation(item, depth + 1):
                    return False
        
        return True
    
    # 错误45：编码错误
    def validate_text_encoding(self, text: str) -> bool:
        try:
            # 错误45：假设文本总是UTF-8编码
            encoded = text.encode('utf-8')
            decoded = encoded.decode('ascii')  # 错误：强制转换为ASCII可能失败
            return True
        except UnicodeDecodeError:
            return False
    
    # 更多运行时错误示例
    def batch_validate(self, items: List[Any]) -> Dict[str, int]:
        results = {'valid': 0, 'invalid': 0}
        
        # 错误36：假设items不为空
        first_item = items[0]  # 错误：如果items为空列表会IndexError
        
        for item in items:
            try:
                # 错误40：属性错误 - 假设所有item都有validate方法
                if item.validate():  # 错误：item可能没有validate方法
                    results['valid'] += 1
                else:
                    results['invalid'] += 1
            except AttributeError:
                results['invalid'] += 1
        
        return results
