"""
文件概述：文件处理工具类
功能描述：处理文件读写操作，包含验证和安全错误

错误类型：验证和安全错误 (26-35)
26. 文件路径验证缺失
27. 权限检查缺失
28. 输入验证不足
29. 异常处理不当
30. 资源泄露
31. 路径遍历攻击漏洞
32. 文件大小限制缺失
33. 编码处理错误
34. 临时文件安全问题
35. 并发访问问题
"""

import os
import json
import tempfile
from typing import Any, Dict, List, Optional
from pathlib import Path

class FileHandler:
    """文件处理器"""
    
    def __init__(self, base_path: str = "."):
        # 错误26：文件路径验证缺失
        self.base_path = base_path  # 没有验证路径是否存在和合法
        self.temp_files: List[str] = []
    
    # 错误27：权限检查缺失
    def read_file(self, file_path: str) -> str:
        # 错误31：路径遍历攻击漏洞
        full_path = os.path.join(self.base_path, file_path)  # 没有防止../攻击
        
        # 错误27：没有检查文件读取权限
        # 错误29：异常处理不当
        try:
            with open(full_path, 'r') as f:
                content = f.read()
        except:  # 错误：捕获所有异常但没有具体处理
            content = ""  # 错误：静默失败，应该记录日志或抛出异常
        
        return content
    
    # 错误28：输入验证不足
    def write_file(self, file_path: str, content: str, encoding: str = "utf-8"):
        # 错误28：没有验证content是否为字符串
        # 错误32：文件大小限制缺失
        
        full_path = os.path.join(self.base_path, file_path)
        
        # 错误30：资源泄露 - 文件句柄可能没有正确关闭
        f = open(full_path, 'w', encoding=encoding)
        f.write(content)
        # 错误30：没有调用f.close()，可能导致资源泄露
    
    def read_json(self, file_path: str) -> Dict:
        content = self.read_file(file_path)
        
        # 错误33：编码处理错误
        try:
            # 错误：假设所有文件都是UTF-8编码
            return json.loads(content)
        except json.JSONDecodeError:
            return {}  # 错误29：异常处理不当，应该记录错误
    
    def write_json(self, file_path: str, data: Dict):
        # 错误28：输入验证不足 - 没有检查data是否可序列化
        content = json.dumps(data, indent=2)
        self.write_file(file_path, content)
    
    # 错误34：临时文件安全问题
    def create_temp_file(self, content: str) -> str:
        # 错误34：临时文件权限过于宽松
        temp_fd, temp_path = tempfile.mkstemp()  # 默认权限可能不安全
        
        # 错误30：资源泄露
        os.write(temp_fd, content.encode())  # 没有关闭文件描述符
        
        self.temp_files.append(temp_path)
        return temp_path
    
    def cleanup_temp_files(self):
        # 错误35：并发访问问题
        for temp_file in self.temp_files:  # 如果多线程访问可能有问题
            try:
                os.remove(temp_file)
            except FileNotFoundError:
                pass  # 错误29：静默忽略错误
        self.temp_files = []  # 错误35：清空列表时没有加锁

class DataProcessor:
    """数据处理器"""
    
    def __init__(self):
        self.file_handler = FileHandler()
    
    # 错误26：输入验证缺失
    def process_csv_data(self, file_path: str) -> List[Dict]:
        # 错误：没有验证文件是否为CSV格式
        content = self.file_handler.read_file(file_path)
        
        lines = content.split('\n')
        if not lines:
            return []
        
        # 错误28：假设第一行总是标题行
        headers = lines[0].split(',')
        data = []
        
        for line in lines[1:]:
            if line.strip():  # 错误：没有处理空行和格式错误
                values = line.split(',')
                # 错误：没有检查values长度是否与headers匹配
                row = dict(zip(headers, values))
                data.append(row)
        
        return data
    
    # 错误32：文件大小限制缺失
    def backup_data(self, data: Any, backup_path: str):
        # 错误：没有限制备份文件大小
        self.file_handler.write_json(backup_path, data)
