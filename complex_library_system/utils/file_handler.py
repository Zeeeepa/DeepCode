"""
文件概述：文件处理工具类
功能描述：处理文件读写操作，包含验证和安全错误

修复内容：
- 添加了完整的路径验证和权限检查
- 实现了防路径遍历攻击的安全机制
- 改进了异常处理和错误日志记录
- 修复了资源泄露问题
- 添加了文件大小限制
- 改进了编码处理
- 修复了临时文件安全问题
- 添加了并发访问保护
"""

import os
import json
import tempfile
import logging
import threading
from typing import Any, Dict, List, Optional
from pathlib import Path
import stat

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileSecurityError(Exception):
    """文件安全相关异常"""
    pass

class FileHandler:
    """文件处理器 - 修复了所有安全和验证问题"""
    
    # 文件大小限制（10MB）
    MAX_FILE_SIZE = 10 * 1024 * 1024
    
    def __init__(self, base_path: str = "."):
        # 修复错误26：添加文件路径验证
        self.base_path = self._validate_base_path(base_path)
        self.temp_files: List[str] = []
        self._lock = threading.Lock()  # 修复错误35：添加并发访问保护
    
    def _validate_base_path(self, path: str) -> str:
        """验证基础路径的安全性和有效性"""
        if not path or not isinstance(path, str):
            raise ValueError("基础路径必须是非空字符串")
        
        # 转换为绝对路径并规范化
        abs_path = os.path.abspath(path)
        
        # 检查路径是否存在
        if not os.path.exists(abs_path):
            try:
                os.makedirs(abs_path, mode=0o755, exist_ok=True)
                logger.info(f"创建基础目录: {abs_path}")
            except OSError as e:
                raise FileSecurityError(f"无法创建基础目录 {abs_path}: {e}")
        
        # 检查是否为目录
        if not os.path.isdir(abs_path):
            raise FileSecurityError(f"基础路径不是目录: {abs_path}")
        
        return abs_path
    
    def _validate_file_path(self, file_path: str) -> str:
        """验证文件路径，防止路径遍历攻击"""
        if not file_path or not isinstance(file_path, str):
            raise ValueError("文件路径必须是非空字符串")
        
        # 修复路径验证逻辑，支持相对路径和绝对路径
        try:
            # 如果是绝对路径，检查是否在基础目录内
            if os.path.isabs(file_path):
                abs_path = os.path.abspath(file_path)
                # 确保绝对路径在基础目录内
                if not abs_path.startswith(self.base_path):
                    raise FileSecurityError(f"绝对路径超出基础目录范围: {file_path}")
                return abs_path
            
            # 处理相对路径
            # 规范化路径，解析所有的 . 和 .. 组件
            normalized_path = os.path.normpath(file_path)
            
            # 检查规范化后的路径是否试图逃出当前目录
            if normalized_path.startswith('..') or normalized_path.startswith('/'):
                raise FileSecurityError(f"检测到路径遍历攻击尝试: {file_path}")
            
            # 检查路径中是否包含危险的路径分隔符组合
            dangerous_patterns = ['../', '..\\', '/../', '\\..\\']
            for pattern in dangerous_patterns:
                if pattern in file_path:
                    raise FileSecurityError(f"检测到危险路径模式: {file_path}")
            
            # 构建完整路径
            full_path = os.path.join(self.base_path, normalized_path)
            full_path = os.path.abspath(full_path)
            
            # 最终安全检查：确保解析后的路径仍在基础目录内
            if not full_path.startswith(self.base_path):
                raise FileSecurityError(f"路径解析后超出基础目录范围: {file_path}")
            
            return full_path
            
        except OSError as e:
            raise FileSecurityError(f"路径验证失败: {e}")
    
    def _check_file_permissions(self, file_path: str, mode: str) -> None:
        """修复错误27：检查文件权限"""
        try:
            if mode == 'r':
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    raise FileSecurityError(f"文件不存在: {file_path}")
                
                # 检查是否为文件（不是目录）
                if not os.path.isfile(file_path):
                    raise FileSecurityError(f"路径不是文件: {file_path}")
                
                # 检查读取权限
                if not os.access(file_path, os.R_OK):
                    raise FileSecurityError(f"没有文件读取权限: {file_path}")
                    
            elif mode == 'w':
                # 检查目录是否存在和可写
                dir_path = os.path.dirname(file_path)
                if not os.path.exists(dir_path):
                    # 尝试创建目录
                    try:
                        os.makedirs(dir_path, mode=0o755, exist_ok=True)
                    except OSError as e:
                        raise FileSecurityError(f"无法创建目录 {dir_path}: {e}")
                
                if not os.access(dir_path, os.W_OK):
                    raise FileSecurityError(f"没有目录写入权限: {dir_path}")
                
                # 如果文件存在，检查文件写权限
                if os.path.exists(file_path):
                    if not os.path.isfile(file_path):
                        raise FileSecurityError(f"路径不是文件: {file_path}")
                    if not os.access(file_path, os.W_OK):
                        raise FileSecurityError(f"没有文件写入权限: {file_path}")
                        
        except OSError as e:
            raise FileSecurityError(f"权限检查失败: {e}")
    
    def _check_file_size(self, file_path: str) -> None:
        """修复错误32：检查文件大小限制"""
        try:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                if file_size > self.MAX_FILE_SIZE:
                    raise FileSecurityError(f"文件大小超过限制 ({file_size} > {self.MAX_FILE_SIZE}): {file_path}")
        except OSError as e:
            raise FileSecurityError(f"无法获取文件大小: {e}")
    
    def read_file(self, file_path: str, encoding: str = "utf-8") -> str:
        """安全地读取文件内容"""
        try:
            # 验证路径和权限
            full_path = self._validate_file_path(file_path)
            
            # 检查权限和大小
            self._check_file_permissions(full_path, 'r')
            self._check_file_size(full_path)
            
            # 修复错误29和30：改进异常处理和资源管理
            with open(full_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            logger.info(f"成功读取文件: {file_path}")
            return content
            
        except (FileNotFoundError, FileSecurityError, ValueError) as e:
            logger.error(f"读取文件失败 {file_path}: {e}")
            raise
        except UnicodeDecodeError as e:
            logger.error(f"文件编码错误 {file_path}: {e}")
            raise FileSecurityError(f"文件编码错误: {e}")
        except OSError as e:
            logger.error(f"文件系统错误 {file_path}: {e}")
            raise FileSecurityError(f"文件系统错误: {e}")
    
    def write_file(self, file_path: str, content: str, encoding: str = "utf-8") -> None:
        """安全地写入文件内容"""
        # 修复错误28：输入验证
        if not isinstance(content, str):
            raise ValueError("内容必须是字符串类型")
        
        # 修复错误32：检查内容大小
        try:
            content_size = len(content.encode(encoding))
            if content_size > self.MAX_FILE_SIZE:
                raise FileSecurityError(f"内容大小超过限制: {content_size} > {self.MAX_FILE_SIZE}")
        except UnicodeEncodeError as e:
            raise FileSecurityError(f"内容编码错误: {e}")
        
        try:
            # 验证路径和权限
            full_path = self._validate_file_path(file_path)
            
            # 检查权限
            self._check_file_permissions(full_path, 'w')
            
            # 修复错误30：使用with语句确保资源正确释放
            with open(full_path, 'w', encoding=encoding) as f:
                f.write(content)
            
            # 设置安全的文件权限
            try:
                os.chmod(full_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)
            except OSError:
                # 在某些系统上可能无法设置权限，记录警告但不抛出异常
                logger.warning(f"无法设置文件权限: {full_path}")
            
            logger.info(f"成功写入文件: {file_path}")
            
        except (FileSecurityError, ValueError) as e:
            logger.error(f"写入文件失败 {file_path}: {e}")
            raise
        except OSError as e:
            logger.error(f"文件系统错误 {file_path}: {e}")
            raise FileSecurityError(f"文件系统错误: {e}")
    
    def read_json(self, file_path: str, encoding: str = "utf-8") -> Dict:
        """安全地读取JSON文件"""
        try:
            content = self.read_file(file_path, encoding)
            
            if not content.strip():
                logger.warning(f"JSON文件为空: {file_path}")
                return {}
            
            # 修复错误33：改进JSON解析错误处理
            data = json.loads(content)
            logger.info(f"成功解析JSON文件: {file_path}")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON格式错误 {file_path}: {e}")
            raise FileSecurityError(f"JSON格式错误: {e}")
        except Exception as e:
            logger.error(f"读取JSON文件失败 {file_path}: {e}")
            raise
    
    def write_json(self, file_path: str, data: Any, encoding: str = "utf-8") -> None:
        """安全地写入JSON文件"""
        try:
            # 修复错误28：验证数据是否可序列化
            content = json.dumps(data, indent=2, ensure_ascii=False)
            self.write_file(file_path, content, encoding)
            logger.info(f"成功写入JSON文件: {file_path}")
            
        except (TypeError, ValueError) as e:
            logger.error(f"JSON序列化失败 {file_path}: {e}")
            raise FileSecurityError(f"数据无法序列化为JSON: {e}")
        except Exception as e:
            logger.error(f"写入JSON文件失败 {file_path}: {e}")
            raise
    
    def create_temp_file(self, content: str, encoding: str = "utf-8") -> str:
        """修复错误34：安全地创建临时文件"""
        if not isinstance(content, str):
            raise ValueError("内容必须是字符串类型")
        
        # 检查内容大小
        try:
            content_size = len(content.encode(encoding))
            if content_size > self.MAX_FILE_SIZE:
                raise FileSecurityError(f"临时文件内容过大: {content_size} > {self.MAX_FILE_SIZE}")
        except UnicodeEncodeError as e:
            raise FileSecurityError(f"临时文件内容编码错误: {e}")
        
        try:
            # 修复错误34：设置安全的临时文件权限
            temp_fd, temp_path = tempfile.mkstemp(
                prefix="filehandler_",
                suffix=".tmp",
                dir=tempfile.gettempdir()
            )
            
            try:
                # 修复错误30：正确管理文件描述符
                with os.fdopen(temp_fd, 'w', encoding=encoding) as f:
                    f.write(content)
                
                # 设置安全权限（仅所有者可读写）
                try:
                    os.chmod(temp_path, stat.S_IRUSR | stat.S_IWUSR)
                except OSError:
                    logger.warning(f"无法设置临时文件权限: {temp_path}")
                
                # 修复错误35：线程安全地添加到列表
                with self._lock:
                    self.temp_files.append(temp_path)
                
                logger.info(f"创建临时文件: {temp_path}")
                return temp_path
                
            except Exception:
                # 如果写入失败，确保关闭文件描述符
                try:
                    os.close(temp_fd)
                except OSError:
                    pass
                # 删除已创建的临时文件
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
                raise
                
        except OSError as e:
            logger.error(f"创建临时文件失败: {e}")
            raise FileSecurityError(f"无法创建临时文件: {e}")
    
    def cleanup_temp_files(self) -> None:
        """修复错误35：线程安全地清理临时文件"""
        with self._lock:
            failed_files = []
            
            for temp_file in self.temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        logger.info(f"删除临时文件: {temp_file}")
                except OSError as e:
                    logger.warning(f"删除临时文件失败 {temp_file}: {e}")
                    failed_files.append(temp_file)
            
            # 只保留删除失败的文件
            self.temp_files = failed_files
            
            if failed_files:
                logger.warning(f"有 {len(failed_files)} 个临时文件删除失败")

class DataProcessor:
    """数据处理器 - 修复了验证和安全问题"""
    
    def __init__(self, base_path: str = "."):
        self.file_handler = FileHandler(base_path)
    
    def process_csv_data(self, file_path: str, encoding: str = "utf-8") -> List[Dict]:
        """安全地处理CSV数据"""
        # 修复错误26：添加输入验证
        if not file_path or not isinstance(file_path, str):
            raise ValueError("文件路径必须是非空字符串")
        
        # 验证文件扩展名
        if not file_path.lower().endswith('.csv'):
            raise ValueError("文件必须是CSV格式")
        
        try:
            content = self.file_handler.read_file(file_path, encoding)
            
            if not content.strip():
                logger.warning(f"CSV文件为空: {file_path}")
                return []
            
            lines = content.strip().split('\n')
            if len(lines) < 2:  # 至少需要标题行和一行数据
                logger.warning(f"CSV文件数据不足: {file_path}")
                return []
            
            # 处理标题行
            headers = [h.strip() for h in lines[0].split(',')]
            if not headers or not all(headers):
                raise ValueError("CSV标题行格式错误")
            
            data = []
            for line_num, line in enumerate(lines[1:], start=2):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                values = [v.strip() for v in line.split(',')]
                
                # 修复：检查列数是否匹配
                if len(values) != len(headers):
                    logger.warning(f"第{line_num}行列数不匹配，跳过: {line}")
                    continue
                
                row = dict(zip(headers, values))
                data.append(row)
            
            logger.info(f"成功处理CSV文件: {file_path}, 共{len(data)}行数据")
            return data
            
        except Exception as e:
            logger.error(f"处理CSV文件失败 {file_path}: {e}")
            raise
    
    def backup_data(self, data: Any, backup_path: str) -> None:
        """修复错误32：安全地备份数据"""
        if not backup_path or not isinstance(backup_path, str):
            raise ValueError("备份路径必须是非空字符串")
        
        try:
            # 先序列化检查大小
            json_content = json.dumps(data, indent=2, ensure_ascii=False)
            content_size = len(json_content.encode('utf-8'))
            
            if content_size > self.file_handler.MAX_FILE_SIZE:
                raise FileSecurityError(f"备份数据过大: {content_size} > {self.file_handler.MAX_FILE_SIZE}")
            
            self.file_handler.write_json(backup_path, data)
            logger.info(f"数据备份完成: {backup_path}")
            
        except Exception as e:
            logger.error(f"数据备份失败 {backup_path}: {e}")
            raise
