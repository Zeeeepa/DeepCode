"""
简化项目结构分析器

本模块提供简洁的项目结构分析，只输出：
1. 项目目录结构
2. 函数定义（标准Python签名格式）
3. 类定义及其方法定义

输出格式简洁明了，专注于代码结构而非详细分析。

主要功能:
- analyze_project(): 分析项目并返回简洁结构
- format_function_signature(): 格式化函数签名
- extract_class_structure(): 提取类结构
"""

import os
import ast
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union


class SimpleStructureAnalyzer:
    """
    简化的项目结构分析器
    
    专注于提取项目的基本结构：目录树、函数签名、类定义。
    输出格式简洁，易于阅读和使用。
    
    属性:
        project_path (Path): 项目路径
        exclude_patterns (set): 排除的文件/目录模式
    """
    
    def __init__(self, project_path: str, exclude_patterns: Optional[set] = None):
        """
        初始化简化结构分析器
        
        参数:
            project_path (str): 项目根目录路径
            exclude_patterns (set, optional): 需要排除的文件模式
        """
        self.project_path = Path(project_path).resolve()
        self.exclude_patterns = exclude_patterns or {
            '__pycache__', '.git', '.svn', '.hg', 'node_modules',
            '.venv', 'venv', 'env', '.pytest_cache', '.mypy_cache',
            'build', 'dist', '.tox', '.idea', '.vscode'
        }
        
        if not self.project_path.exists():
            raise ValueError(f"项目路径不存在: {self.project_path}")
        if not self.project_path.is_dir():
            raise ValueError(f"项目路径不是目录: {self.project_path}")
    
    def analyze_project(self) -> Dict[str, Any]:
        """
        分析项目结构，返回简洁的结果
        
        返回:
            dict: 包含项目结构和代码定义的字典
        """
        print(f"分析项目: {self.project_path.name}")
        
        result = {
            'project_name': self.project_path.name,
            'project_path': str(self.project_path),
            'directory_structure': self._get_directory_tree(),
            'files': {}
        }
        
        # 分析所有Python文件
        python_files = list(self.project_path.rglob('*.py'))
        processed_files = 0
        
        for py_file in python_files:
            if self._should_exclude_file(py_file):
                continue
            
            relative_path = str(py_file.relative_to(self.project_path))
            file_structure = self._analyze_file(py_file)
            
            if file_structure:
                result['files'][relative_path] = file_structure
                processed_files += 1
        
        print(f"完成分析: {processed_files} 个Python文件")
        return result
    
    def save_result(self, result: Dict[str, Any], output_path: str) -> None:
        """
        保存分析结果到JSON文件
        
        参数:
            result (dict): 分析结果
            output_path (str): 输出文件路径
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存: {output_file}")
    
    def _get_directory_tree(self) -> List[str]:
        """获取目录树结构（包含所有文件类型）"""
        tree = []
        
        for root, dirs, files in os.walk(self.project_path):
            # 移除排除的目录
            dirs[:] = [d for d in dirs if d not in self.exclude_patterns]
            
            level = root.replace(str(self.project_path), '').count(os.sep)
            indent = '  ' * level
            rel_root = os.path.relpath(root, self.project_path)
            
            if rel_root != '.':
                tree.append(f"{indent}{os.path.basename(root)}/")
            
            # 添加所有文件（排除隐藏文件和特殊文件）
            filtered_files = [f for f in files if not f.startswith('.') and not f.endswith('.pyc')]
            sub_indent = '  ' * (level + 1)
            for file_name in sorted(filtered_files):
                tree.append(f"{sub_indent}{file_name}")
        
        return tree
    
    def _analyze_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """分析单个Python文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            
            file_info = {
                'functions': [],
                'classes': {}
            }
            
            # 分析顶级节点
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_sig = self._format_function_signature(node)
                    if func_sig:
                        file_info['functions'].append(func_sig)
                
                elif isinstance(node, ast.ClassDef):
                    class_info = self._extract_class_structure(node)
                    if class_info:
                        file_info['classes'][node.name] = class_info
            
            # 只返回有内容的文件信息
            if file_info['functions'] or file_info['classes']:
                return file_info
            
            return None
            
        except Exception as e:
            print(f"警告: 分析文件失败 {file_path.name}: {e}")
            return None
    
    def _format_function_signature(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
        """格式化函数签名为标准Python格式"""
        try:
            # 构建函数签名
            func_name = node.name
            args_parts = []
            
            # 处理参数
            args = node.args
            
            # 普通参数
            defaults_offset = len(args.args) - len(args.defaults)
            for i, arg in enumerate(args.args):
                arg_str = arg.arg
                
                # 添加类型注解
                if arg.annotation:
                    try:
                        type_str = ast.unparse(arg.annotation)
                        arg_str += f": {type_str}"
                    except:
                        pass
                
                # 添加默认值
                if i >= defaults_offset:
                    default_idx = i - defaults_offset
                    try:
                        default_str = ast.unparse(args.defaults[default_idx])
                        arg_str += f" = {default_str}"
                    except:
                        arg_str += " = <default>"
                
                args_parts.append(arg_str)
            
            # *args
            if args.vararg:
                vararg_str = f"*{args.vararg.arg}"
                if args.vararg.annotation:
                    try:
                        type_str = ast.unparse(args.vararg.annotation)
                        vararg_str += f": {type_str}"
                    except:
                        pass
                args_parts.append(vararg_str)
            
            # **kwargs
            if args.kwarg:
                kwarg_str = f"**{args.kwarg.arg}"
                if args.kwarg.annotation:
                    try:
                        type_str = ast.unparse(args.kwarg.annotation)
                        kwarg_str += f": {type_str}"
                    except:
                        pass
                args_parts.append(kwarg_str)
            
            # 构建完整签名
            args_str = ", ".join(args_parts)
            
            # 添加async前缀
            async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
            
            # 添加返回类型
            return_type = ""
            if node.returns:
                try:
                    return_type = f" -> {ast.unparse(node.returns)}"
                except:
                    return_type = " -> <return_type>"
            
            return f"{async_prefix}def {func_name}({args_str}){return_type}"
            
        except Exception as e:
            return f"def {node.name}(...)"  # 失败时返回简化格式
    
    def _extract_class_structure(self, node: ast.ClassDef) -> Dict[str, Any]:
        """提取类结构信息"""
        try:
            class_info = {
                'definition': self._format_class_definition(node),
                'methods': []
            }
            
            # 提取类中的方法
            for child_node in node.body:
                if isinstance(child_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_sig = self._format_function_signature(child_node)
                    if method_sig:
                        # 为类方法添加缩进
                        indented_sig = "    " + method_sig
                        class_info['methods'].append(indented_sig)
            
            return class_info
            
        except Exception as e:
            return {
                'definition': f"class {node.name}:",
                'methods': []
            }
    
    def _format_class_definition(self, node: ast.ClassDef) -> str:
        """格式化类定义"""
        try:
            class_name = node.name
            
            # 处理基类
            if node.bases:
                bases = []
                for base in node.bases:
                    try:
                        base_name = ast.unparse(base)
                        bases.append(base_name)
                    except:
                        bases.append("<base_class>")
                
                if bases:
                    return f"class {class_name}({', '.join(bases)}):"
            
            return f"class {class_name}:"
            
        except Exception:
            return f"class {node.name}:"
    
    def _should_exclude_file(self, file_path: Path) -> bool:
        """检查文件是否应该被排除"""
        # 检查是否在排除目录中
        for part in file_path.parts:
            if part in self.exclude_patterns:
                return True
        
        # 排除测试文件
        if (file_path.name.startswith('test_') or 
            file_path.name.endswith('_test.py')):
            return True
        
        return False





if __name__ == "__main__":
    #测试analyze_project
    project_path = "/Users/wwchdemac/python_projects/debug_agent/test_input/webpage"
    analyzer = SimpleStructureAnalyzer(project_path)
    result = analyzer.analyze_project()
    #以优雅的方式打印Json，使其在终端的可读性强
    print(json.dumps(result, indent=4, ensure_ascii=False))
        #最后统计终端打印的字符数量，要仔细统计JSON里面的所有字符
    print(f"终端打印的字符数量: {len(json.dumps(result, indent=4, ensure_ascii=False))}")