"""
基于Code2flow的函数调用流程图分析器

本模块封装了code2flow工具的功能，用于生成Python项目的函数调用图。
Code2flow是一个强大的工具，可以分析动态语言的函数调用关系并生成可视化图表。

主要功能:
- analyze_function_calls(): 分析函数调用关系
- generate_call_graph(): 生成函数调用图
- extract_function_info(): 提取函数信息
- create_flow_diagram(): 创建流程图
"""

import os
import subprocess
import tempfile
import json
import ast
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
import sys
import re


class Code2FlowAnalyzer:
    """
    基于Code2flow的函数调用分析器
    
    使用code2flow工具分析Python项目的函数调用关系，
    生成详细的函数调用图和流程图。
    
    属性:
        repo_path (Path): 待分析的仓库路径
        exclude_patterns (list): 需要排除的文件模式
        analysis_result (dict): 分析结果缓存
    """
    
    def __init__(self, repo_path: str, exclude_patterns: Optional[List[str]] = None):
        """
        初始化Code2flow分析器
        
        参数:
            repo_path (str): 待分析的仓库路径
            exclude_patterns (list, optional): 需要排除的文件模式
        """
        self.repo_path = Path(repo_path).resolve()
        self.exclude_patterns = exclude_patterns or [
            '*/test*', '*/.*', '*/__pycache__/*', '*/venv/*', '*/env/*'
        ]
        self.analysis_result = {}
        
        # 检查路径有效性
        if not self.repo_path.exists():
            raise ValueError(f"仓库路径不存在: {self.repo_path}")
        if not self.repo_path.is_dir():
            raise ValueError(f"仓库路径不是目录: {self.repo_path}")
    
    def analyze_function_calls(self, target_function: Optional[str] = None, 
                             max_depth: int = 3) -> Dict[str, Any]:
        """
        分析函数调用关系
        
        参数:
            target_function (str, optional): 目标函数名，如果指定则以此为中心分析
            max_depth (int): 分析深度，默认为3
        
        返回:
            dict: 包含函数调用分析结果的字典
        """
        print(f"开始分析函数调用关系: {self.repo_path}")
        
        # 1. 获取Python文件列表
        python_files = self._get_python_files()
        
        # 2. 使用code2flow生成调用图
        call_graph_data = self._generate_code2flow_graph(
            python_files, target_function, max_depth
        )
        
        # 3. 解析AST获取函数详细信息
        function_details = self._extract_function_details(python_files)
        
        # 4. 合并分析结果
        self.analysis_result = {
            'repo_path': str(self.repo_path),
            'target_function': target_function,
            'max_depth': max_depth,
            'call_graph': call_graph_data,
            'function_details': function_details,
            'statistics': self._generate_statistics(call_graph_data, function_details),
            'analysis_timestamp': self._get_timestamp()
        }
        
        print("函数调用关系分析完成!")
        return self.analysis_result
    
    def generate_call_graph(self, output_path: str, format: str = 'svg',
                           target_function: Optional[str] = None, 
                           upstream_depth: int = 1, downstream_depth: int = 2) -> str:
        """
        使用code2flow生成函数调用图
        
        参数:
            output_path (str): 输出文件路径
            format (str): 输出格式 ('svg', 'png', 'dot')
            target_function (str, optional): 目标函数名
            upstream_depth (int): 上游深度
            downstream_depth (int): 下游深度
        
        返回:
            str: 生成的文件路径
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 获取Python文件
        python_files = self._get_python_files()
        
        if not python_files:
            raise Exception("未找到Python文件")
        
        # 构建code2flow命令
        cmd = ['code2flow'] + [str(f) for f in python_files]
        
        # 添加参数
        if target_function:
            cmd.extend(['--target-function', target_function])
            cmd.extend(['--upstream-depth', str(upstream_depth)])
            cmd.extend(['--downstream-depth', str(downstream_depth)])
        
        # 设置输出格式
        if format == 'dot':
            # DOT格式直接输出到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
        else:
            # SVG/PNG格式需要graphviz处理
            try:
                # 先生成DOT
                dot_result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=60
                )
                
                if dot_result.returncode != 0:
                    raise Exception(f"code2flow执行失败: {dot_result.stderr}")
                
                # 使用graphviz转换为指定格式
                dot_cmd = ['dot', f'-T{format}', '-o', str(output_file)]
                graphviz_result = subprocess.run(
                    dot_cmd, input=dot_result.stdout, text=True, 
                    capture_output=True, timeout=30
                )
                
                if graphviz_result.returncode != 0:
                    raise Exception(f"graphviz转换失败: {graphviz_result.stderr}")
                
            except subprocess.TimeoutExpired:
                raise Exception("code2flow执行超时")
        
        print(f"函数调用图已生成: {output_file}")
        return str(output_file)
    
    def extract_function_info(self, file_path: str) -> Dict[str, Any]:
        """
        提取指定文件的函数信息
        
        参数:
            file_path (str): 文件路径
        
        返回:
            dict: 函数信息字典
        """
        file_path = Path(file_path)
        if not file_path.exists() or file_path.suffix != '.py':
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            functions = {}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'line_start': node.lineno,
                        'line_end': getattr(node, 'end_lineno', node.lineno),
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node),
                        'calls': self._extract_function_calls_from_node(node),
                        'is_async': isinstance(node, ast.AsyncFunctionDef)
                    }
                    functions[node.name] = func_info
                
                elif isinstance(node, ast.ClassDef):
                    # 处理类中的方法
                    class_name = node.name
                    for class_node in node.body:
                        if isinstance(class_node, ast.FunctionDef):
                            method_name = f"{class_name}.{class_node.name}"
                            func_info = {
                                'name': method_name,
                                'class': class_name,
                                'line_start': class_node.lineno,
                                'line_end': getattr(class_node, 'end_lineno', class_node.lineno),
                                'args': [arg.arg for arg in class_node.args.args],
                                'docstring': ast.get_docstring(class_node),
                                'calls': self._extract_function_calls_from_node(class_node),
                                'is_async': isinstance(class_node, ast.AsyncFunctionDef)
                            }
                            functions[method_name] = func_info
            
            return {
                'file_path': str(file_path),
                'functions': functions,
                'total_functions': len(functions)
            }
            
        except Exception as e:
            print(f"警告: 解析文件失败 {file_path}: {e}")
            return {}
    
    def create_flow_diagram(self, function_name: str, output_path: str, 
                           style: str = 'detailed') -> str:
        """
        为特定函数创建详细的流程图
        
        参数:
            function_name (str): 函数名
            output_path (str): 输出路径
            style (str): 图表样式 ('detailed', 'simple', 'compact')
        
        返回:
            str: 生成的文件路径
        """
        if not self.analysis_result:
            raise ValueError("请先执行 analyze_function_calls() 方法")
        
        # 生成针对特定函数的调用图
        if style == 'detailed':
            upstream_depth = 2
            downstream_depth = 3
        elif style == 'simple':
            upstream_depth = 1
            downstream_depth = 2
        else:  # compact
            upstream_depth = 1
            downstream_depth = 1
        
        return self.generate_call_graph(
            output_path, 'svg', function_name, upstream_depth, downstream_depth
        )
    
    def save_analysis_result(self, output_path: str) -> None:
        """
        保存分析结果到JSON文件
        
        参数:
            output_path (str): 输出文件路径
        """
        if not self.analysis_result:
            raise ValueError("没有分析结果可以保存，请先执行 analyze_function_calls()")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_result, f, indent=2, ensure_ascii=False)
        
        print(f"分析结果已保存到: {output_file}")
    
    # ==================== 私有方法 ====================
    
    def _get_python_files(self) -> List[Path]:
        """获取所有Python文件"""
        python_files = []
        
        for root, dirs, files in os.walk(self.repo_path):
            # 移除排除的目录
            dirs[:] = [d for d in dirs if not self._should_exclude_dir(d)]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    if not self._should_exclude_file(file_path):
                        python_files.append(file_path)
        
        return python_files
    
    def _should_exclude_dir(self, dir_name: str) -> bool:
        """检查是否应该排除目录"""
        exclude_dirs = {
            '__pycache__', '.git', '.svn', '.hg', 'node_modules',
            '.venv', 'venv', 'env', '.pytest_cache', '.mypy_cache',
            'build', 'dist', '.tox'
        }
        return dir_name in exclude_dirs
    
    def _should_exclude_file(self, file_path: Path) -> bool:
        """检查是否应该排除文件"""
        # 排除测试文件和特殊文件
        if (file_path.name.startswith('test_') or 
            file_path.name.endswith('_test.py') or
            file_path.name in ('conftest.py', 'setup.py')):
            return True
        
        # 检查自定义排除模式
        for pattern in self.exclude_patterns:
            if file_path.match(pattern):
                return True
        
        return False
    
    def _generate_code2flow_graph(self, python_files: List[Path], 
                                 target_function: Optional[str], 
                                 max_depth: int) -> Dict[str, Any]:
        """使用code2flow生成调用图数据"""
        if not python_files:
            return {}
        
        try:
            # 构建code2flow命令
            cmd = ['code2flow'] + [str(f) for f in python_files[:10]]  # 限制文件数量
            
            if target_function:
                cmd.extend(['--target-function', target_function])
                cmd.extend(['--downstream-depth', str(max_depth)])
            
            # 执行code2flow
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )
            
            if result.returncode == 0:
                # 解析DOT输出
                dot_content = result.stdout
                return self._parse_dot_content(dot_content)
            else:
                print(f"code2flow执行失败: {result.stderr}")
                return {}
                
        except subprocess.TimeoutExpired:
            print("code2flow执行超时")
            return {}
        except FileNotFoundError:
            print("⚠️ code2flow工具未安装，跳过函数依赖分析")
            print("💡 安装方法: pip install code2flow")
            return {}
        except Exception as e:
            print(f"code2flow分析异常: {e}")
            return {}
    
    def _parse_dot_content(self, dot_content: str) -> Dict[str, Any]:
        """解析DOT格式的内容"""
        nodes = {}
        edges = []
        
        # 简单的DOT解析（可以使用专门的DOT解析库来改进）
        lines = dot_content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # 解析节点定义
            if '->' not in line and '[' in line and ']' in line:
                # 节点格式: "node_name" [label="Function Name"];
                match = re.match(r'"([^"]+)"\s*\[([^\]]+)\]', line)
                if match:
                    node_id, attributes = match.groups()
                    label_match = re.search(r'label="([^"]+)"', attributes)
                    if label_match:
                        nodes[node_id] = {
                            'id': node_id,
                            'label': label_match.group(1),
                            'attributes': attributes
                        }
            
            # 解析边定义
            elif '->' in line:
                # 边格式: "node1" -> "node2";
                match = re.match(r'"([^"]+)"\s*->\s*"([^"]+)"', line)
                if match:
                    source, target = match.groups()
                    edges.append({
                        'source': source,
                        'target': target
                    })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'dot_content': dot_content
        }
    
    def _extract_function_details(self, python_files: List[Path]) -> Dict[str, Any]:
        """提取所有Python文件的函数详细信息"""
        all_functions = {}
        
        for file_path in python_files:
            file_info = self.extract_function_info(str(file_path))
            if file_info and 'functions' in file_info:
                relative_path = file_path.relative_to(self.repo_path)
                all_functions[str(relative_path)] = file_info
        
        return all_functions
    
    def _extract_function_calls_from_node(self, node: ast.AST) -> List[str]:
        """从AST节点中提取函数调用"""
        calls = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        
        return calls
    
    def _generate_statistics(self, call_graph_data: Dict, 
                           function_details: Dict) -> Dict[str, Any]:
        """生成统计信息"""
        total_files = len(function_details)
        total_functions = sum(
            info.get('total_functions', 0) 
            for info in function_details.values()
        )
        
        # 统计调用关系
        total_calls = len(call_graph_data.get('edges', []))
        unique_nodes = len(call_graph_data.get('nodes', {}))
        
        return {
            'total_files': total_files,
            'total_functions': total_functions,
            'total_call_relationships': total_calls,
            'unique_function_nodes': unique_nodes,
            'average_functions_per_file': (
                total_functions / total_files if total_files > 0 else 0
            )
        }
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        import datetime
        return datetime.datetime.now().isoformat()



if __name__ == "__main__":
    #测试函数analyze_function_calls
    project_path = "/Users/wwchdemac/python_projects/debug_agent/test_input/webpage"
    analyzer = Code2FlowAnalyzer(project_path)
    result = analyzer.analyze_function_calls()
    #以优雅的方式打印Json，使其在终端的可读性强
    print(json.dumps(result, indent=4, ensure_ascii=False))
    #最后统计终端打印的字符数量，要仔细统计JSON里面的所有字符
    print(f"终端打印的字符数量: {len(json.dumps(result, indent=4, ensure_ascii=False))}")