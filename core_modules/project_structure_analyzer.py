"""
项目结构和代码定义分析器

本模块使用AST（抽象语法树）直接解析Python代码文件，提取项目结构、
函数定义、类定义等信息，并输出为JSON格式。

主要功能:
- analyze_project(): 分析整个项目结构
- extract_file_structure(): 提取单个文件的结构信息
- get_function_info(): 获取函数的详细定义信息
- get_class_info(): 获取类的详细定义信息
- extract_type_annotations(): 提取类型注解信息
"""

import os
import ast
import json
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime
import re


class ProjectStructureAnalyzer:
    """
    项目结构和代码定义分析器
    
    使用AST解析Python代码，提取详细的项目结构、函数定义、
    类定义等信息，支持类型注解分析。
    
    属性:
        project_path (Path): 项目路径
        exclude_patterns (set): 排除的文件模式
        include_test_files (bool): 是否包含测试文件
        extract_docstrings (bool): 是否提取文档字符串
        extract_decorators (bool): 是否提取装饰器信息
    """
    
    def __init__(self, project_path: str, 
                 exclude_patterns: Optional[Set[str]] = None,
                 include_test_files: bool = False,
                 extract_docstrings: bool = True,
                 extract_decorators: bool = True):
        """
        初始化项目结构分析器
        
        参数:
            project_path (str): 项目根目录路径
            exclude_patterns (set, optional): 需要排除的文件模式
            include_test_files (bool): 是否包含测试文件
            extract_docstrings (bool): 是否提取文档字符串
            extract_decorators (bool): 是否提取装饰器信息
        """
        self.project_path = Path(project_path).resolve()
        self.exclude_patterns = exclude_patterns or {
            '__pycache__', '.git', '.svn', '.hg', 'node_modules',
            '.venv', 'venv', 'env', '.pytest_cache', '.mypy_cache',
            'build', 'dist', '.tox', '.idea', '.vscode'
        }
        self.include_test_files = include_test_files
        self.extract_docstrings = extract_docstrings
        self.extract_decorators = extract_decorators
        
        # 验证项目路径
        if not self.project_path.exists():
            raise ValueError(f"项目路径不存在: {self.project_path}")
        if not self.project_path.is_dir():
            raise ValueError(f"项目路径不是目录: {self.project_path}")
    
    def analyze_project(self) -> Dict[str, Any]:
        """
        分析整个项目的结构和代码定义
        
        返回:
            dict: 包含项目完整结构信息的字典
        """
        print(f"开始分析项目结构: {self.project_path}")
        
        # 获取项目基本信息
        project_info = self._get_project_info()
        
        # 获取目录结构
        directory_structure = self._get_directory_structure()
        
        # 分析所有Python文件
        files_analysis = self._analyze_all_files()
        
        # 生成统计信息
        statistics = self._generate_statistics(files_analysis)
        
        # 构建完整的分析结果
        result = {
            'project_info': project_info,
            'directory_structure': directory_structure,
            'files_analysis': files_analysis,
            'statistics': statistics,
            'analysis_metadata': {
                'analyzer_version': '1.0.0',
                'analysis_timestamp': datetime.now().isoformat(),
                'include_test_files': self.include_test_files,
                'extract_docstrings': self.extract_docstrings,
                'extract_decorators': self.extract_decorators
            }
        }
        
        print(f"项目分析完成! 发现 {statistics['total_files']} 个Python文件")
        return result
    
    def extract_file_structure(self, file_path: str) -> Dict[str, Any]:
        """
        提取单个文件的结构信息
        
        参数:
            file_path (str): 文件路径
        
        返回:
            dict: 文件结构信息
        """
        file_path = Path(file_path).resolve()
        
        if not file_path.exists() or file_path.suffix != '.py':
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # 解析AST
            tree = ast.parse(source_code, filename=str(file_path))
            
            # 计算相对路径
            try:
                relative_path = str(file_path.relative_to(self.project_path))
            except ValueError:
                # 如果文件不在项目路径内，使用文件名
                relative_path = file_path.name
            
            # 提取文件信息
            file_info = {
                'file_path': relative_path,
                'absolute_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'line_count': len(source_code.splitlines()),
                'imports': self._extract_imports(tree),
                'functions': {},
                'classes': {},
                'global_variables': {},
                'constants': {},
                'docstring': ast.get_docstring(tree)
            }
            
            # 直接遍历模块的顶级节点
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    func_info = self._get_function_info(node, source_code)
                    file_info['functions'][node.name] = func_info
                
                elif isinstance(node, ast.AsyncFunctionDef):
                    func_info = self._get_function_info(node, source_code)
                    func_info['is_async'] = True
                    file_info['functions'][node.name] = func_info
                
                elif isinstance(node, ast.ClassDef):
                    class_info = self._get_class_info(node, source_code)
                    file_info['classes'][node.name] = class_info
                
                elif isinstance(node, ast.Assign):
                    # 提取全局变量和常量
                    var_info = self._extract_variable_assignment(node, source_code)
                    if var_info:
                        if var_info['is_constant']:
                            file_info['constants'].update(var_info['variables'])
                        else:
                            file_info['global_variables'].update(var_info['variables'])
            
            return file_info
            
        except Exception as e:
            print(f"警告: 分析文件失败 {file_path}: {e}")
            return {
                'file_path': str(file_path.relative_to(self.project_path)),
                'error': str(e),
                'analysis_failed': True
            }
    
    def save_analysis_result(self, result: Dict[str, Any], output_path: str) -> None:
        """
        保存分析结果到JSON文件
        
        参数:
            result (dict): 分析结果
            output_path (str): 输出文件路径
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"分析结果已保存到: {output_file}")
    
    # ==================== 私有方法 ====================
    
    def _get_project_info(self) -> Dict[str, Any]:
        """获取项目基本信息"""
        return {
            'project_name': self.project_path.name,
            'project_path': str(self.project_path),
            'project_size': sum(f.stat().st_size for f in self.project_path.rglob('*.py') 
                               if not self._should_exclude_file(f))
        }
    
    def _get_directory_structure(self) -> Dict[str, Any]:
        """获取目录结构"""
        def build_tree(path: Path, max_depth: int = 10, current_depth: int = 0) -> Dict[str, Any]:
            if current_depth >= max_depth:
                return {'type': 'directory', 'truncated': True}
            
            if path.is_file():
                return {
                    'type': 'file',
                    'size': path.stat().st_size,
                    'extension': path.suffix
                }
            
            children = {}
            try:
                for item in sorted(path.iterdir()):
                    if not self._should_exclude_path(item):
                        children[item.name] = build_tree(item, max_depth, current_depth + 1)
            except PermissionError:
                return {'type': 'directory', 'error': 'Permission denied'}
            
            return {
                'type': 'directory',
                'children': children
            }
        
        return build_tree(self.project_path)
    
    def _analyze_all_files(self) -> Dict[str, Dict[str, Any]]:
        """分析所有Python文件"""
        files_analysis = {}
        
        for py_file in self.project_path.rglob('*.py'):
            if not self._should_exclude_file(py_file):
                relative_path = str(py_file.relative_to(self.project_path))
                files_analysis[relative_path] = self.extract_file_structure(py_file)
        
        return files_analysis
    
    def _get_function_info(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], 
                          source_code: str) -> Dict[str, Any]:
        """获取函数详细信息"""
        func_info = {
            'name': node.name,
            'line_start': node.lineno,
            'line_end': getattr(node, 'end_lineno', node.lineno),
            'col_offset': node.col_offset,
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'arguments': self._extract_function_arguments(node),
            'return_type': self._extract_type_annotation(node.returns),
            'decorators': [],
            'docstring': ast.get_docstring(node) if self.extract_docstrings else None,
            'complexity_metrics': self._calculate_complexity(node),
            'calls_made': self._extract_function_calls(node),
            'variables_used': self._extract_variables_used(node)
        }
        
        # 提取装饰器信息
        if self.extract_decorators:
            func_info['decorators'] = [
                self._get_decorator_name(decorator) 
                for decorator in node.decorator_list
            ]
        
        # 提取函数体的源代码
        try:
            lines = source_code.splitlines()
            if node.lineno <= len(lines) and getattr(node, 'end_lineno', node.lineno) <= len(lines):
                func_body_lines = lines[node.lineno-1:getattr(node, 'end_lineno', node.lineno)]
                func_info['source_code'] = '\n'.join(func_body_lines)
        except:
            pass
        
        return func_info
    
    def _get_class_info(self, node: ast.ClassDef, source_code: str) -> Dict[str, Any]:
        """获取类详细信息"""
        class_info = {
            'name': node.name,
            'line_start': node.lineno,
            'line_end': getattr(node, 'end_lineno', node.lineno),
            'col_offset': node.col_offset,
            'base_classes': [self._get_base_class_name(base) for base in node.bases],
            'decorators': [],
            'docstring': ast.get_docstring(node) if self.extract_docstrings else None,
            'methods': {},
            'class_variables': {},
            'properties': {}
        }
        
        # 提取装饰器
        if self.extract_decorators:
            class_info['decorators'] = [
                self._get_decorator_name(decorator) 
                for decorator in node.decorator_list
            ]
        
        # 分析类成员
        for child_node in node.body:
            if isinstance(child_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_info = self._get_function_info(child_node, source_code)
                method_info['method_type'] = self._get_method_type(child_node)
                class_info['methods'][child_node.name] = method_info
            
            elif isinstance(child_node, ast.Assign):
                # 类变量
                var_info = self._extract_variable_assignment(child_node, source_code)
                if var_info:
                    class_info['class_variables'].update(var_info['variables'])
        
        return class_info
    
    def _extract_function_arguments(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[Dict[str, Any]]:
        """提取函数参数信息"""
        args_info = []
        args = node.args
        
        # 处理普通参数
        for i, arg in enumerate(args.args):
            arg_info = {
                'name': arg.arg,
                'type_annotation': self._extract_type_annotation(arg.annotation),
                'default_value': None,
                'kind': 'positional'
            }
            
            # 检查是否有默认值
            defaults_offset = len(args.args) - len(args.defaults)
            if i >= defaults_offset:
                default_idx = i - defaults_offset
                arg_info['default_value'] = self._get_default_value(args.defaults[default_idx])
            
            args_info.append(arg_info)
        
        # 处理 *args
        if args.vararg:
            args_info.append({
                'name': args.vararg.arg,
                'type_annotation': self._extract_type_annotation(args.vararg.annotation),
                'kind': 'var_positional'
            })
        
        # 处理仅限关键字参数
        for i, arg in enumerate(args.kwonlyargs):
            arg_info = {
                'name': arg.arg,
                'type_annotation': self._extract_type_annotation(arg.annotation),
                'default_value': None,
                'kind': 'keyword_only'
            }
            
            if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
                arg_info['default_value'] = self._get_default_value(args.kw_defaults[i])
            
            args_info.append(arg_info)
        
        # 处理 **kwargs
        if args.kwarg:
            args_info.append({
                'name': args.kwarg.arg,
                'type_annotation': self._extract_type_annotation(args.kwarg.annotation),
                'kind': 'var_keyword'
            })
        
        return args_info
    
    def _extract_type_annotation(self, annotation: Optional[ast.AST]) -> Optional[str]:
        """提取类型注解信息"""
        if annotation is None:
            return None
        
        try:
            return ast.unparse(annotation)
        except:
            # 如果unparse失败，尝试简单的类型提取
            if isinstance(annotation, ast.Name):
                return annotation.id
            elif isinstance(annotation, ast.Constant):
                return repr(annotation.value)
            else:
                return str(type(annotation).__name__)
    
    def _extract_imports(self, tree: ast.AST) -> Dict[str, List[str]]:
        """提取导入信息"""
        imports = {
            'standard_imports': [],
            'from_imports': [],
            'relative_imports': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_info = alias.name
                    if alias.asname:
                        import_info += f" as {alias.asname}"
                    imports['standard_imports'].append(import_info)
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                level = node.level
                
                for alias in node.names:
                    import_info = f"from {module} import {alias.name}"
                    if alias.asname:
                        import_info += f" as {alias.asname}"
                    
                    if level > 0:
                        imports['relative_imports'].append(import_info)
                    else:
                        imports['from_imports'].append(import_info)
        
        return imports
    
    def _extract_function_calls(self, node: ast.AST) -> List[str]:
        """提取函数中的调用"""
        calls = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = self._get_call_name(child.func)
                if call_name:
                    calls.append(call_name)
        
        return list(set(calls))  # 去重
    
    def _extract_variables_used(self, node: ast.AST) -> List[str]:
        """提取函数中使用的变量"""
        variables = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                variables.add(child.id)
        
        return list(variables)
    
    def _calculate_complexity(self, node: ast.AST) -> Dict[str, int]:
        """计算代码复杂度指标"""
        complexity = {
            'cyclomatic_complexity': 1,  # 基础复杂度
            'lines_of_code': 0,
            'nested_depth': 0
        }
        
        # 计算圈复杂度
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, 
                                ast.Try, ast.With, ast.AsyncWith)):
                complexity['cyclomatic_complexity'] += 1
            elif isinstance(child, ast.BoolOp):
                complexity['cyclomatic_complexity'] += len(child.values) - 1
        
        # 计算代码行数
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            complexity['lines_of_code'] = (getattr(node, 'end_lineno', node.lineno) - 
                                         node.lineno + 1)
        
        return complexity
    
    def _extract_variable_assignment(self, node: ast.Assign, source_code: str) -> Optional[Dict[str, Any]]:
        """提取变量赋值信息"""
        variables = {}
        
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                var_info = {
                    'name': var_name,
                    'line_number': node.lineno,
                    'type_annotation': None,
                    'value': self._get_assignment_value(node.value),
                    'is_constant': var_name.isupper()  # 按照约定，全大写为常量
                }
                variables[var_name] = var_info
        
        if variables:
            return {
                'variables': variables,
                'is_constant': all(var['is_constant'] for var in variables.values())
            }
        
        return None
    
    def _get_assignment_value(self, node: ast.AST) -> Any:
        """获取赋值的值"""
        try:
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Name):
                return f"<variable: {node.id}>"
            elif isinstance(node, ast.List):
                return f"<list with {len(node.elts)} elements>"
            elif isinstance(node, ast.Dict):
                return f"<dict with {len(node.keys)} keys>"
            else:
                return f"<{type(node).__name__}>"
        except:
            return "<unknown>"
    
    def _get_call_name(self, node: ast.AST) -> Optional[str]:
        """获取函数调用名称"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value_name = self._get_call_name(node.value)
            return f"{value_name}.{node.attr}" if value_name else node.attr
        else:
            return None
    
    def _get_decorator_name(self, node: ast.AST) -> str:
        """获取装饰器名称"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_call_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return f"{self._get_decorator_name(node.func)}(...)"
        else:
            return str(type(node).__name__)
    
    def _get_base_class_name(self, node: ast.AST) -> str:
        """获取基类名称"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_call_name(node.value)}.{node.attr}"
        else:
            return str(type(node).__name__)
    
    def _get_default_value(self, node: ast.AST) -> Any:
        """获取参数默认值"""
        try:
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Name):
                return f"<{node.id}>"
            else:
                return f"<{type(node).__name__}>"
        except:
            return "<unknown>"
    
    def _get_method_type(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
        """判断方法类型"""
        # 检查装饰器
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id == 'staticmethod':
                    return 'static'
                elif decorator.id == 'classmethod':
                    return 'class'
                elif decorator.id == 'property':
                    return 'property'
        
        # 检查是否是特殊方法
        if node.name.startswith('__') and node.name.endswith('__'):
            return 'magic'
        
        # 检查是否是私有方法
        if node.name.startswith('_'):
            return 'private'
        
        return 'instance'
    

    
    def _should_exclude_path(self, path: Path) -> bool:
        """检查路径是否应该被排除"""
        return path.name in self.exclude_patterns
    
    def _should_exclude_file(self, file_path: Path) -> bool:
        """检查文件是否应该被排除"""
        # 检查是否在排除目录中
        for part in file_path.parts:
            if part in self.exclude_patterns:
                return True
        
        # 检查是否是测试文件
        if not self.include_test_files:
            if (file_path.name.startswith('test_') or 
                file_path.name.endswith('_test.py') or
                'test' in file_path.parts):
                return True
        
        return False
    
    def _generate_statistics(self, files_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """生成统计信息"""
        stats = {
            'total_files': 0,
            'total_functions': 0,
            'total_classes': 0,
            'total_methods': 0,
            'total_lines_of_code': 0,
            'files_with_errors': 0,
            'average_functions_per_file': 0,
            'average_methods_per_class': 0,
            'type_annotation_coverage': 0
        }
        
        annotated_functions = 0
        total_function_params = 0
        annotated_params = 0
        
        for file_info in files_analysis.values():
            if file_info.get('analysis_failed'):
                stats['files_with_errors'] += 1
                continue
            
            stats['total_files'] += 1
            stats['total_lines_of_code'] += file_info.get('line_count', 0)
            
            # 统计函数
            functions = file_info.get('functions', {})
            stats['total_functions'] += len(functions)
            
            for func_info in functions.values():
                if func_info.get('return_type'):
                    annotated_functions += 1
                
                for arg in func_info.get('arguments', []):
                    total_function_params += 1
                    if arg.get('type_annotation'):
                        annotated_params += 1
            
            # 统计类和方法
            classes = file_info.get('classes', {})
            stats['total_classes'] += len(classes)
            
            for class_info in classes.values():
                methods = class_info.get('methods', {})
                stats['total_methods'] += len(methods)
                
                for method_info in methods.values():
                    if method_info.get('return_type'):
                        annotated_functions += 1
                    
                    for arg in method_info.get('arguments', []):
                        total_function_params += 1
                        if arg.get('type_annotation'):
                            annotated_params += 1
        
        # 计算平均值
        if stats['total_files'] > 0:
            stats['average_functions_per_file'] = stats['total_functions'] / stats['total_files']
        
        if stats['total_classes'] > 0:
            stats['average_methods_per_class'] = stats['total_methods'] / stats['total_classes']
        
        # 计算类型注解覆盖率
        total_annotatable = (stats['total_functions'] + stats['total_methods'] + 
                           total_function_params)
        if total_annotatable > 0:
            stats['type_annotation_coverage'] = (
                (annotated_functions + annotated_params) / total_annotatable * 100
            )
        
        return stats


# 命令行使用示例
def main():
    """
    主函数，演示ProjectStructureAnalyzer的基本用法
    """
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python project_structure_analyzer.py <project_path> [output_file]")
        print("示例: python project_structure_analyzer.py ./test_input/snake_game ./output/structure.json")
        return
    
    project_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "./project_structure_analysis.json"
    
    try:
        # 创建分析器实例
        analyzer = ProjectStructureAnalyzer(
            project_path,
            include_test_files=True,  # 包含测试文件
            extract_docstrings=True,
            extract_decorators=True
        )
        
        # 执行分析
        result = analyzer.analyze_project()
        
        # 保存结果
        analyzer.save_analysis_result(result, output_file)
        
        # 打印统计信息
        stats = result['statistics']
        print(f"\n=== 项目结构分析结果 ===")
        print(f"项目名称: {result['project_info']['project_name']}")
        print(f"总文件数: {stats['total_files']}")
        print(f"总函数数: {stats['total_functions']}")
        print(f"总类数: {stats['total_classes']}")
        print(f"总方法数: {stats['total_methods']}")
        print(f"总代码行数: {stats['total_lines_of_code']}")
        print(f"平均每文件函数数: {stats['average_functions_per_file']:.2f}")
        print(f"平均每类方法数: {stats['average_methods_per_class']:.2f}")
        print(f"类型注解覆盖率: {stats['type_annotation_coverage']:.1f}%")
        
        if stats['files_with_errors'] > 0:
            print(f"⚠️  {stats['files_with_errors']} 个文件分析失败")
        
        print(f"\n✅ 分析完成! 结果保存在: {output_file}")
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 