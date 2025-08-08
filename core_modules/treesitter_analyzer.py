"""
Tree-sitter代码分析器

使用tree-sitter增量解析器来分析多种编程语言的代码结构。
支持生成详细的语法树（AST）和提取代码元素。

主要功能:
- analyze_project(): 分析整个项目并生成语法树
- parse_file(): 解析单个文件
- extract_functions(): 提取函数定义
- extract_classes(): 提取类定义
- extract_imports(): 提取导入语句
- get_language_stats(): 获取语言统计信息
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
import time

try:
    import tree_sitter
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    # 创建占位符类以避免NameError
    class Language:
        pass
    class Parser:
        pass


class TreeSitterAnalyzer:
    """
    基于tree-sitter的代码分析器
    
    使用tree-sitter解析器生成多种编程语言的语法树，
    并提取详细的代码结构信息。
    
    属性:
        project_path (Path): 项目路径
        supported_languages (dict): 支持的语言映射
        parsers (dict): 语言解析器缓存
        exclude_patterns (set): 排除的文件/目录模式
    """
    
    def __init__(self, project_path: str, exclude_patterns: Optional[Set[str]] = None):
        """
        初始化Tree-sitter分析器
        
        参数:
            project_path (str): 项目根目录路径
            exclude_patterns (set, optional): 需要排除的文件模式
        """
        if not TREE_SITTER_AVAILABLE:
            print("⚠️ tree-sitter未安装，将使用备用解析器（基于AST和正则表达式）")
        
        self.project_path = Path(project_path).resolve()
        self.exclude_patterns = exclude_patterns or {
            '__pycache__', '.git', '.svn', '.hg', 'node_modules',
            '.venv', 'venv', 'env', '.pytest_cache', '.mypy_cache',
            'build', 'dist', '.tox', '.idea', '.vscode', 'target',
            'out', 'bin', 'obj', '.vs', 'debug_output'
        }
        
        # 支持的语言和文件扩展名映射
        self.language_extensions = {
            'python': ['.py', '.pyw'],
            'javascript': ['.js', '.jsx'],
            'typescript': ['.ts', '.tsx'],
            'java': ['.java'],
            'cpp': ['.cpp', '.cxx', '.cc', '.c++'],
            'c': ['.c', '.h'],
            'csharp': ['.cs'],
            'go': ['.go'],
            'rust': ['.rs'],
            'php': ['.php'],
            'ruby': ['.rb'],
            'swift': ['.swift'],
            'kotlin': ['.kt', '.kts'],
            'scala': ['.scala'],
            'html': ['.html', '.htm'],
            'css': ['.css', '.scss', '.sass'],
            'json': ['.json'],
            'yaml': ['.yaml', '.yml'],
            'xml': ['.xml'],
            'sql': ['.sql'],
            'bash': ['.sh', '.bash']
        }
        
        # 解析器缓存
        self.parsers = {}
        self.languages = {}
        
        if not self.project_path.exists():
            raise ValueError(f"项目路径不存在: {self.project_path}")
        if not self.project_path.is_dir():
            raise ValueError(f"项目路径不是目录: {self.project_path}")
    
    def _get_language_for_file(self, file_path: Path) -> Optional[str]:
        """
        根据文件扩展名确定语言类型
        
        参数:
            file_path (Path): 文件路径
        
        返回:
            str: 语言类型，如果不支持则返回None
        """
        suffix = file_path.suffix.lower()
        for language, extensions in self.language_extensions.items():
            if suffix in extensions:
                return language
        return None
    
    def _get_parser(self, language: str) -> Optional[Parser]:
        """
        获取指定语言的解析器
        
        参数:
            language (str): 语言类型
        
        返回:
            Parser: tree-sitter解析器，如果不支持则返回None
        """
        if language in self.parsers:
            return self.parsers[language]
        
        try:
            # 尝试加载预编译的语言库
            # 注意：这需要先安装对应的tree-sitter语言包
            language_lib_name = f"tree-sitter-{language}"
            
            # 简化版本：只支持Python作为示例
            # 实际使用中需要安装对应语言的tree-sitter库
            if language == 'python':
                try:
                    # 如果有预编译的Python语言库
                    python_language = Language('build/my-languages.so', 'python')
                    parser = Parser()
                    parser.set_language(python_language)
                    self.parsers[language] = parser
                    self.languages[language] = python_language
                    return parser
                except:
                    # 备用方案：使用AST模拟tree-sitter行为
                    return None
            
            # 其他语言的支持需要单独配置
            return None
            
        except Exception as e:
            print(f"⚠️ 无法加载{language}语言解析器: {str(e)}")
            return None
    
    def _should_exclude_file(self, file_path: Path) -> bool:
        """
        判断是否应该排除文件
        
        参数:
            file_path (Path): 文件路径
        
        返回:
            bool: 是否排除
        """
        try:
            relative_path = file_path.relative_to(self.project_path)
            path_parts = relative_path.parts
            
            # 检查路径中是否包含排除模式
            for part in path_parts:
                if part in self.exclude_patterns:
                    return True
            
            # 检查文件名是否匹配排除模式
            if file_path.name.startswith('.') and file_path.name not in ['.gitignore', '.env']:
                return True
            
            # 检查文件大小（排除过大的文件）
            if file_path.stat().st_size > 10 * 1024 * 1024:  # 10MB
                return True
            
            return False
            
        except (ValueError, OSError):
            return True
    
    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        解析单个文件并提取语法树信息
        
        参数:
            file_path (Path): 文件路径
        
        返回:
            dict: 文件解析结果
        """
        language = self._get_language_for_file(file_path)
        if not language:
            return {
                "language": "unknown",
                "error": f"不支持的文件类型: {file_path.suffix}",
                "parsed": False
            }
        
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            parser = self._get_parser(language)
            
            if parser:
                # 使用tree-sitter解析
                tree = parser.parse(bytes(content, "utf8"))
                
                result = {
                    "language": language,
                    "file_size": len(content),
                    "line_count": len(content.splitlines()),
                    "parsed": True,
                    "syntax_tree": self._extract_tree_structure(tree.root_node, content),
                    "functions": self._extract_functions_from_tree(tree.root_node, content),
                    "classes": self._extract_classes_from_tree(tree.root_node, content),
                    "imports": self._extract_imports_from_tree(tree.root_node, content),
                    "errors": []
                }
                
                # 检查语法错误
                if tree.root_node.has_error:
                    result["errors"].append("语法错误")
                
                return result
            else:
                # 备用解析方案（针对Python使用AST）
                if language == 'python':
                    return self._parse_python_with_ast(content, file_path)
                else:
                    return self._parse_generic_file(content, file_path, language)
        
        except Exception as e:
            return {
                "language": language,
                "error": f"解析失败: {str(e)}",
                "parsed": False
            }
    
    def _parse_python_with_ast(self, content: str, file_path: Path) -> Dict[str, Any]:
        """
        使用Python内置AST解析Python文件（备用方案）
        
        参数:
            content (str): 文件内容
            file_path (Path): 文件路径
        
        返回:
            dict: 解析结果
        """
        try:
            import ast
            
            tree = ast.parse(content)
            
            functions = []
            classes = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                        "decorators": [ast.unparse(dec) for dec in node.decorator_list] if hasattr(ast, 'unparse') else [],
                        "docstring": ast.get_docstring(node)
                    })
                
                elif isinstance(node, ast.ClassDef):
                    class_methods = []
                    for class_node in node.body:
                        if isinstance(class_node, ast.FunctionDef):
                            class_methods.append({
                                "name": class_node.name,
                                "line": class_node.lineno,
                                "args": [arg.arg for arg in class_node.args.args]
                            })
                    
                    classes.append({
                        "name": node.name,
                        "line": node.lineno,
                        "bases": [ast.unparse(base) for base in node.bases] if hasattr(ast, 'unparse') else [],
                        "methods": class_methods,
                        "docstring": ast.get_docstring(node)
                    })
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append({
                                "type": "import",
                                "module": alias.name,
                                "alias": alias.asname,
                                "line": node.lineno
                            })
                    else:  # ImportFrom
                        for alias in node.names:
                            imports.append({
                                "type": "from_import",
                                "module": node.module,
                                "name": alias.name,
                                "alias": alias.asname,
                                "line": node.lineno
                            })
            
            return {
                "language": "python",
                "file_size": len(content),
                "line_count": len(content.splitlines()),
                "parsed": True,
                "parser_type": "ast",
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "errors": []
            }
        
        except SyntaxError as e:
            return {
                "language": "python",
                "error": f"Python语法错误: {str(e)}",
                "parsed": False
            }
        except Exception as e:
            return {
                "language": "python", 
                "error": f"AST解析失败: {str(e)}",
                "parsed": False
            }
    
    def _parse_generic_file(self, content: str, file_path: Path, language: str) -> Dict[str, Any]:
        """
        通用文件解析（简单文本分析）
        
        参数:
            content (str): 文件内容
            file_path (Path): 文件路径
            language (str): 语言类型
        
        返回:
            dict: 解析结果
        """
        lines = content.splitlines()
        
        # 简单的模式匹配
        functions = []
        classes = []
        imports = []
        
        # 根据语言类型进行简单解析
        if language == 'javascript' or language == 'typescript':
            # 简单的JS/TS函数检测
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if line.startswith('function ') or ' function ' in line:
                    func_name = self._extract_js_function_name(line)
                    if func_name:
                        functions.append({"name": func_name, "line": i, "type": "function"})
                
                if line.startswith('class '):
                    class_name = self._extract_js_class_name(line)
                    if class_name:
                        classes.append({"name": class_name, "line": i, "type": "class"})
                
                if line.startswith('import ') or line.startswith('const ') and 'require(' in line:
                    imports.append({"line": i, "statement": line})
        
        elif language == 'java':
            # 简单的Java解析
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if 'public class ' in line or 'class ' in line:
                    class_name = self._extract_java_class_name(line)
                    if class_name:
                        classes.append({"name": class_name, "line": i, "type": "class"})
                
                if ('public ' in line or 'private ' in line) and '(' in line and ')' in line:
                    func_name = self._extract_java_method_name(line)
                    if func_name:
                        functions.append({"name": func_name, "line": i, "type": "method"})
                
                if line.startswith('import '):
                    imports.append({"line": i, "statement": line})
        
        return {
            "language": language,
            "file_size": len(content),
            "line_count": len(lines),
            "parsed": True,
            "parser_type": "regex",
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "errors": []
        }
    
    def _extract_js_function_name(self, line: str) -> Optional[str]:
        """从JavaScript行中提取函数名"""
        import re
        patterns = [
            r'function\s+(\w+)',
            r'(\w+)\s*=\s*function',
            r'(\w+)\s*:\s*function',
            r'(\w+)\s*=\s*\(',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1)
        return None
    
    def _extract_js_class_name(self, line: str) -> Optional[str]:
        """从JavaScript行中提取类名"""
        import re
        match = re.search(r'class\s+(\w+)', line)
        return match.group(1) if match else None
    
    def _extract_java_class_name(self, line: str) -> Optional[str]:
        """从Java行中提取类名"""
        import re
        match = re.search(r'class\s+(\w+)', line)
        return match.group(1) if match else None
    
    def _extract_java_method_name(self, line: str) -> Optional[str]:
        """从Java行中提取方法名"""
        import re
        match = re.search(r'\s+(\w+)\s*\([^)]*\)\s*{?', line)
        return match.group(1) if match else None
    
    def _extract_tree_structure(self, node, content: str) -> Dict[str, Any]:
        """从tree-sitter节点提取树结构"""
        return {
            "type": node.type,
            "start_point": node.start_point,
            "end_point": node.end_point,
            "children_count": len(node.children),
            "has_error": node.has_error
        }
    
    def _extract_functions_from_tree(self, node, content: str) -> List[Dict[str, Any]]:
        """从tree-sitter树中提取函数定义"""
        # 这里需要根据具体语言的tree-sitter语法来实现
        # 作为示例，返回空列表
        return []
    
    def _extract_classes_from_tree(self, node, content: str) -> List[Dict[str, Any]]:
        """从tree-sitter树中提取类定义"""
        # 这里需要根据具体语言的tree-sitter语法来实现
        return []
    
    def _extract_imports_from_tree(self, node, content: str) -> List[Dict[str, Any]]:
        """从tree-sitter树中提取导入语句"""
        return []
    
    def analyze_project(self) -> Dict[str, Any]:
        """
        分析整个项目并生成详细报告
        
        返回:
            dict: 完整的项目分析结果
        """
        start_time = time.time()
        print(f"🌳 开始Tree-sitter分析项目: {self.project_path.name}")
        
        result = {
            "analyzer": "TreeSitterAnalyzer",
            "project_name": self.project_path.name,
            "project_path": str(self.project_path),
            "analysis_time": None,
            "supported_languages": list(self.language_extensions.keys()),
            "language_stats": defaultdict(int),
            "files": {},
            "summary": {
                "total_files": 0,
                "parsed_files": 0,
                "failed_files": 0,
                "total_functions": 0,
                "total_classes": 0,
                "total_imports": 0
            }
        }
        
        # 收集所有支持的文件
        all_files = []
        for language, extensions in self.language_extensions.items():
            for ext in extensions:
                all_files.extend(self.project_path.rglob(f'*{ext}'))
        
        # 分析每个文件
        for file_path in all_files:
            if self._should_exclude_file(file_path):
                continue
            
            relative_path = str(file_path.relative_to(self.project_path))
            language = self._get_language_for_file(file_path)
            
            if language:
                result["language_stats"][language] += 1
                result["summary"]["total_files"] += 1
                
                # 解析文件
                file_result = self.parse_file(file_path)
                result["files"][relative_path] = file_result
                
                if file_result["parsed"]:
                    result["summary"]["parsed_files"] += 1
                    result["summary"]["total_functions"] += len(file_result.get("functions", []))
                    result["summary"]["total_classes"] += len(file_result.get("classes", []))
                    result["summary"]["total_imports"] += len(file_result.get("imports", []))
                else:
                    result["summary"]["failed_files"] += 1
        
        # 计算分析时间
        analysis_time = time.time() - start_time
        result["analysis_time"] = f"{analysis_time:.2f}秒"
        
        # 转换defaultdict为普通dict
        result["language_stats"] = dict(result["language_stats"])
        
        print(f"✅ Tree-sitter分析完成: {result['summary']['parsed_files']}/{result['summary']['total_files']} 文件成功解析")
        
        return result
    
    def get_language_stats(self) -> Dict[str, int]:
        """
        获取项目中各种语言的文件统计
        
        返回:
            dict: 语言统计信息
        """
        stats = defaultdict(int)
        
        for language, extensions in self.language_extensions.items():
            for ext in extensions:
                files = list(self.project_path.rglob(f'*{ext}'))
                valid_files = [f for f in files if not self._should_exclude_file(f)]
                stats[language] += len(valid_files)
        
        return dict(stats)
    
    def analyze_project_structure(self, repo_path: str) -> Dict[str, Any]:
        """
        分析项目结构的简化接口（与其他分析器兼容）
        
        参数:
            repo_path (str): 项目路径
        
        返回:
            dict: 项目结构分析结果
        """
        # 更新项目路径
        old_path = self.project_path
        self.project_path = Path(repo_path).resolve()
        
        try:
            return self.analyze_project()
        finally:
            # 恢复原路径
            self.project_path = old_path 