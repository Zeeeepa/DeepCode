"""
基于pydeps的仓库结构和依赖关系分析器

本模块使用pydeps工具来分析Python项目的模块依赖关系。
pydeps是一个强大的Python模块依赖分析工具，可以生成可视化的依赖图。

主要功能:
- analyze_repo(): 使用pydeps分析整个仓库
- extract_directory_structure(): 提取目录结构
- generate_dependency_graph(): 生成依赖图
- get_module_dependencies(): 获取模块依赖关系
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys


class PyDepsAnalyzer:
    """
    基于pydeps的代码仓库分析器
    
    使用pydeps工具分析Python项目的模块依赖关系，
    并提供额外的目录结构分析功能。
    
    属性:
        repo_path (Path): 待分析的仓库路径
        exclude_dirs (set): 需要排除的目录名称
        analysis_result (dict): 分析结果缓存
    """
    
    def __init__(self, repo_path: str, exclude_dirs: Optional[set] = None):
        """
        初始化pydeps分析器
        
        参数:
            repo_path (str): 待分析的仓库路径
            exclude_dirs (set, optional): 需要排除的目录
        """
        self.repo_path = Path(repo_path).resolve()
        self.exclude_dirs = exclude_dirs or {
            '__pycache__', '.git', '.svn', '.hg', 'node_modules', 
            '.venv', 'venv', 'env', '.pytest_cache', '.mypy_cache',
            'build', 'dist', '.tox', '.coverage', 'htmlcov'
        }
        self.analysis_result = {}
        
        # 检查路径有效性
        if not self.repo_path.exists():
            raise ValueError(f"仓库路径不存在: {self.repo_path}")
        if not self.repo_path.is_dir():
            raise ValueError(f"仓库路径不是目录: {self.repo_path}")
    
    def analyze_repo(self) -> Dict[str, Any]:
        """
        分析整个仓库的结构和依赖关系
        
        返回:
            dict: 包含分析结果的字典，包括：
                - directory_structure: 目录结构
                - module_dependencies: 使用pydeps获取的模块依赖
                - statistics: 统计信息
                - pydeps_available: pydeps是否可用
        """
        print(f"开始分析仓库: {self.repo_path}")
        
        # 1. 检查pydeps是否可用
        pydeps_available = self._check_pydeps_available()
        
        # 2. 提取目录结构
        print("正在分析目录结构...")
        directory_structure = self.extract_directory_structure()
        
        # 3. 获取Python文件列表
        python_files = self._get_python_files()
        
        # 4. 使用pydeps分析依赖关系（如果可用）
        module_dependencies = {}
        pydeps_info = {}
        
        if pydeps_available and python_files:
            print("正在使用pydeps分析模块依赖关系...")
            try:
                module_dependencies, pydeps_info = self._analyze_with_pydeps()
            except Exception as e:
                print(f"pydeps分析失败: {e}")
                module_dependencies = self._fallback_dependency_analysis()
        else:
            print("pydeps不可用，使用简单的依赖分析...")
            module_dependencies = self._fallback_dependency_analysis()
        
        # 5. 生成统计信息
        statistics = self._generate_statistics(
            directory_structure, module_dependencies, python_files
        )
        
        # 6. 整合分析结果
        self.analysis_result = {
            'repo_path': str(self.repo_path),
            'directory_structure': directory_structure,
            'module_dependencies': module_dependencies,
            'pydeps_info': pydeps_info,
            'statistics': statistics,
            'pydeps_available': pydeps_available,
            'analysis_timestamp': self._get_timestamp()
        }
        
        print("仓库分析完成!")
        return self.analysis_result
    
    def extract_directory_structure(self) -> Dict[str, Any]:
        """
        提取仓库的目录结构
        
        返回:
            dict: 目录结构字典
        """
        def _scan_directory(path: Path, relative_to: Path) -> List[Dict[str, Any]]:
            """递归扫描目录结构"""
            items = []
            
            try:
                for item in sorted(path.iterdir()):
                    # 跳过排除的目录
                    if item.name in self.exclude_dirs:
                        continue
                    
                    relative_path = item.relative_to(relative_to)
                    
                    if item.is_dir():
                        dir_info = {
                            'type': 'directory',
                            'name': item.name,
                            'path': str(relative_path),
                            'children': _scan_directory(item, relative_to)
                        }
                        items.append(dir_info)
                    
                    elif item.is_file() and item.suffix == '.py':
                        file_info = {
                            'type': 'file',
                            'name': item.name,
                            'path': str(relative_path),
                            'size': item.stat().st_size,
                            'extension': item.suffix
                        }
                        items.append(file_info)
                        
            except PermissionError:
                pass
            
            return items
        
        return {
            'type': 'directory',
            'name': self.repo_path.name,
            'path': '.',
            'children': _scan_directory(self.repo_path, self.repo_path)
        }
    
    def generate_dependency_graph(self, output_path: str, format: str = 'svg') -> str:
        """
        使用pydeps生成依赖图
        
        参数:
            output_path (str): 输出文件路径
            format (str): 输出格式 ('svg', 'png', 'dot')
        
        返回:
            str: 生成的文件路径
        """
        if not self._check_pydeps_available():
            raise Exception("pydeps不可用，无法生成依赖图")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 构建pydeps命令
        cmd = [
            sys.executable, '-m', 'pydeps',
            str(self.repo_path),
            '-o', str(output_file)
        ]
        
        # 添加格式参数
        if format == 'svg':
            cmd.extend(['-T', 'svg'])
        elif format == 'png':
            cmd.extend(['-T', 'png'])
        elif format == 'dot':
            cmd.extend(['--dot-output', str(output_file)])
        
        # 排除外部依赖，只关注项目内部模块
        cmd.extend(['--max-bacon', '2'])
        cmd.extend(['--noshow'])  # 不显示图形界面
        
        try:
            print(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                raise Exception(f"pydeps执行失败: {result.stderr}")
            
            print(f"依赖图已生成: {output_file}")
            return str(output_file)
            
        except subprocess.TimeoutExpired:
            raise Exception("pydeps执行超时")
        except Exception as e:
            raise Exception(f"生成依赖图失败: {e}")
    
    def get_module_dependencies(self) -> Dict[str, List[str]]:
        """
        获取模块依赖关系
        
        返回:
            dict: 模块依赖字典
        """
        if not self.analysis_result:
            raise ValueError("请先执行 analyze_repo() 方法")
        
        return self.analysis_result.get('module_dependencies', {})
    
    def save_analysis_result(self, output_path: str) -> None:
        """
        保存分析结果到JSON文件
        
        参数:
            output_path (str): 输出文件路径
        """
        if not self.analysis_result:
            raise ValueError("没有分析结果可以保存，请先执行 analyze_repo()")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_result, f, indent=2, ensure_ascii=False)
        
        print(f"分析结果已保存到: {output_file}")
    
    # ==================== 私有方法 ====================
    
    def _check_pydeps_available(self) -> bool:
        """检查pydeps是否可用"""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pydeps', '--help'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _get_python_files(self) -> List[Path]:
        """获取所有Python文件"""
        python_files = []
        for root, dirs, files in os.walk(self.repo_path):
            # 移除排除的目录
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    python_files.append(Path(root) / file)
        
        return python_files
    
    def _analyze_with_pydeps(self) -> tuple:
        """使用pydeps分析依赖关系"""
        dependencies = {}
        pydeps_info = {}
        
        try:
            # 创建临时文件保存pydeps输出
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # 运行pydeps获取依赖信息（使用--show-deps获取文本格式的依赖信息）
            cmd = [
                sys.executable, '-m', 'pydeps',
                str(self.repo_path),
                '--show-deps',
                '--deps-output', tmp_path,
                '--max-bacon', '3',
                '--noshow'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # 读取pydeps输出的文本文件
                try:
                    with open(tmp_path, 'r', encoding='utf-8') as f:
                        pydeps_text = f.read()
                    
                    # 解析pydeps文本数据
                    dependencies = self._parse_pydeps_text_output(pydeps_text)
                    pydeps_info = {
                        'success': True,
                        'command': ' '.join(cmd),
                        'stdout': result.stdout,
                        'deps_count': len(dependencies)
                    }
                except Exception as e:
                    pydeps_info = {
                        'success': False,
                        'error': f"解析pydeps输出失败: {e}",
                        'stdout': result.stdout,
                        'stderr': result.stderr
                    }
            else:
                pydeps_info = {
                    'success': False,
                    'error': f"pydeps执行失败: {result.stderr}",
                    'command': ' '.join(cmd)
                }
            
            # 清理临时文件
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
                
        except Exception as e:
            pydeps_info = {
                'success': False,
                'error': f"pydeps分析异常: {e}"
            }
        
        return dependencies, pydeps_info
    
    def _parse_pydeps_text_output(self, pydeps_text: str) -> Dict[str, List[str]]:
        """解析pydeps文本输出数据"""
        dependencies = {}
        
        # pydeps --show-deps的输出格式通常是：
        # module_name -> dependency1, dependency2, ...
        lines = pydeps_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # 解析依赖关系行
            if ' -> ' in line:
                parts = line.split(' -> ', 1)
                if len(parts) == 2:
                    module = parts[0].strip()
                    deps_str = parts[1].strip()
                    
                    # 解析依赖列表
                    if deps_str:
                        deps = [dep.strip() for dep in deps_str.split(',') if dep.strip()]
                        dependencies[module] = deps
                    else:
                        dependencies[module] = []
            elif ':' in line:
                # 另一种可能的格式：module: dependency1, dependency2
                parts = line.split(':', 1)
                if len(parts) == 2:
                    module = parts[0].strip()
                    deps_str = parts[1].strip()
                    
                    if deps_str:
                        deps = [dep.strip() for dep in deps_str.split(',') if dep.strip()]
                        dependencies[module] = deps
                    else:
                        dependencies[module] = []
        
        return dependencies
    
    def _fallback_dependency_analysis(self) -> Dict[str, List[str]]:
        """备用的简单依赖分析"""
        import ast
        dependencies = {}
        
        python_files = self._get_python_files()
        
        for file_path in python_files:
            try:
                module_name = self._get_module_name(file_path)
                imports = self._extract_imports(file_path)
                dependencies[module_name] = imports
            except Exception:
                continue
        
        return dependencies
    
    def _extract_imports(self, file_path: Path) -> List[str]:
        """提取文件的import语句"""
        imports = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except Exception:
            pass
        
        return imports
    
    def _get_module_name(self, file_path: Path) -> str:
        """根据文件路径生成模块名"""
        relative_path = file_path.relative_to(self.repo_path)
        module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
        return '.'.join(module_parts)
    
    def _generate_statistics(self, directory_structure: Dict, 
                           module_deps: Dict, python_files: List) -> Dict[str, Any]:
        """生成统计信息"""
        def count_files(structure):
            count = 0
            if structure['type'] == 'file':
                return 1
            elif structure['type'] == 'directory':
                for child in structure.get('children', []):
                    count += count_files(child)
            return count
        
        total_modules = len(module_deps)
        total_files = count_files(directory_structure)
        total_python_files = len(python_files)
        
        return {
            'total_files': total_files,
            'total_python_files': total_python_files,
            'total_modules': total_modules,
            'average_dependencies_per_module': (
                sum(len(deps) for deps in module_deps.values()) / total_modules
                if total_modules > 0 else 0
            )
        }
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        import datetime
        return datetime.datetime.now().isoformat()


# 命令行使用示例
def main():
    """
    主函数，演示PyDepsAnalyzer的基本用法
    """
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python pydeps_analyzer.py <repo_path> [output_path]")
        print("示例: python pydeps_analyzer.py ./test_input/snake_game ./output")
        return
    
    repo_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./analysis_output"
    
    try:
        # 创建分析器实例
        analyzer = PyDepsAnalyzer(repo_path)
        
        # 执行分析
        result = analyzer.analyze_repo()
        
        # 创建输出目录
        Path(output_dir).mkdir(exist_ok=True)
        
        # 保存分析结果
        result_file = f"{output_dir}/analysis_result.json"
        analyzer.save_analysis_result(result_file)
        
        # 生成依赖图（如果pydeps可用）
        if result['pydeps_available']:
            try:
                svg_file = f"{output_dir}/dependency_graph.svg"
                analyzer.generate_dependency_graph(svg_file, 'svg')
                print(f"依赖图已生成: {svg_file}")
            except Exception as e:
                print(f"生成依赖图失败: {e}")
        else:
            print("pydeps不可用，跳过依赖图生成")
            print("安装pydeps: pip install pydeps")
        
        # 打印统计信息
        stats = result['statistics']
        print(f"\n=== 分析结果统计 ===")
        print(f"总文件数: {stats['total_files']}")
        print(f"Python文件数: {stats['total_python_files']}")
        print(f"模块数: {stats['total_modules']}")
        print(f"平均每模块依赖数: {stats['average_dependencies_per_module']:.2f}")
        
        print(f"\n分析完成! 结果保存在: {output_dir}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 