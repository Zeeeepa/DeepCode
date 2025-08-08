#!/usr/bin/env python3
"""
快速调试工具

简化版的Tree-sitter调试助手，专为日常开发调试设计。
支持命令行参数，快速分析错误和定位相关文件。

使用方法:
    python quick_debug.py [项目路径] [--error "错误信息"] [--find symbol] [--struct]

示例:
    python quick_debug.py /path/to/project --struct                    # 显示项目结构
    python quick_debug.py /path/to/project --error "ModuleNotFoundError: No module named 'flask'"
    python quick_debug.py /path/to/project --find create_app           # 查找函数定义
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from core_modules.treesitter_analyzer import TreeSitterAnalyzer
    TREESITTER_AVAILABLE = True
except ImportError:
    TREESITTER_AVAILABLE = False


class QuickDebugTool:
    """快速调试工具"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        if not self.project_path.exists():
            raise ValueError(f"项目路径不存在: {project_path}")
        
        self.analyzer = None
        self.analysis_result = None
        
        if TREESITTER_AVAILABLE:
            self.analyzer = TreeSitterAnalyzer(project_path)
            print("🔄 正在分析项目...")
            self.analysis_result = self.analyzer.analyze_project()
            print("✅ 分析完成")
        else:
            print("⚠️ Tree-sitter不可用")
    
    def show_compact_structure(self):
        """显示紧凑的项目结构"""
        print(f"\n📁 {self.project_path.name}/")
        
        if not self.analysis_result:
            print("❌ 无法显示结构")
            return
        
        # 按目录组织文件
        dirs = {}
        for file_path, file_info in self.analysis_result.get('files', {}).items():
            dir_path = str(Path(file_path).parent)
            if dir_path == '.':
                dir_path = 'root'
            
            if dir_path not in dirs:
                dirs[dir_path] = []
            
            # 文件信息
            language = file_info.get('language', '?')
            functions = len(file_info.get('functions', []))
            classes = len(file_info.get('classes', []))
            status = "✅" if file_info.get('parsed', False) else "❌"
            
            file_name = Path(file_path).name
            stats = []
            if functions > 0:
                stats.append(f"{functions}fn")
            if classes > 0:
                stats.append(f"{classes}cls")
            
            stats_str = f"({','.join(stats)})" if stats else ""
            dirs[dir_path].append(f"  {file_name} [{language}] {stats_str} {status}")
        
        # 显示目录结构
        for dir_path in sorted(dirs.keys()):
            if dir_path != 'root':
                print(f"├── 📁 {dir_path}/")
            for file_line in sorted(dirs[dir_path]):
                print(file_line)
        
        # 语言统计
        stats = self.analysis_result.get('language_stats', {})
        if stats:
            print(f"\n📊 语言分布: " + ", ".join([f"{lang}({count})" for lang, count in stats.items() if count > 0]))
    
    def quick_error_analysis(self, error_message: str):
        """快速错误分析"""
        print(f"\n🔍 分析错误: {error_message}")
        
        if not self.analysis_result:
            print("❌ 无法分析错误")
            return
        
        import re
        relevant_files = []
        
        # 模块导入错误
        module_match = re.search(r"No module named '([^']+)'", error_message)
        if module_match:
            module_name = module_match.group(1)
            print(f"📦 查找模块: {module_name}")
            
            for file_path, file_info in self.analysis_result.get('files', {}).items():
                imports = file_info.get('imports', [])
                for imp in imports:
                    if isinstance(imp, dict):
                        if module_name in imp.get('module', '') or module_name == imp.get('name', ''):
                            relevant_files.append(file_path)
        
        # 文件路径错误
        file_match = re.search(r'File "([^"]+)"', error_message)
        if file_match:
            file_pattern = file_match.group(1)
            filename = Path(file_pattern).name
            print(f"📄 查找文件: {filename}")
            
            for file_path in self.analysis_result.get('files', {}):
                if filename in file_path:
                    relevant_files.append(file_path)
        
        # 符号错误
        name_match = re.search(r"name '([^']+)' is not defined", error_message)
        if name_match:
            symbol_name = name_match.group(1)
            print(f"🔍 查找符号: {symbol_name}")
            relevant_files.extend(self._find_symbol(symbol_name))
        
        # 显示结果
        if relevant_files:
            print("📋 相关文件:")
            for file_path in set(relevant_files):
                print(f"  📄 {file_path}")
        else:
            print("❌ 未找到相关文件")
    
    def find_symbol(self, symbol_name: str):
        """查找符号定义"""
        print(f"\n🔍 查找 '{symbol_name}' 的定义:")
        
        if not self.analysis_result:
            print("❌ 无法查找符号")
            return
        
        found = False
        for file_path, file_info in self.analysis_result.get('files', {}).items():
            # 查找函数
            for func in file_info.get('functions', []):
                if isinstance(func, dict) and func.get('name') == symbol_name:
                    line = func.get('line', '?')
                    args = func.get('args', [])
                    print(f"  🔧 函数: {file_path}:{line} - {symbol_name}({', '.join(args)})")
                    found = True
            
            # 查找类
            for cls in file_info.get('classes', []):
                if isinstance(cls, dict) and cls.get('name') == symbol_name:
                    line = cls.get('line', '?')
                    print(f"  📦 类: {file_path}:{line} - {symbol_name}")
                    found = True
        
        if not found:
            print(f"  ❌ 未找到 '{symbol_name}' 的定义")
    
    def _find_symbol(self, symbol_name: str) -> list:
        """内部符号查找方法"""
        files = []
        for file_path, file_info in self.analysis_result.get('files', {}).items():
            for func in file_info.get('functions', []):
                if isinstance(func, dict) and func.get('name') == symbol_name:
                    files.append(file_path)
            for cls in file_info.get('classes', []):
                if isinstance(cls, dict) and cls.get('name') == symbol_name:
                    files.append(file_path)
        return files
    
    def quick_deps(self):
        """快速显示依赖关系"""
        print("\n📦 主要依赖:")
        
        if not self.analysis_result:
            print("❌ 无法显示依赖")
            return
        
        all_deps = set()
        internal_deps = set()
        
        for file_path, file_info in self.analysis_result.get('files', {}).items():
            if file_info.get('language') == 'python':
                for imp in file_info.get('imports', []):
                    if isinstance(imp, dict):
                        module = imp.get('module', '')
                        if module:
                            all_deps.add(module)
                            # 检查是否是内部模块
                            if any(module in fp for fp in self.analysis_result.get('files', {})):
                                internal_deps.add(module)
        
        external_deps = all_deps - internal_deps
        
        if external_deps:
            print("  外部依赖:", ", ".join(sorted(external_deps)[:10]))
        if internal_deps:
            print("  内部模块:", ", ".join(sorted(internal_deps)))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="快速调试工具")
    parser.add_argument("project_path", nargs='?', 
                       default="/Users/wwchdemac/python_projects/debug_agent/test_input/webpage",
                       help="项目路径")
    parser.add_argument("--error", "-e", help="错误信息")
    parser.add_argument("--find", "-f", help="查找符号定义")
    parser.add_argument("--struct", "-s", action="store_true", help="显示项目结构")
    parser.add_argument("--deps", "-d", action="store_true", help="显示依赖关系")
    
    args = parser.parse_args()
    
    if not TREESITTER_AVAILABLE:
        print("⚠️ Tree-sitter不可用，将使用备用解析器")
    
    try:
        tool = QuickDebugTool(args.project_path)
        
        # 默认显示结构
        if not any([args.error, args.find, args.struct, args.deps]):
            args.struct = True
        
        if args.struct:
            tool.show_compact_structure()
        
        if args.error:
            tool.quick_error_analysis(args.error)
        
        if args.find:
            tool.find_symbol(args.find)
        
        if args.deps:
            tool.quick_deps()
    
    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 