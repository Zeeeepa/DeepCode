"""
Tree-sitter语言库设置脚本

用于编译和安装tree-sitter支持的各种编程语言的解析器。
这个脚本会下载并编译常用的语言库。

主要功能:
- download_language_grammars(): 下载语言语法文件
- compile_languages(): 编译语言库
- setup_all_languages(): 一键安装所有支持的语言
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import tempfile


class TreeSitterSetup:
    """
    Tree-sitter语言库设置工具
    
    用于自动下载、编译和安装tree-sitter语言解析器。
    """
    
    def __init__(self, build_dir: str = "build"):
        """
        初始化设置工具
        
        参数:
            build_dir (str): 构建目录
        """
        self.build_dir = Path(build_dir)
        self.build_dir.mkdir(exist_ok=True)
        
        # 支持的语言和对应的GitHub仓库
        self.language_repos = {
            'python': 'https://github.com/tree-sitter/tree-sitter-python',
            'javascript': 'https://github.com/tree-sitter/tree-sitter-javascript',
            'typescript': 'https://github.com/tree-sitter/tree-sitter-typescript',
            'java': 'https://github.com/tree-sitter/tree-sitter-java',
            'cpp': 'https://github.com/tree-sitter/tree-sitter-cpp',
            'c': 'https://github.com/tree-sitter/tree-sitter-c',
            'csharp': 'https://github.com/tree-sitter/tree-sitter-c-sharp',
            'go': 'https://github.com/tree-sitter/tree-sitter-go',
            'rust': 'https://github.com/tree-sitter/tree-sitter-rust',
            'php': 'https://github.com/tree-sitter/tree-sitter-php',
            'ruby': 'https://github.com/tree-sitter/tree-sitter-ruby',
            'swift': 'https://github.com/tree-sitter/tree-sitter-swift',
            'kotlin': 'https://github.com/fwcd/tree-sitter-kotlin',
            'scala': 'https://github.com/tree-sitter/tree-sitter-scala',
            'html': 'https://github.com/tree-sitter/tree-sitter-html',
            'css': 'https://github.com/tree-sitter/tree-sitter-css',
            'json': 'https://github.com/tree-sitter/tree-sitter-json',
            'yaml': 'https://github.com/ikatyang/tree-sitter-yaml',
            'xml': 'https://github.com/oberon00/tree-sitter-xml',
            'sql': 'https://github.com/derekstride/tree-sitter-sql',
            'bash': 'https://github.com/tree-sitter/tree-sitter-bash'
        }
    
    def check_dependencies(self) -> bool:
        """
        检查必要的依赖工具
        
        返回:
            bool: 依赖是否满足
        """
        required_tools = ['git', 'gcc', 'python3']
        
        print("🔍 检查依赖工具...")
        for tool in required_tools:
            if not shutil.which(tool):
                print(f"❌ 缺少必要工具: {tool}")
                return False
            else:
                print(f"✅ 找到工具: {tool}")
        
        # 检查tree-sitter Python包
        try:
            import tree_sitter
            print("✅ tree-sitter Python包已安装")
        except ImportError:
            print("❌ 缺少tree-sitter Python包，请运行: pip install tree-sitter")
            return False
        
        return True
    
    def download_language_grammar(self, language: str) -> Optional[Path]:
        """
        下载指定语言的语法文件
        
        参数:
            language (str): 语言名称
        
        返回:
            Path: 下载的语法目录路径，失败返回None
        """
        if language not in self.language_repos:
            print(f"❌ 不支持的语言: {language}")
            return None
        
        repo_url = self.language_repos[language]
        grammar_dir = self.build_dir / f"tree-sitter-{language}"
        
        print(f"📥 下载 {language} 语法...")
        
        try:
            # 如果目录已存在，先删除
            if grammar_dir.exists():
                shutil.rmtree(grammar_dir)
            
            # 克隆仓库
            subprocess.run([
                'git', 'clone', '--depth', '1', repo_url, str(grammar_dir)
            ], check=True, capture_output=True)
            
            print(f"✅ {language} 语法下载完成")
            return grammar_dir
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 下载 {language} 语法失败: {e}")
            return None
    
    def compile_language(self, language: str, grammar_dir: Path) -> bool:
        """
        编译指定语言的解析器
        
        参数:
            language (str): 语言名称
            grammar_dir (Path): 语法目录
        
        返回:
            bool: 编译是否成功
        """
        print(f"🔨 编译 {language} 解析器...")
        
        try:
            from tree_sitter import Language
            
            # 查找源文件
            src_dir = grammar_dir / 'src'
            if not src_dir.exists():
                print(f"❌ 找不到源文件目录: {src_dir}")
                return False
            
            # TypeScript需要特殊处理（包含多个子项目）
            if language == 'typescript':
                # TypeScript仓库包含typescript和tsx两个子项目
                typescript_dir = grammar_dir / 'typescript' / 'src'
                tsx_dir = grammar_dir / 'tsx' / 'src'
                
                if typescript_dir.exists():
                    Language.build_library(
                        str(self.build_dir / 'my-languages.so'),
                        [str(typescript_dir)]
                    )
                    print(f"✅ {language} (TypeScript) 编译完成")
                    
                if tsx_dir.exists():
                    Language.build_library(
                        str(self.build_dir / 'my-languages.so'),
                        [str(tsx_dir)]
                    )
                    print(f"✅ {language} (TSX) 编译完成")
            else:
                # 标准编译流程
                Language.build_library(
                    str(self.build_dir / 'my-languages.so'),
                    [str(src_dir)]
                )
                print(f"✅ {language} 编译完成")
            
            return True
            
        except Exception as e:
            print(f"❌ 编译 {language} 失败: {e}")
            return False
    
    def compile_all_languages(self, languages: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        编译多个语言的解析器
        
        参数:
            languages (List[str], optional): 要编译的语言列表，None表示全部
        
        返回:
            Dict[str, bool]: 各语言的编译结果
        """
        if languages is None:
            languages = list(self.language_repos.keys())
        
        results = {}
        grammar_dirs = {}
        
        print(f"🚀 开始批量编译 {len(languages)} 种语言...")
        
        # 先下载所有语法
        for language in languages:
            grammar_dir = self.download_language_grammar(language)
            if grammar_dir:
                grammar_dirs[language] = grammar_dir
            else:
                results[language] = False
        
        # 批量编译
        if grammar_dirs:
            try:
                from tree_sitter import Language
                
                # 收集所有源目录
                all_sources = []
                for language, grammar_dir in grammar_dirs.items():
                    src_dir = grammar_dir / 'src'
                    if src_dir.exists():
                        all_sources.append(str(src_dir))
                
                # 一次性编译所有语言
                if all_sources:
                    print("🔨 批量编译所有语言...")
                    Language.build_library(
                        str(self.build_dir / 'my-languages.so'),
                        all_sources
                    )
                    
                    # 标记所有语言为编译成功
                    for language in grammar_dirs.keys():
                        results[language] = True
                    
                    print("✅ 批量编译完成")
                else:
                    print("❌ 没有找到有效的源文件")
                    
            except Exception as e:
                print(f"❌ 批量编译失败: {e}")
                # 回退到单个编译
                for language, grammar_dir in grammar_dirs.items():
                    results[language] = self.compile_language(language, grammar_dir)
        
        return results
    
    def setup_essential_languages(self) -> Dict[str, bool]:
        """
        安装基本的语言支持（Python, JavaScript, Java等）
        
        返回:
            Dict[str, bool]: 安装结果
        """
        essential_languages = ['python', 'javascript', 'java', 'cpp', 'c']
        return self.compile_all_languages(essential_languages)
    
    def verify_installation(self, language: str) -> bool:
        """
        验证语言库是否正确安装
        
        参数:
            language (str): 语言名称
        
        返回:
            bool: 验证是否通过
        """
        try:
            from tree_sitter import Language, Parser
            
            library_path = self.build_dir / 'my-languages.so'
            if not library_path.exists():
                return False
            
            # 尝试加载语言
            lang = Language(str(library_path), language)
            parser = Parser()
            parser.set_language(lang)
            
            # 尝试解析一个简单的代码片段
            test_code = self._get_test_code(language)
            tree = parser.parse(bytes(test_code, "utf8"))
            
            return not tree.root_node.has_error
            
        except Exception:
            return False
    
    def _get_test_code(self, language: str) -> str:
        """
        获取用于测试的简单代码片段
        
        参数:
            language (str): 语言名称
        
        返回:
            str: 测试代码
        """
        test_codes = {
            'python': 'def hello(): pass',
            'javascript': 'function hello() {}',
            'java': 'class Test { public void hello() {} }',
            'cpp': 'int main() { return 0; }',
            'c': 'int main() { return 0; }'
        }
        
        return test_codes.get(language, '# test')
    
    def generate_usage_example(self) -> str:
        """
        生成使用示例代码
        
        返回:
            str: 示例代码
        """
        return """
# Tree-sitter使用示例

from tree_sitter import Language, Parser

# 加载编译好的语言库
PY_LANGUAGE = Language('build/my-languages.so', 'python')
JS_LANGUAGE = Language('build/my-languages.so', 'javascript')

# 创建解析器
py_parser = Parser()
py_parser.set_language(PY_LANGUAGE)

js_parser = Parser()
js_parser.set_language(JS_LANGUAGE)

# 解析Python代码
python_code = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''

tree = py_parser.parse(bytes(python_code, "utf8"))
print("Python AST:")
print(tree.root_node.sexp())

# 解析JavaScript代码
js_code = '''
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n-1) + fibonacci(n-2);
}
'''

tree = js_parser.parse(bytes(js_code, "utf8"))
print("JavaScript AST:")
print(tree.root_node.sexp())
"""


def main():
    """主函数，演示如何使用设置工具"""
    setup = TreeSitterSetup()
    
    if not setup.check_dependencies():
        print("\n❌ 依赖检查失败，请安装必要的工具")
        return
    
    print("\n🚀 开始安装Tree-sitter语言支持...")
    
    # 安装基本语言
    results = setup.setup_essential_languages()
    
    print("\n📊 安装结果:")
    for language, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {language}")
    
    # 验证安装
    print("\n🔍 验证安装...")
    for language in results.keys():
        if setup.verify_installation(language):
            print(f"✅ {language} 验证通过")
        else:
            print(f"❌ {language} 验证失败")
    
    # 生成使用示例
    example_file = Path("treesitter_example.py")
    with open(example_file, 'w', encoding='utf-8') as f:
        f.write(setup.generate_usage_example())
    
    print(f"\n📝 使用示例已保存到: {example_file}")
    print("\n🎉 Tree-sitter设置完成！")


if __name__ == "__main__":
    main() 