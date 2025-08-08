#!/usr/bin/env python3
"""
依赖检查脚本

检查Debug Agent项目所需的所有依赖是否正确安装。
"""

import sys
import importlib
from typing import Dict, List, Tuple


def check_import(module_name: str, optional: bool = False) -> Tuple[bool, str]:
    """
    检查单个模块是否可以导入
    
    参数:
        module_name (str): 模块名
        optional (bool): 是否为可选依赖
    
    返回:
        tuple: (是否成功, 状态信息)
    """
    try:
        importlib.import_module(module_name)
        return True, f"✅ {module_name}"
    except ImportError as e:
        status = "⚠️" if optional else "❌"
        return False, f"{status} {module_name} - {str(e)}"


def check_command_availability(command: str) -> Tuple[bool, str]:
    """
    检查命令行工具是否可用
    
    参数:
        command (str): 命令名
    
    返回:
        tuple: (是否成功, 状态信息)
    """
    import subprocess
    try:
        result = subprocess.run([command, '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return True, f"✅ {command} - 可用"
        else:
            return False, f"❌ {command} - 命令执行失败"
    except FileNotFoundError:
        return False, f"❌ {command} - 命令未找到"
    except subprocess.TimeoutExpired:
        return False, f"❌ {command} - 命令超时"
    except Exception as e:
        return False, f"❌ {command} - {str(e)}"


def main():
    """主检查函数"""
    print("🔍 Debug Agent 依赖检查")
    print("=" * 50)
    
    # 核心依赖
    print("\n📦 核心Python依赖:")
    core_deps = [
        ("requests", False),
        ("json", False),  # 内置模块
        ("os", False),    # 内置模块
        ("pathlib", False),  # 内置模块
        ("typing", False),   # 内置模块
    ]
    
    core_success = 0
    for dep, optional in core_deps:
        success, message = check_import(dep, optional)
        print(f"  {message}")
        if success:
            core_success += 1
    
    # 分析工具依赖
    print("\n🔧 代码分析工具:")
    analysis_deps = [
        ("pydeps", False),
        ("ast", False),  # 内置模块
    ]
    
    analysis_success = 0
    for dep, optional in analysis_deps:
        success, message = check_import(dep, optional)
        print(f"  {message}")
        if success:
            analysis_success += 1
    
    # 可选增强依赖
    print("\n🎨 可选增强功能:")
    optional_deps = [
        ("graphviz", True),
        ("PIL", True),  # Pillow
        ("dotenv", True),
        ("colorama", True),
        ("tqdm", True),
        ("jsonschema", True),
    ]
    
    optional_success = 0
    for dep, optional in optional_deps:
        success, message = check_import(dep, optional)
        print(f"  {message}")
        if success:
            optional_success += 1
    
    # 命令行工具
    print("\n🛠️ 外部命令行工具:")
    command_tools = [
        "code2flow",
        "dot",  # Graphviz
    ]
    
    command_success = 0
    for tool in command_tools:
        success, message = check_command_availability(tool)
        print(f"  {message}")
        if success:
            command_success += 1
    
    # 开发依赖（仅检查，不强制）
    print("\n🧪 开发和测试工具 (可选):")
    dev_deps = [
        ("pytest", True),
        ("black", True),
        ("flake8", True),
        ("isort", True),
    ]
    
    dev_success = 0
    for dep, optional in dev_deps:
        success, message = check_import(dep, optional)
        print(f"  {message}")
        if success:
            dev_success += 1
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 检查总结:")
    print(f"  核心依赖: {core_success}/{len(core_deps)} ({'✅' if core_success == len(core_deps) else '❌'})")
    print(f"  分析工具: {analysis_success}/{len(analysis_deps)} ({'✅' if analysis_success == len(analysis_deps) else '❌'})")
    print(f"  外部工具: {command_success}/{len(command_tools)} ({'✅' if command_success >= 1 else '⚠️'})")
    print(f"  可选功能: {optional_success}/{len(optional_deps)} ({'✅' if optional_success >= 3 else '⚠️'})")
    print(f"  开发工具: {dev_success}/{len(dev_deps)} ({'✅' if dev_success >= 2 else '⚠️'})")
    
    # 给出建议
    print("\n💡 建议:")
    
    if core_success < len(core_deps):
        print("  ❌ 请安装核心依赖: pip install -r requirements-core.txt")
    
    if analysis_success < len(analysis_deps):
        print("  ❌ 请安装分析工具: pip install pydeps")
    
    if command_success == 0:
        print("  ⚠️ 建议安装code2flow: pip install code2flow")
        print("  ⚠️ 建议安装Graphviz系统包:")
        print("     - macOS: brew install graphviz")
        print("     - Ubuntu: sudo apt-get install graphviz")
        print("     - Windows: https://graphviz.org/download/")
    
    if optional_success < 3:
        print("  ⚠️ 建议安装完整依赖以获得最佳体验: pip install -r requirements.txt")
    
    # 系统兼容性检查
    print(f"\n🐍 Python版本: {sys.version}")
    if sys.version_info < (3, 8):
        print("  ❌ 警告: Python版本过低，建议使用Python 3.8+")
    elif sys.version_info >= (3, 8):
        print("  ✅ Python版本符合要求")
    
    # 最终状态
    print("\n" + "=" * 50)
    
    essential_ok = core_success == len(core_deps) and analysis_success == len(analysis_deps)
    
    if essential_ok:
        if command_success >= 1 and optional_success >= 3:
            print("🎉 所有依赖检查通过！系统已准备就绪。")
            return 0
        else:
            print("✅ 核心功能可用，部分增强功能缺失。")
            return 0
    else:
        print("❌ 缺少关键依赖，请先安装必需组件。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 