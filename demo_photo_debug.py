#!/usr/bin/env python3
"""
复杂项目调试系统演示

测试调试系统在面对包含50种错误的复杂项目时的表现。

项目特点：
- 多文件结构 (8个Python文件)
- 50种不同类型的编程错误
- 涵盖导入、类型、逻辑、文件操作、异常处理等错误
- 测试调试系统的综合能力
"""

import os
import json
from pathlib import Path
from agents.debug_system import DebugSystem


def show_project_overview():
    """显示项目概览"""
    print("📊 复杂图书管理系统 - 测试项目概览")
    print("=" * 60)
    
    project_info = {
        "项目名称": "图书管理系统",
        "错误数量": "50种不同类型",
        "文件数量": "8个Python文件 + 1个数据文件",
        "项目结构": [
            "📁 models/ - 数据模型 (Book, Library, User)",
            "📁 utils/ - 工具类 (FileHandler, Validator)",
            "📄 main.py - 主程序",
            "📄 data/library_data.json - 示例数据"
        ],
        "错误分类": [
            "🔧 代码结构错误 (1-10)",
            "📊 数据处理错误 (11-20)", 
            "🔍 逻辑错误 (21-30)",
            "🛡️ 验证和安全错误 (31-40)",
            "🏃 运行时错误 (41-50)"
        ]
    }
    
    for key, value in project_info.items():
        if isinstance(value, list):
            print(f"{key}:")
            for item in value:
                print(f"  {item}")
        else:
            print(f"{key}: {value}")
        print()


def check_project_exists():
    """检查复杂项目是否存在"""
    project_path = Path("complex_library_system")
    
    if not project_path.exists():
        print("❌ 复杂测试项目不存在！")
        print("请先运行以下命令创建项目:")
        print("python create_complex_test.py")
        return False
    
    # 检查关键文件
    required_files = [
        "main.py",
        "models/__init__.py",
        "models/book.py",
        "models/library.py",
        "models/user.py",
        "utils/__init__.py",
        "utils/file_handler.py",
        "utils/validator.py",
        "data/library_data.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not (project_path / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 缺少必要文件: {missing_files}")
        return False
    
    print("✅ 复杂测试项目检查通过")
    return True


def demonstrate_initial_error():
    """演示初始错误"""
    print("\n🔍 让我们先看看这个复杂项目的初始错误...")
    print("-" * 50)
    
    import subprocess
    
    # 运行项目查看错误
    try:
        result = subprocess.run(
            ["python", "main.py"],
            cwd="complex_library_system",
            capture_output=True,
            text=True,
            timeout=10
        )
        
        print("📤 程序输出:")
        if result.stdout:
            print(result.stdout)
        
        print("❌ 错误信息:")
        if result.stderr:
            print(result.stderr)
        
        print(f"🔢 返回码: {result.returncode}")
        
    except subprocess.TimeoutExpired:
        print("⏰ 程序执行超时")
    except Exception as e:
        print(f"💥 执行异常: {e}")


def run_complex_debug():
    """运行复杂项目的调试"""
    print("\n🚀 启动智能调试系统...")
    print("=" * 60)
    
    # 配置参数
    repo_path = "/Users/wwchdemac/python_projects/debug_agent/test_input/photo_cutout_tool"
    main_file = "backend/main.py"
    expected_behavior = """
希望能够正确运行，不要报错
"""
    
    # 初始化调试系统
    debug_system = DebugSystem()
    
    print(f"📂 目标项目: {repo_path}")
    print(f"📄 主程序文件: {main_file}")
    print(f"🎯 最大尝试次数: {debug_system.max_attempts}")
    print()
    
    # 开始调试
    try:
        result = debug_system.debug_program(
            repo_path=repo_path,
            main_file=main_file,
            expected_behavior=expected_behavior
        )
        
        return result
        
    except Exception as e:
        print(f"💥 调试系统异常: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_debug_results(result):
    """分析调试结果"""
    if not result:
        print("\n❌ 调试失败，无结果可分析")
        return
    
    print("\n📊 调试结果分析")
    print("=" * 60)
    
    # 基本统计
    print(f"✅ 调试成功: {'是' if result['success'] else '否'}")
    print(f"🔄 尝试次数: {result['attempts']}")
    print(f"📋 最终错误分类: {result.get('final_error_category', '未知')}")
    
    # 详细分析
    debug_log = result.get('debug_log', [])
    
    if debug_log:
        print(f"\n📝 调试过程详情:")
        print("-" * 40)
        
        for i, attempt in enumerate(debug_log, 1):
            print(f"\n第{i}次尝试:")
            
            # 判断结果
            judge_result = attempt.get('judge_result', {})
            if judge_result:
                is_correct = judge_result.get('is_correct', False)
                error_category = judge_result.get('error_category', '未知')
                print(f"  🤖 判断: {'✅ 正确' if is_correct else '❌ 错误'} ({error_category})")
                if not is_correct:
                    reason = judge_result.get('reason', 'N/A')[:100] + "..."
                    print(f"  💭 原因: {reason}")
            
            # 修复结果
            coder_result = attempt.get('coder_result', {})
            if coder_result:
                success = coder_result.get('success', False)
                modification_summary = coder_result.get('modification_summary', 'N/A')
                print(f"  🛠️ 修复: {'✅ 成功' if success else '❌ 失败'}")
                print(f"  📝 概述: {modification_summary}")
    
    # 输出文件信息
    print(f"\n📄 生成的文件:")
    output_files = [
        ("代码库索引", result.get('repo_index_path')),
        ("修改历史", result.get('modification_history_path')),
        ("调试报告", result.get('modification_history_path', '').replace('modification_history.json', 'debug_report.json'))
    ]
    
    for name, path in output_files:
        if path and os.path.exists(path):
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name}: 未生成")


def show_final_program_output(result):
    """显示最终程序输出"""
    if not result:
        return
    
    final_output = result.get('final_output', '')
    if not final_output:
        return
    
    print(f"\n📤 最终程序输出:")
    print("=" * 60)
    print(final_output)
    print("=" * 60)


def main():
    """主函数"""
    print("🧪 复杂项目调试系统演示")
    print("🎯 测试目标：50种错误的图书管理系统")
    print()
    
    # 1. 显示项目概览
    # show_project_overview()
    
    # # 2. 检查项目存在性
    # if not check_project_exists():
    #     return
    
    # # 3. 检查配置
    # print("\n⚙️ 检查配置...")
    # from agents.config import AgentConfig
    # config = AgentConfig()
    
    # api_key = config.get_config("api_key")
    # base_url = config.get_config("base_url") 
    
    # if not api_key or not base_url:
    #     print("❌ 请先配置环境变量:")
    #     print("   AGENT_API_KEY=your_openrouter_api_key")
    #     print("   AGENT_BASE_URL=https://openrouter.ai/api/v1")
    #     print("   AGENT_MODEL=anthropic/claude-sonnet-4")
    #     return
    
    # print("✅ 配置检查通过")
    
    # # 4. 演示初始错误
    # demonstrate_initial_error()
    
    # # 5. 等待用户确认
    # print("\n" + "="*60)
    # print("🤔 这个复杂项目包含50种错误，调试过程可能需要较长时间...")
    # print("💡 预计会进行多轮修复，展示调试系统的持续改进能力")
    
    # user_input = input("\n是否继续进行调试？(y/N): ").strip().lower()
    # if user_input not in ['y', 'yes']:
    #     print("👋 调试演示已取消")
    #     return
    
    # 6. 运行调试
    result = run_complex_debug()
    
    # 7. 分析结果
    analyze_debug_results(result)
    
    # 8. 显示最终输出
    show_final_program_output(result)
    
    # 9. 总结
    if result and result.get('success'):
        print("\n🎉 恭喜！调试系统成功修复了复杂项目！")
        print("🏆 这证明了智能调试系统在面对复杂多文件项目时的强大能力！")
    else:
        print("\n🤔 调试过程遇到了挑战...")
        print("💪 这正是复杂项目测试的价值所在！")
    
    print("\n✨ 复杂项目调试演示完成！")


if __name__ == "__main__":
    main() 