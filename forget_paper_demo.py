#!/usr/bin/env python3
"""
Will Model Forget论文复现分析Demo

使用实际的Will Model Forget项目和论文指南进行完整的复现分析。

主要功能:
- 基于真实Will Model Forget项目代码的分析
- 使用paper_test_2_reproduction_guide.md作为指南
- 支持补充信息文档功能，可以提供额外的实现指导
- 目标是完美复现论文算法和训练流程
- 保证实验的完整性和可复现性

新增功能:
- additional_guides: 支持多个markdown文档作为补充信息
- 补充信息会在分析和编码阶段被AI使用
- 可以包含配置说明、实现技巧、问题解决方案等
- 帮助提高复现的准确性和完整性
"""

import os
import sys
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.paper_reproduction_system import PaperReproductionSystem


def create_forget_target_metrics():
    """
    创建Will Model Forget论文复现的目标指标
    
    返回一个简单的字符串描述，告诉系统我们想要完美复现论文。
    """
    return "完美复现Will Model Forget论文的所有算法、实验和结果，确保与论文描述完全一致"


def run_forget_paper_reproduction():
    """运行Will Model Forget论文的完整复现流程（包含分析和执行）"""
    
    print("🚀 Will Model Forget论文完整复现系统")
    print("=" * 60)
    
    # 配置参数
    config = {
        "repo_path": "/Users/wwchdemac/python_projects/debug_agent/test_input/will model forget",
        "paper_guide": "/Users/wwchdemac/python_projects/debug_agent/test_papers/paper_test_2_reproduction_guide.md", 
        "additional_guides": [
            # 可以在这里添加补充信息文档路径
            # 例如："/path/to/config_guide.md",
            #      "/path/to/implementation_tips.md",
            #      "/path/to/troubleshooting.md"
            "/Users/wwchdemac/python_projects/debug_agent/test_papers/paper_test_2_addendum.md"
        ],
        "target_metrics": create_forget_target_metrics(),
        "max_iterations": 5  # L0, L1, L2, L3, L4
    }
    
    # 验证路径存在性
    if not os.path.exists(config["repo_path"]):
        print(f"❌ 代码库路径不存在: {config['repo_path']}")
        return False
        
    if not os.path.exists(config["paper_guide"]):
        print(f"❌ 论文指南文件不存在: {config['paper_guide']}")
        return False
    
    # 验证补充信息文档路径
    valid_additional_guides = []
    if config["additional_guides"]:
        print("📄 验证补充信息文档...")
        for guide_path in config["additional_guides"]:
            if os.path.exists(guide_path):
                valid_additional_guides.append(guide_path)
                print(f"  ✅ {os.path.basename(guide_path)}")
            else:
                print(f"  ⚠️  文件不存在，将跳过: {guide_path}")
        
        if len(valid_additional_guides) < len(config["additional_guides"]):
            print(f"📋 补充信息文档: {len(valid_additional_guides)}/{len(config['additional_guides'])} 个有效")
    
    print(f"📁 代码库路径: {config['repo_path']}")
    print(f"📄 论文指南: {config['paper_guide']}")
    if valid_additional_guides:
        print(f"📚 补充信息文档: {len(valid_additional_guides)} 个")
        for i, guide in enumerate(valid_additional_guides, 1):
            print(f"   {i}. {os.path.basename(guide)}")
    else:
        print(f"📚 补充信息文档: 未提供")
    print(f"🎯 目标指标: {config['target_metrics']}")
    print(f"🔄 计划迭代: {config['max_iterations']} 次（L0-L4层级）")
    
    try:
        # 初始化完整的复现系统（包含分析器和执行器）
        reproduction_system = PaperReproductionSystem()
        
        print("\n🔬 开始完整的论文复现流程...")
        print("包含: 分析 → 生成任务 → 执行任务 → 修改代码")
        if valid_additional_guides:
            print("补充信息: 将结合补充文档进行更精确的分析和实现")
        print("-" * 60)
        
        # 执行完整的复现流程
        result = reproduction_system.reproduce_paper(
            repo_path=config["repo_path"],
            paper_guide=config["paper_guide"],
            additional_guides=valid_additional_guides if valid_additional_guides else None,
            target_metrics=config["target_metrics"],
            max_iterations=config["max_iterations"]
        )
        
        print("\n" + "=" * 60)
        print("📋 Will Model Forget论文复现结果总结")
        print("=" * 60)
        
        if result.get("success", False):
            print("✅ 论文复现成功完成！")
            print(f"📊 总迭代次数: {result.get('total_iterations', 0)}")
            print(f"🎯 最终达到层级: {result.get('final_level_achieved', 'Unknown')}")
            
            # 显示补充信息使用情况
            additional_info = result.get("additional_info", {})
            if additional_info and additional_info.get("processed_count", 0) > 0:
                print(f"📚 使用补充信息: {additional_info['processed_count']} 个文档, {additional_info['final_char_count']} 字符")
            
            improvements = result.get("improvements_made", [])
            if improvements:
                print(f"🔧 改进数量: {len(improvements)} 项")
                print("主要改进:")
                for i, improvement in enumerate(improvements[:5], 1):
                    print(f"  {i}. {improvement}")
                if len(improvements) > 5:
                    print(f"  ... 还有 {len(improvements)-5} 项改进")
        else:
            print("❌ 论文复现未完全成功")
            
        # 显示输出目录
        if result.get("output_dir"):
            print(f"💾 详细结果保存在: {result['output_dir']}")
            
        return result.get("success", False)
        
    except Exception as e:
        print(f"❌ 复现过程中出现异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def show_target_metrics_summary():
    """显示目标指标摘要"""
    metrics = create_forget_target_metrics()
    
    print("\n🎯 Will Model Forget论文复现目标:")
    print("-" * 40)
    print(f"📌 {metrics}")
    print("\n💡 提示: 您可以在 create_forget_target_metrics() 函数中编辑目标描述")


def run_forget_with_additional_guides_example():
    """
    展示如何使用补充信息功能的示例
    
    这个函数展示了如何配置和使用补充信息文档来增强论文复现的效果。
    用户可以根据自己的需求修改additional_guides列表。
    """
    print("🚀 Will Model Forget论文复现系统 - 补充信息功能示例")
    print("=" * 60)
    
    # 示例配置 - 包含补充信息文档
    config = {
        "repo_path": "/Users/wwchdemac/python_projects/debug_agent/test_input/will model forget",
        "paper_guide": "/Users/wwchdemac/python_projects/debug_agent/test_papers/paper_test_2_reproduction_guide.md", 
        "additional_guides": [
            # 在这里添加您的补充信息文档路径
            # 例如：
            # "/path/to/your/config_guide.md",           # 配置相关的补充信息
            # "/path/to/your/implementation_details.md", # 实现细节补充
            # "/path/to/your/troubleshooting_tips.md",   # 问题解决技巧
            # "/path/to/your/performance_tuning.md",     # 性能调优建议
            "/Users/wwchdemac/python_projects/debug_agent/test_papers/paper_test_2_addendum.md"
        ],
        "target_metrics": create_forget_target_metrics(),
        "max_iterations": 5
    }
    
    print("📋 补充信息功能说明:")
    print("   • 支持多个 markdown 文档作为补充信息")
    print("   • 补充信息会传递给分析器和编码器")
    print("   • 可以包含配置说明、实现技巧、问题解决方案等")
    print("   • 帮助AI更准确地理解和实现论文算法")
    
    print(f"\n📁 代码库路径: {config['repo_path']}")
    print(f"📄 主要论文指南: {config['paper_guide']}")
    
    if config["additional_guides"]:
        print(f"📚 配置的补充信息文档: {len(config['additional_guides'])} 个")
        for i, guide in enumerate(config["additional_guides"], 1):
            print(f"   {i}. {guide}")
    else:
        print("📚 补充信息文档: 未配置（在config['additional_guides']中添加路径）")
    
    print("\n💡 使用提示:")
    print("   1. 在 config['additional_guides'] 列表中添加您的补充信息文档路径")
    print("   2. 文档格式必须是 .md 或 .markdown")
    print("   3. 系统会自动验证文件存在性并处理")
    print("   4. 补充信息会在每个复现层级中被使用")
    
    if not config["additional_guides"]:
        print("\n⚠️  当前未配置补充信息文档，将仅使用主要论文指南")
        print("   如需使用补充信息功能，请修改 additional_guides 列表")
        return False
    
    # 验证补充信息文档路径
    valid_additional_guides = []
    if config["additional_guides"]:
        print("\n📄 验证补充信息文档...")
        for guide_path in config["additional_guides"]:
            if os.path.exists(guide_path):
                valid_additional_guides.append(guide_path)
                print(f"  ✅ {os.path.basename(guide_path)}")
            else:
                print(f"  ⚠️  文件不存在，将跳过: {guide_path}")
        
        if len(valid_additional_guides) < len(config["additional_guides"]):
            print(f"📋 补充信息文档: {len(valid_additional_guides)}/{len(config['additional_guides'])} 个有效")
    
    if not valid_additional_guides:
        print("\n❌ 没有有效的补充信息文档，无法演示补充信息功能")
        return False
    
    print(f"\n📚 将使用的补充信息文档: {len(valid_additional_guides)} 个")
    for i, guide in enumerate(valid_additional_guides, 1):
        print(f"   {i}. {os.path.basename(guide)}")
    
    # 如果配置了补充信息文档，则运行完整流程
    try:
        reproduction_system = PaperReproductionSystem()
        
        result = reproduction_system.reproduce_paper(
            repo_path=config["repo_path"],
            paper_guide=config["paper_guide"],
            additional_guides=valid_additional_guides,
            target_metrics=config["target_metrics"],
            max_iterations=config["max_iterations"]
        )
        
        return result.get("success", False)
        
    except Exception as e:
        print(f"❌ 示例运行失败: {str(e)}")
        return False


if __name__ == "__main__":
    print("🔬 Will Model Forget论文完整复现系统")
    print("目标: 完美复现论文算法和训练流程，保证实验完整性")
    print("流程: 分析 → 生成任务 → 执行任务 → 修改代码 → 验证结果")
    
    # 显示功能选项
    print("\n📋 可用功能:")
    print("  1. 标准复现流程（仅使用主要论文指南）")
    print("  2. 补充信息功能演示（展示如何配置补充文档）")
    print("  3. 查看目标指标摘要")
    
    try:
        choice = input("\n请选择功能 (1-3, 默认为1): ").strip()
        if not choice:
            choice = "1"
    except KeyboardInterrupt:
        print("\n👋 用户取消操作")
        sys.exit(0)
    
    if choice == "1":
        print("\n🚀 运行标准复现流程...")
        # 显示目标指标摘要
        show_target_metrics_summary()
        
        # 运行完整复现流程
        success = run_forget_paper_reproduction()
        
        if success:
            print("\n🎉 Will Model Forget论文复现完成！代码已实际修改！")
            sys.exit(0)
        else:
            print("\n⚠️ 复现过程中遇到问题，请检查日志")
            sys.exit(1)
            
    elif choice == "2":
        print("\n📚 展示补充信息功能...")
        run_forget_with_additional_guides_example()
        print("\n💡 如需实际使用补充信息功能，请:")
        print("   1. 创建您的补充信息markdown文档")
        print("   2. 在 run_forget_with_additional_guides_example() 函数中配置路径")
        print("   3. 重新运行此选项")
        
    elif choice == "3":
        print("\n📊 显示目标指标摘要...")
        show_target_metrics_summary()
        
    else:
        print(f"\n❌ 无效选择: {choice}")
        print("请运行程序并选择 1、2 或 3")
        sys.exit(1)