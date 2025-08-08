#!/usr/bin/env python3
"""
Agent框架演示脚本

展示如何使用智能Agent框架进行代码分析。

使用方法:
    python demo_agents.py
"""

from agents import AnalysisAgent
from pathlib import Path


def demo_analysis_agent():
    """演示代码分析Agent的使用"""
    print("🤖 Agent框架演示")
    print("=" * 50)
    
    # 示例代码
    sample_code = '''
def fibonacci(n):
    """计算斐波那契数列第n项"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    for i in range(10):
        print(f"fibonacci({i}) = {fibonacci(i)}")

if __name__ == "__main__":
    main()
'''
    
    try:
        print("🔍 创建代码分析Agent...")
        agent = AnalysisAgent()
        
        print("\n📝 示例代码:")
        print(sample_code)
        
        # 测试API连接
        print("\n🔗 测试API连接...")
        api_test = agent.test_api()
        print(f"API状态: {api_test['status']}")
        print(f"API响应: {api_test['response'][:100]}...")
        
        if api_test['status'] == 'success':
            print("\n🔍 开始分析代码...")
            
            # 结构分析
            print("\n📊 结构分析:")
            structure_result = agent.analyze_code(sample_code, "structure")
            if "error" not in structure_result:
                print(structure_result['result'][:300] + "...")
            else:
                print(f"❌ {structure_result['error']}")
            
            # 质量分析
            print("\n📈 质量分析:")
            quality_result = agent.analyze_code(sample_code, "quality")
            if "error" not in quality_result:
                print(quality_result['result'][:300] + "...")
            else:
                print(f"❌ {quality_result['error']}")
                
        else:
            print("❌ API连接失败，请检查配置")
            
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")


def show_config_guide():
    """显示配置指南"""
    print("\n📋 配置指南:")
    print("1. 在项目根目录创建 .env 文件")
    print("2. 在 .env 文件中添加以下配置:")
    print()
    print("AGENT_API_KEY=your_openrouter_api_key")
    print("AGENT_BASE_URL=https://openrouter.ai/api/v1")
    print("AGENT_MODEL=anthropic/claude-sonnet-4")
    print("AGENT_TEMPERATURE=0.7")
    print("AGENT_MAX_TOKENS=16384")
    print()
    print("3. 运行演示: python demo_agents.py")


def test_config():
    """测试配置是否正确"""
    print("\n⚙️ 配置测试:")
    try:
        from agents.config import AgentConfig
        config = AgentConfig()
        
        api_key = config.get_config("api_key")
        base_url = config.get_config("base_url")
        model = config.get_config("model")
        
        print(f"✅ API Key: {'已设置' if api_key else '❌ 未设置'}")
        print(f"✅ Base URL: {base_url if base_url else '❌ 未设置'}")
        print(f"✅ Model: {model if model else '❌ 未设置'}")
        
        return bool(api_key and base_url)
        
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False


if __name__ == "__main__":
    if test_config():
        demo_analysis_agent()
    else:
        show_config_guide() 