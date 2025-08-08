#!/usr/bin/env python3
"""
论文算法分析脚本 - 复现导向版本
功能：使用Gemini最先进模型通过OpenRouter API分析论文，专门用于指导完整复现
重点：提取详细的核心算法、实验方法、参数配置等复现所需信息
作者：AI助手
"""

import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv


def load_api_key():
    """
    从.env文件加载OpenRouter API密钥
    
    Returns:
        str: API密钥，如果失败返回None
    """
    # 获取项目根目录
    root_dir = Path(__file__).parent.parent
    env_file = root_dir / '.env'
    
    print(f"查找.env文件: {env_file}")
    
    if not env_file.exists():
        print("错误：找不到.env文件")
        print("请在项目根目录创建.env文件，内容如下：")
        print("OPENROUTER_API_KEY=your_openrouter_api_key_here")
        return None
    
    # 加载环境变量
    load_dotenv(env_file)
    
    api_key = os.getenv('AGENT_API_KEY')
    if not api_key:
        print("错误：.env文件中未找到OPENROUTER_API_KEY")
        print("请确保.env文件包含以下内容：")
        print("OPENROUTER_API_KEY=your_openrouter_api_key_here")
        return None
    
    print("✓ 成功加载API密钥")
    return api_key


def read_extracted_text(file_path):
    """
    读取提取的论文文本
    
    Args:
        file_path (str): 文本文件路径
        
    Returns:
        str: 文件内容，如果失败返回None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            print(f"警告：文件 {file_path} 为空")
            return None
            
        print(f"✓ 成功读取文件 {file_path}，内容长度: {len(content)} 字符")
        return content
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None
    except Exception as e:
        print(f"错误：读取文件 {file_path} 时出现异常 - {str(e)}")
        return None


def call_gemini_api(api_key, paper_content, paper_name):
    """
    调用Gemini最先进模型API分析论文算法（复现导向）
    
    Args:
        api_key (str): OpenRouter API密钥
        paper_content (str): 论文文本内容
        paper_name (str): 论文名称
        
    Returns:
        str: 分析结果，如果失败返回None
    """
    
    # 构造复现导向的分析提示词
    prompt = f"""你是一位经验丰富的算法研究员，需要为完整复现以下学术论文提供详细指导。请分析论文并提供复现所需的所有关键信息。

论文名称：{paper_name}

请按照以下结构进行详细分析，确保提供足够的细节来支持完整复现：

## 1. 论文背景与问题定义
- 研究背景和动机
- 要解决的具体问题（用数学公式精确定义）
- 输入输出定义（数据格式、维度、约束条件）
- 评估指标和成功标准

## 2. 核心算法详细剖析 ⭐️ [重点]
### 2.1 算法整体架构
- 算法名称和分类
- 主要组件和模块划分
- 数据流和处理流程
- 关键创新点和技术贡献

### 2.2 算法步骤详解
- 完整的算法流程（步骤编号）
- 每个步骤的具体操作和计算
- 关键函数和数学公式
- 条件判断和分支逻辑
- 循环和迭代过程

### 2.3 核心算法伪代码
请提供详细的伪代码，包括：
- 初始化过程
- 主要循环结构
- 关键计算步骤
- 更新规则和优化过程
- 终止条件

## 3. 重要参数和超参数配置 ⭐️ [复现关键]
### 3.1 模型参数
- 网络架构参数（层数、节点数、激活函数）
- 模型特有参数及其含义
- 参数初始化方法

### 3.2 训练超参数
- 学习率及其调度策略
- 批次大小和训练轮数
- 优化器选择和配置
- 正则化参数
- 其他重要超参数及其取值

### 3.3 算法特定配置
- 算法中的阈值和控制参数
- 采样策略和概率参数
- 搜索和探索参数

## 4. 实验设置完整复现指南 ⭐️ [关键]
### 4.1 数据集和预处理
- 使用的具体数据集（名称、版本、获取方式）
- 数据预处理步骤（归一化、分割、增强）
- 训练/验证/测试集划分比例和方法
- 数据格式和存储要求

### 4.2 计算环境要求
- 硬件配置（GPU型号、内存要求）
- 软件环境（Python版本、深度学习框架）
- 依赖库和版本要求

### 4.3 训练过程详解
- 完整的训练流程
- 损失函数定义和计算
- 梯度更新策略
- 早停和检查点保存策略
- 训练监控和日志记录

## 5. 基线方法和对比实验
### 5.1 对比方法
- 基线算法的具体实现
- 公平对比的设置原则
- 相同的数据和评估条件

### 5.2 消融实验
- 移除哪些组件进行测试
- 各组件的贡献分析
- 关键设计选择的验证

## 6. 评估指标和结果分析
### 6.1 评估方法
- 具体的评估指标定义
- 评估代码实现要点
- 统计显著性测试方法

### 6.2 预期结果
- 主要实验结果的数值
- 性能提升的量化分析
- 结果的可重复性说明

## 7. 实现细节和注意事项 ⭐️ [避坑指南]
### 7.1 关键实现细节
- 容易出错的实现要点
- 数值稳定性考虑
- 边界条件处理
- 内存和计算效率优化

### 7.2 常见问题和解决方案
- 可能遇到的训练问题
- 调试和验证方法
- 性能优化建议

## 8. 复现检查清单
- [ ] 数据集准备和预处理验证
- [ ] 模型架构正确实现
- [ ] 参数配置完全匹配
- [ ] 训练过程监控正常
- [ ] 评估结果达到预期
- [ ] 消融实验验证有效

## 9. 扩展和改进方向
- 算法的局限性分析
- 可能的改进方案
- 适用场景的扩展
- 未来研究方向

请用中文详细分析，确保每个部分都包含足够的技术细节来支持完整复现。特别注意提取论文中的数值、公式、具体配置等关键信息。

论文内容：
{paper_content}"""

    # API请求配置 - 使用最先进的Gemini模型
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost:3000",
        "X-Title": "Paper Algorithm Reproduction Analysis"
    }
    
    data = {
        "model": "google/gemini-2.5-pro",  # Gemini最先进的实验模型
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 16384,  # 增加token限制以获得更详细的输出
        "temperature": 0.1   # 降低温度以获得更准确和一致的分析
    }
    
    print("正在调用Gemini最先进模型进行复现导向分析...")
    print(f"模型: {data['model']}")
    print(f"输入内容长度: {len(paper_content)} 字符")
    print(f"最大输出token: {data['max_tokens']}")
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=180)  # 增加超时时间
        
        if response.status_code == 200:
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                analysis = result['choices'][0]['message']['content']
                print("✓ API调用成功")
                
                # 显示生成内容的统计信息
                token_usage = result.get('usage', {})
                if token_usage:
                    print(f"Token使用情况:")
                    print(f"  输入token: {token_usage.get('prompt_tokens', 'N/A')}")
                    print(f"  输出token: {token_usage.get('completion_tokens', 'N/A')}")
                    print(f"  总计token: {token_usage.get('total_tokens', 'N/A')}")
                
                return analysis
            else:
                print("错误：API响应格式异常")
                print(f"响应内容: {result}")
                return None
                
        else:
            print(f"错误：API调用失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("错误：API调用超时（可能是生成内容较长）")
        return None
    except requests.exceptions.RequestException as e:
        print(f"错误：网络请求异常 - {str(e)}")
        return None
    except Exception as e:
        print(f"错误：API调用过程中出现异常 - {str(e)}")
        return None


def save_analysis_result(analysis, output_file):
    """
    保存分析结果到文件
    
    Args:
        analysis (str): 分析结果
        output_file (str): 输出文件路径
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # 添加文件头部信息
            f.write("# 论文算法复现指南\n\n")
            f.write("*本文档由AI自动生成，专门用于指导论文算法的完整复现*\n\n")
            f.write("---\n\n")
            f.write(analysis)
            
        print(f"✓ 复现指南已保存到: {output_file}")
        
        # 显示文件大小信息
        file_size = Path(output_file).stat().st_size
        print(f"  文件大小: {file_size/1024:.1f} KB")
        
    except Exception as e:
        print(f"错误：保存分析结果时出现异常 - {str(e)}")


def analyze_single_paper(api_key, txt_file, script_dir):
    """
    分析单个论文（复现导向）
    
    Args:
        api_key (str): API密钥
        txt_file (str): 文本文件名
        script_dir (Path): 脚本目录
        
    Returns:
        bool: 是否成功
    """
    txt_path = script_dir / txt_file
    
    if not txt_path.exists():
        print(f"警告：文件不存在 - {txt_file}")
        return False
    
    # 读取论文内容
    paper_content = read_extracted_text(str(txt_path))
    if not paper_content:
        return False
    
    # 生成论文名称
    paper_name = txt_file.replace('_extracted_text.txt', '').replace('_', ' ')
    
    print(f"\n分析论文（复现导向）: {paper_name}")
    print("=" * 70)
    
    # 调用API分析
    analysis = call_gemini_api(api_key, paper_content, paper_name)
    if not analysis:
        return False
    
    # 生成输出文件名
    output_file = script_dir / txt_file.replace('_extracted_text.txt', '_reproduction_guide.md')
    
    # 保存结果
    save_analysis_result(analysis, str(output_file))
    return True


def main():
    """主函数"""
    print("论文算法复现指导工具")
    print("使用Gemini最先进模型 - 专门用于论文复现")
    print("=" * 70)
    
    # 加载API密钥
    api_key = load_api_key()
    if not api_key:
        return
    
    script_dir = Path(__file__).parent
    
    # 查找所有提取的文本文件
    txt_files = list(script_dir.glob("*_extracted_text.txt"))
    
    if not txt_files:
        print("错误：未找到任何提取的文本文件 (*_extracted_text.txt)")
        print("请先运行 extract_text.py 提取论文文本")
        return
    
    print(f"找到 {len(txt_files)} 个文本文件:")
    for txt_file in txt_files:
        print(f"  - {txt_file.name}")
    
    print(f"\n开始生成复现指南...")
    print("注意：使用先进模型可能需要较长时间，请耐心等待")
    
    successful_analyses = 0
    
    # 分析每个文件
    for txt_file in txt_files:
        success = analyze_single_paper(api_key, txt_file.name, script_dir)
        if success:
            successful_analyses += 1
    
    # 输出总结
    print("\n" + "=" * 70)
    print("复现指南生成完成总结:")
    print(f"成功生成: {successful_analyses}/{len(txt_files)} 个复现指南")
    
    if successful_analyses > 0:
        print("\n生成的复现指南文件:")
        guide_files = list(script_dir.glob("*_reproduction_guide.md"))
        for guide_file in guide_files:
            file_size = guide_file.stat().st_size
            print(f"  ✓ {guide_file.name} ({file_size/1024:.1f} KB)")
        
        print("\n📋 复现建议:")
        print("1. 仔细阅读生成的复现指南")
        print("2. 按照检查清单逐项验证")
        print("3. 特别注意标记为⭐️的重点部分")
        print("4. 遇到问题时参考'常见问题和解决方案'部分")


if __name__ == "__main__":
    main() 