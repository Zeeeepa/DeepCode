"""
代码修改分析工具模块

提供Agent所需的各种代码修改和分析工具函数。

主要功能:
- 代码变更分析
- 修改历史记录
- 修改概述生成
- LLM输出内容清理
"""

import os
import json
import re
from typing import Dict, Any, List


def clean_llm_code_output(content: str, file_extension: str = ".py") -> str:
    """
    清理LLM输出的代码内容，移除markdown标记和其他格式问题
    
    用于确保LLM生成的代码内容可以直接写入文件而不会导致语法错误。
    主要处理markdown代码块标记、多余的空行等问题。
    
    参数:
        content (str): LLM输出的原始内容
        file_extension (str): 目标文件扩展名，用于确定清理策略
    
    返回:
        str: 清理后的纯代码内容
    """
    if not content or not content.strip():
        return ""
    
    # 记录原始长度
    original_length = len(content)
    
    # 1. 移除markdown代码块标记
    # 匹配 ```python, ```py, ``` 等开始标记
    content = re.sub(r'^```(?:python|py|)\s*\n', '', content, flags=re.MULTILINE)
    # 移除结尾的```标记
    content = re.sub(r'\n```\s*$', '', content)
    # 移除单独一行的```
    content = re.sub(r'^\s*```\s*$', '', content, flags=re.MULTILINE)
    
    # 2. 移除常见的markdown格式标记
    # 移除文件路径标记，如 "# filename.py" 或 "## filename.py"
    content = re.sub(r'^#{1,6}\s+[a-zA-Z0-9_./\\]+\.(py|js|ts|java|cpp|c|h).*$', '', content, flags=re.MULTILINE)
    
    # 3. 处理Python文件特有的问题
    if file_extension.lower() in ['.py', '.pyx', '.pyw']:
        # 移除可能的解释性文本（通常在代码前后）
        lines = content.split('\n')
        cleaned_lines = []
        code_started = False
        
        for line in lines:
            stripped = line.strip()
            
            # 检测代码开始的标志
            if not code_started:
                # Python代码的典型开始标志
                if (stripped.startswith(('import ', 'from ', 'def ', 'class ', '#', '"""', "'''", 'if __name__')) or
                    stripped.startswith(('try:', 'with ', 'for ', 'while ', '@')) or
                    re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=', stripped)):  # 变量赋值
                    code_started = True
                    cleaned_lines.append(line)
                elif stripped and not re.match(r'^[A-Za-z\s:,.!?-]+$', stripped):  # 不是纯文本描述
                    code_started = True
                    cleaned_lines.append(line)
                # 跳过看起来像解释文本的行
                continue
            else:
                cleaned_lines.append(line)
        
        if cleaned_lines:
            content = '\n'.join(cleaned_lines)
    
    # 4. 清理多余的空行（但保留必要的空行）
    # 移除开头和结尾的空行
    content = content.strip()
    
    # 将连续的多个空行替换为最多2个空行
    content = re.sub(r'\n{4,}', '\n\n\n', content)
    
    # 5. 确保文件以换行符结尾（符合Python规范）
    if content and not content.endswith('\n'):
        content += '\n'
    
    # 6. 最后检查：如果内容看起来仍然包含markdown或无效内容，尝试更激进的清理
    if '```' in content or content.strip().startswith('```'):
        print(f"⚠️ 内容仍包含markdown标记，尝试更激进的清理...")
        # 按行处理，只保留看起来像代码的行
        lines = content.split('\n')
        code_lines = []
        for line in lines:
            if not line.strip().startswith('```') and '```' not in line:
                code_lines.append(line)
        content = '\n'.join(code_lines)
    
    # 记录清理结果
    cleaned_length = len(content)
    if original_length != cleaned_length:
        print(f"🧹 内容清理完成: {original_length:,} → {cleaned_length:,} 字符 (减少 {original_length-cleaned_length:,})")
    
    return content


def validate_code_content(content: str, file_extension: str = ".py") -> Dict[str, Any]:
    """
    验证代码内容的有效性
    
    检查代码内容是否包含明显的格式问题或无效语法。
    
    参数:
        content (str): 要验证的代码内容
        file_extension (str): 文件扩展名
    
    返回:
        Dict[str, Any]: 验证结果，包含is_valid, issues, suggestions等字段
    """
    if not content or not content.strip():
        return {
            "is_valid": False,
            "issues": ["内容为空"],
            "suggestions": ["确保LLM生成了有效的代码内容"]
        }
    
    issues = []
    suggestions = []
    
    # 检查markdown残留
    if '```' in content:
        issues.append("包含markdown代码块标记")
        suggestions.append("使用clean_llm_code_output()清理内容")
    
    # 检查Python语法（简单检查）
    if file_extension.lower() == '.py':
        lines = content.split('\n')
        
        # 检查第一行是否有明显问题
        first_line = next((line.strip() for line in lines if line.strip()), "")
        if first_line.startswith('```'):
            issues.append("第一行包含markdown标记")
        
        # 检查是否包含基本的Python结构
        has_python_content = any(
            line.strip().startswith(('import ', 'from ', 'def ', 'class ', '#')) or
            '=' in line or 'return' in line or 'if' in line
            for line in lines
        )
        
        if not has_python_content:
            issues.append("不包含明显的Python代码结构")
            suggestions.append("检查LLM是否正确生成了Python代码")
    
    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "suggestions": suggestions,
        "content_length": len(content),
        "line_count": len(content.split('\n'))
    }


def analyze_code_changes(original: str, modified: str) -> list:
    """
    分析代码变化
    
    参数:
        original (str): 原始代码
        modified (str): 修改后代码
    
    返回:
        list: 变化列表
    """
    changes = []
    
    try:
        original_lines = original.splitlines()
        modified_lines = modified.splitlines()
        
        # 简单的变化检测
        if len(modified_lines) > len(original_lines):
            changes.append(f"增加了 {len(modified_lines) - len(original_lines)} 行代码")
        elif len(modified_lines) < len(original_lines):
            changes.append(f"删除了 {len(original_lines) - len(modified_lines)} 行代码")
        
        # 检测特定关键词的变化
        keywords = ['def ', 'class ', 'try:', 'except', 'if ', 'ZeroDivisionError', 'Exception']
        for keyword in keywords:
            original_count = original.count(keyword)
            modified_count = modified.count(keyword)
            if modified_count > original_count:
                changes.append(f"新增 {keyword.strip()} 相关代码")
            elif modified_count < original_count:
                changes.append(f"移除 {keyword.strip()} 相关代码")
        
        # 检测异常处理改进
        if 'ZeroDivisionError' in modified and 'ZeroDivisionError' not in original:
            changes.append("添加了专门的除零异常处理")
        
        if modified.count('try:') > original.count('try:'):
            changes.append("增强了异常处理机制")
        
        if modified.count('def ') > original.count('def '):
            changes.append("新增了函数定义")
        
    except Exception:
        changes.append("代码结构发生了变化")
    
    return changes if changes else ["修改了文件内容"]


def update_modification_history(output_dir: str, result: Dict[str, Any]) -> None:
    """
    更新修改历史记录
    
    参数:
        output_dir (str): 输出目录
        result (Dict[str, Any]): 修复结果
    """
    try:
        history_file = os.path.join(output_dir, "modification_history.json")
        
        # 读取现有历史
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = {
                "total_iterations": 0,
                "modifications": []
            }
        
        # 添加新的修改记录
        modification_record = {
            "iteration": result["iteration"],
            "timestamp": str(__import__('datetime').datetime.now()),
            "file_path": result["file_path"],
            "action_taken": result["action_taken"],
            "modification_summary": result["modification_summary"],
            "changes_made": result["changes_made"],
            "success": result["success"]
        }
        
        history["modifications"].append(modification_record)
        history["total_iterations"] = max(history["total_iterations"], result["iteration"])
        
        # 保存更新的历史
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 更新修改历史: {history_file}")
        
    except Exception as e:
        print(f"⚠️ 更新修改历史失败: {str(e)}")


def read_modification_history(output_dir: str) -> Dict[str, Any]:
    """
    读取修改历史记录
    
    参数:
        output_dir (str): 输出目录
    
    返回:
        Dict[str, Any]: 修改历史记录
    """
    try:
        history_file = os.path.join(output_dir, "modification_history.json")
        
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            return history
        else:
            return {
                "total_iterations": 0,
                "modifications": []
            }
    
    except Exception as e:
        print(f"⚠️ 读取修改历史失败: {str(e)}")
        return {
            "total_iterations": 0,
            "modifications": [],
            "error": str(e)
        }


def generate_code_summary(code_content: str, file_path: str) -> str:
    """
    生成代码概述
    
    参数:
        code_content (str): 代码内容
        file_path (str): 文件路径
    
    返回:
        str: 代码概述
    """
    try:
        lines = code_content.splitlines()
        
        # 统计基本信息
        total_lines = len(lines)
        code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        
        # 统计函数和类
        function_count = code_content.count('def ')
        class_count = code_content.count('class ')
        
        # 检测主要特征
        features = []
        if 'try:' in code_content:
            features.append("异常处理")
        if 'import ' in code_content:
            features.append("模块导入")
        if 'if __name__ == "__main__"' in code_content:
            features.append("主程序入口")
        
        summary = f"文件: {os.path.basename(file_path)}\n"
        summary += f"总行数: {total_lines}, 代码行: {code_lines}, 注释行: {comment_lines}\n"
        summary += f"函数数: {function_count}, 类数: {class_count}\n"
        
        if features:
            summary += f"特征: {', '.join(features)}"
        
        return summary
        
    except Exception as e:
        return f"生成代码概述失败: {str(e)}"


def validate_code_syntax(code_content: str, file_path: str = None) -> Dict[str, Any]:
    """
    验证代码语法
    
    参数:
        code_content (str): 代码内容
        file_path (str): 文件路径（可选）
    
    返回:
        Dict[str, Any]: 验证结果
    """
    try:
        if file_path and file_path.endswith('.py'):
            # Python语法检查
            try:
                compile(code_content, file_path or '<string>', 'exec')
                return {
                    "valid": True,
                    "language": "python",
                    "message": "语法检查通过"
                }
            except SyntaxError as e:
                return {
                    "valid": False,
                    "language": "python",
                    "error": str(e),
                    "line": e.lineno,
                    "message": f"语法错误在第{e.lineno}行: {e.msg}"
                }
        else:
            # 基本检查
            return {
                "valid": True,
                "language": "unknown",
                "message": "未进行语法检查（非Python文件）"
            }
    
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "message": f"语法检查异常: {str(e)}"
        }


def extract_imports(code_content: str) -> List[str]:
    """
    提取代码中的导入语句
    
    参数:
        code_content (str): 代码内容
    
    返回:
        List[str]: 导入语句列表
    """
    imports = []
    
    try:
        lines = code_content.splitlines()
        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
    
    except Exception:
        pass
    
    return imports


def extract_functions(code_content: str) -> List[Dict[str, Any]]:
    """
    提取代码中的函数定义
    
    参数:
        code_content (str): 代码内容
    
    返回:
        List[Dict[str, Any]]: 函数信息列表
    """
    functions = []
    
    try:
        lines = code_content.splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('def '):
                # 提取函数名
                func_def = stripped[4:]  # 去掉 'def '
                if '(' in func_def:
                    func_name = func_def[:func_def.index('(')]
                    functions.append({
                        "name": func_name.strip(),
                        "line": i,
                        "definition": stripped
                    })
    
    except Exception:
        pass
    
    return functions


def extract_classes(code_content: str) -> List[Dict[str, Any]]:
    """
    提取代码中的类定义
    
    参数:
        code_content (str): 代码内容
    
    返回:
        List[Dict[str, Any]]: 类信息列表
    """
    classes = []
    
    try:
        lines = code_content.splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('class '):
                # 提取类名
                class_def = stripped[6:]  # 去掉 'class '
                if ':' in class_def:
                    class_name = class_def[:class_def.index(':')]
                    if '(' in class_name:
                        class_name = class_name[:class_name.index('(')]
                    classes.append({
                        "name": class_name.strip(),
                        "line": i,
                        "definition": stripped
                    })
    
    except Exception:
        pass
    
    return classes


def calculate_code_complexity(code_content: str) -> Dict[str, Any]:
    """
    计算代码复杂度
    
    参数:
        code_content (str): 代码内容
    
    返回:
        Dict[str, Any]: 复杂度分析结果
    """
    try:
        lines = code_content.splitlines()
        total_lines = len(lines)
        code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        
        # 统计控制结构
        if_count = code_content.count('if ')
        for_count = code_content.count('for ')
        while_count = code_content.count('while ')
        try_count = code_content.count('try:')
        
        # 简单的复杂度计算
        complexity_score = if_count + for_count + while_count + try_count
        
        if complexity_score < 5:
            complexity_level = "低"
        elif complexity_score < 15:
            complexity_level = "中"
        else:
            complexity_level = "高"
        
        return {
            "total_lines": total_lines,
            "code_lines": code_lines,
            "control_structures": {
                "if_statements": if_count,
                "for_loops": for_count,
                "while_loops": while_count,
                "try_blocks": try_count
            },
            "complexity_score": complexity_score,
            "complexity_level": complexity_level
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "complexity_level": "无法计算"
        } 