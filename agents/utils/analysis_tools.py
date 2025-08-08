"""
错误分析工具模块

提供Agent所需的各种错误分析工具函数。

主要功能:
- 错误类型识别
- 严重性评估
- 输出模式分析
- 任务验证和增强
"""

import re
from typing import Dict, Any, List






def analyze_output_patterns(stdout: str) -> Dict[str, Any]:
    """
    分析输出模式
    
    参数:
        stdout (str): 程序输出
    
    返回:
        Dict[str, Any]: 分析结果
    """
    analysis = {
        "has_traceback": False,
        "has_error": False,
        "has_warning": False,
        "has_success_indicators": False,
        "line_count": 0,
        "error_keywords": [],
        "file_mentions": [],
        "function_mentions": []
    }
    
    lines = stdout.splitlines()
    analysis["line_count"] = len(lines)
    
    # 检查traceback
    if "Traceback" in stdout:
        analysis["has_traceback"] = True
    
    # 检查错误关键词
    error_patterns = [
        r"Error\b", r"Exception\b", r"Failed\b", r"fault\b",
        r"invalid\b", r"cannot\b", r"unable\b"
    ]
    
    for pattern in error_patterns:
        matches = re.findall(pattern, stdout, re.IGNORECASE)
        if matches:
            analysis["has_error"] = True
            analysis["error_keywords"].extend(matches)
    
    # 检查警告
    warning_patterns = [r"Warning\b", r"Warn\b", r"Deprecated\b"]
    for pattern in warning_patterns:
        if re.search(pattern, stdout, re.IGNORECASE):
            analysis["has_warning"] = True
            break
    
    # 检查成功指示器
    success_patterns = [
        r"success", r"completed", r"done", r"ok\b", r"passed"
    ]
    for pattern in success_patterns:
        if re.search(pattern, stdout, re.IGNORECASE):
            analysis["has_success_indicators"] = True
            break
    
    # 提取文件提及
    file_patterns = [
        r'File "([^"]+)"',
        r"File '([^']+)'",
        r'([a-zA-Z_][a-zA-Z0-9_/\\]*\.py)'
    ]
    for pattern in file_patterns:
        matches = re.findall(pattern, stdout)
        analysis["file_mentions"].extend(matches)
    
    # 提取函数提及
    function_pattern = r'in ([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    function_matches = re.findall(function_pattern, stdout)
    analysis["function_mentions"] = function_matches
    
    return analysis


def validate_and_enhance_tasks(tasks: List[Dict]) -> List[Dict]:
    """
    验证并增强任务列表
    
    参数:
        tasks (List[Dict]): 原始任务列表
    
    返回:
        List[Dict]: 验证和增强后的任务列表
    """
    enhanced_tasks = []
    
    for i, task in enumerate(tasks):
        # 确保必需字段存在
        enhanced_task = {
            "task_id": task.get("task_id", f"task_{i+1}"),
            "priority": task.get("priority", i + 1),
            "fixing_type": task.get("fixing_type", "Change_File"),
            "which_file_to_fix": task.get("which_file_to_fix", ""),
            "fixing_plan_in_detail": task.get("fixing_plan_in_detail", ""),
            "raw_code": task.get("raw_code", ""),
            "dependencies": task.get("dependencies", []),
            "estimated_impact": task.get("estimated_impact", "medium")
        }
        
        # 验证fixing_type
        valid_types = ["Add_File", "Change_File", "Delete_File"]
        if enhanced_task["fixing_type"] not in valid_types:
            enhanced_task["fixing_type"] = "Change_File"
        
        # 验证priority
        if not isinstance(enhanced_task["priority"], int):
            enhanced_task["priority"] = i + 1
        
        # 增强估计影响
        if not enhanced_task["estimated_impact"]:
            if enhanced_task["fixing_type"] == "Add_File":
                enhanced_task["estimated_impact"] = "low"
            elif enhanced_task["fixing_type"] == "Delete_File":
                enhanced_task["estimated_impact"] = "high"
            else:
                enhanced_task["estimated_impact"] = "medium"
        
        enhanced_tasks.append(enhanced_task)
    
    return enhanced_tasks


def generate_execution_plan(tasks: List[Dict]) -> Dict[str, Any]:
    """
    生成执行计划
    
    参数:
        tasks (List[Dict]): 任务列表
    
    返回:
        Dict[str, Any]: 执行计划
    """
    if not tasks:
        return {
            "total_tasks": 0,
            "execution_order": [],
            "risk_assessment": "无任务需要执行"
        }
    
    # 按依赖关系和优先级排序
    sorted_tasks = sorted(tasks, key=lambda x: (x.get("priority", 99), len(x.get("dependencies", []))))
    
    execution_order = [task["task_id"] for task in sorted_tasks]
    
    # 风险评估
    high_risk_count = sum(1 for task in tasks if task.get("estimated_impact") == "high")
    total_tasks = len(tasks)
    
    if high_risk_count > total_tasks * 0.5:
        risk_level = "高风险"
    elif high_risk_count > total_tasks * 0.2:
        risk_level = "中风险"
    else:
        risk_level = "低风险"
    
    return {
        "total_tasks": total_tasks,
        "execution_order": execution_order,
        "risk_assessment": f"{risk_level}: {high_risk_count}/{total_tasks} 个高影响任务",
        "estimated_duration": f"{total_tasks * 2}-{total_tasks * 5} 分钟",
        "dependencies_resolved": True
    }


def generate_fallback_result(error_message: str) -> Dict[str, Any]:
    """
    生成备用结果
    
    参数:
        error_message (str): 错误信息
    
    返回:
        Dict[str, Any]: 备用分析结果
    """
    return {
        "analysis_stages": {
            "file_identification": {"error": error_message},
            "file_reading": {"error": "跳过"},
            "deep_analysis": {"error": "跳过"}
        },
        "tasks": [{
            "task_id": "fallback_task",
            "priority": 1,
            "fixing_type": "Change_File",
            "which_file_to_fix": "",
            "fixing_plan_in_detail": f"由于分析失败({error_message})，需要手动检查",
            "raw_code": "",
            "dependencies": [],
            "estimated_impact": "unknown"
        }],
        "execution_plan": {
            "total_tasks": 1,
            "execution_order": ["fallback_task"],
            "risk_assessment": "分析失败，需要手动介入"
        },
        "fallback": True,
        "error": error_message
    }


def analyze_error_evolution(current_stdout: str, modification_history: Dict[str, Any], iteration: int) -> str:
    """
    分析错误演化
    
    参数:
        current_stdout (str): 当前错误输出
        modification_history (Dict[str, Any]): 修改历史
        iteration (int): 当前迭代次数
    
    返回:
        str: 错误演化分析
    """
    if not modification_history or iteration <= 1:
        return "首次分析，无历史对比数据"
    
    modifications = modification_history.get("modifications", [])
    if not modifications:
        return "无修改历史记录"
    
    # 简单的错误演化分析
    # 检查是否有改进
    if "success" in current_stdout.lower() or len(current_stdout.strip()) == 0:
        return "✅ 错误已解决"
    
    # 检查错误演化
    analysis = f"第{iteration}次迭代分析:\n"
    analysis += f"- 当前输出长度: {len(current_stdout)} 字符\n"
    analysis += f"- 历史修改次数: {len(modifications)}\n"
    
    # 检查最近的修改是否相关
    recent_mod = modifications[-1] if modifications else None
    if recent_mod:
        analysis += f"- 最近修改: {recent_mod.get('modification_summary', 'N/A')}\n"
    
    return analysis


def summarize_call_graph(function_dependencies: Dict) -> Dict:
    """
    总结调用图
    
    参数:
        function_dependencies (Dict): 函数依赖关系
    
    返回:
        Dict: 调用图摘要
    """
    if not function_dependencies:
        return {"total_functions": 0, "total_calls": 0, "complexity": "low"}
    
    total_functions = len(function_dependencies)
    total_calls = sum(len(calls) for calls in function_dependencies.values())
    
    # 简单的复杂度评估
    if total_calls > total_functions * 3:
        complexity = "high"
    elif total_calls > total_functions * 1.5:
        complexity = "medium"
    else:
        complexity = "low"
    
    return {
        "total_functions": total_functions,
        "total_calls": total_calls,
        "complexity": complexity,
        "avg_calls_per_function": total_calls / total_functions if total_functions > 0 else 0
    } 