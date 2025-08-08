"""
JSON处理工具模块

提供Agent所需的各种JSON处理工具函数。

主要功能:
- JSON解析和修复
- 部分内容提取
- 格式化处理
"""

import json
import re
from typing import Dict, Any, List


def parse_json_response(response: str) -> Dict:
    """
    解析LLM的JSON响应 - 增强版本，更robust地处理各种格式问题
    
    参数:
        response (str): LLM的原始响应
    
    返回:
        Dict: 解析后的JSON数据
    """
    try:
        # 1. 首先尝试直接解析
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        # 2. 尝试提取JSON代码块
        if "```json" in response:
            json_start = response.find("```json") + 7
            json_end = response.find("```", json_start)
            if json_end == -1:
                # 没有结束标记，可能被截断
                json_str = response[json_start:].strip()
            else:
                json_str = response[json_start:json_end].strip()
        elif "{" in response and "}" in response:
            # 3. 寻找第一个{到最后一个}
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_str = response[json_start:json_end]
        else:
            json_str = response.strip()
        
        # 4. 尝试解析提取的JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # 5. 如果解析失败，尝试修复常见问题
            print(f"⚠️ JSON解析失败，尝试修复: {str(e)}")
            
            # 尝试修复截断的JSON
            fixed_json = try_fix_truncated_json(json_str)
            if fixed_json:
                try:
                    return json.loads(fixed_json)
                except json.JSONDecodeError:
                    pass
            
            # 如果所有尝试都失败，返回错误信息但包含部分数据
            return {
                "error": f"JSON解析失败: {str(e)}",
                "raw_response": response,
                "partial_tasks": extract_partial_tasks(response)
            }

    except Exception as e:
        return {
            "error": f"响应处理异常: {str(e)}",
            "raw_response": response
        }


def try_fix_truncated_json(json_str: str) -> str:
    """
    尝试修复被截断的JSON
    
    参数:
        json_str (str): 原始JSON字符串
    
    返回:
        str: 修复后的JSON字符串，如果无法修复返回None
    """
    try:
        # 查找最后一个完整的任务
        if '"tasks":' in json_str and '[' in json_str:
            # 找到tasks数组的开始
            tasks_start = json_str.find('"tasks":')
            array_start = json_str.find('[', tasks_start)
            
            # 统计大括号，找到最后一个完整的任务对象
            brace_count = 0
            last_complete_pos = array_start
            
            for i, char in enumerate(json_str[array_start:], array_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_complete_pos = i + 1
            
            # 构造修复后的JSON
            if last_complete_pos > array_start:
                before_tasks = json_str[:array_start+1]
                tasks_part = json_str[array_start+1:last_complete_pos]
                
                # 构造完整的JSON
                fixed = f'{before_tasks}{tasks_part}],"prevention_recommendations":"修复被截断的响应","testing_recommendations":"添加相关测试"}}'
                return fixed
                
    except Exception:
        pass
    
    return None


def extract_partial_tasks(response: str) -> List[Dict]:
    """
    从响应中提取部分任务信息，即使JSON不完整
    
    参数:
        response (str): 原始响应内容
    
    返回:
        List[Dict]: 提取到的任务列表
    """
    tasks = []
    try:
        # 使用正则表达式提取任务ID和基本信息
        
        # 查找task_id模式
        task_id_pattern = r'"task_id":\s*"([^"]+)"'
        task_ids = re.findall(task_id_pattern, response)
        
        # 查找fixing_type模式  
        fixing_type_pattern = r'"fixing_type":\s*"([^"]+)"'
        fixing_types = re.findall(fixing_type_pattern, response)
        
        # 查找文件路径模式
        file_pattern = r'"which_file_to_fix":\s*"([^"]+)"'
        files = re.findall(file_pattern, response)
        
        # 组合找到的信息
        max_len = max(len(task_ids), len(fixing_types), len(files))
        for i in range(max_len):
            task = {
                "task_id": task_ids[i] if i < len(task_ids) else f"partial_task_{i+1}",
                "priority": i + 1,
                "fixing_type": fixing_types[i] if i < len(fixing_types) else "Change_File",
                "which_file_to_fix": files[i] if i < len(files) else "",
                "fixing_plan_in_detail": "从截断响应中恢复的部分任务",
                "raw_code": "",
                "dependencies": [],
                "estimated_impact": "无法确定",
                "partial_recovery": True
            }
            tasks.append(task)
            
    except Exception:
        pass
        
    return tasks


def validate_json_structure(data: Dict, required_keys: List[str] = None) -> bool:
    """
    验证JSON结构的有效性
    
    参数:
        data (Dict): 要验证的数据
        required_keys (List[str]): 必需的键列表
    
    返回:
        bool: 结构是否有效
    """
    if not isinstance(data, dict):
        return False
    
    if required_keys:
        for key in required_keys:
            if key not in data:
                return False
    
    return True


def clean_json_response(response: str) -> str:
    """
    清理JSON响应，移除可能的干扰内容
    
    参数:
        response (str): 原始响应
    
    返回:
        str: 清理后的响应
    """
    # 移除markdown代码块标记
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'\s*```', '', response)
    
    # 移除多余的空白字符
    response = response.strip()
    
    # 移除可能的注释行
    lines = response.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('//') and not line.startswith('#'):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def format_json_for_display(data: Dict, indent: int = 2) -> str:
    """
    格式化JSON数据用于显示
    
    参数:
        data (Dict): 要格式化的数据
        indent (int): 缩进空格数
    
    返回:
        str: 格式化后的JSON字符串
    """
    try:
        return json.dumps(data, indent=indent, ensure_ascii=False)
    except Exception:
        return str(data)


def extract_field_values(json_data: Dict, field_name: str) -> List[Any]:
    """
    递归提取JSON中所有指定字段的值
    
    参数:
        json_data (Dict): JSON数据
        field_name (str): 要提取的字段名
    
    返回:
        List[Any]: 提取到的值列表
    """
    values = []
    
    def recursive_extract(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == field_name:
                    values.append(value)
                elif isinstance(value, (dict, list)):
                    recursive_extract(value)
        elif isinstance(obj, list):
            for item in obj:
                recursive_extract(item)
    
    recursive_extract(json_data)
    return values 