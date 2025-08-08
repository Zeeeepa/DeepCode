"""
增强版分析Agent

实现多阶段分析流程：
1. 文件识别阶段：基于stdout + RepoIndex识别需要读取的文件
2. 文件读取阶段：完整读取文件内容
3. 深度分析阶段：基于完整信息进行分析
4. 多任务生成：生成有依赖关系的修复任务列表

主要功能:
- analyze_error(): 多阶段错误分析和任务生成
- _identify_relevant_files(): 识别相关文件
- _read_files_completely(): 完整读取文件内容
- _deep_analysis_with_content(): 基于完整内容的深度分析
- _generate_task_sequence(): 生成任务序列
"""

import sys

import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
from .base_agent import BaseAgent
from .utils import (
    file_tools, json_tools, analysis_tools,
    read_files_completely, parse_json_response, 
    validate_and_enhance_tasks, 
    generate_execution_plan,
    generate_fallback_result, extract_file_from_error,
    get_current_timestamp, get_basic_file_list,
    estimate_context_usage, create_repo_index
)


class AnalyzerAgent(BaseAgent):
    """
    增强版分析Agent
    
    通过多阶段分析流程，从错误输出到完整文件内容的深度理解，
    生成系统性的多任务修复方案。
    """
    
    def __init__(self, **kwargs):
        """
        初始化分析Agent
        
        参数:
            **kwargs: 配置参数
        """
        super().__init__(**kwargs)
        self.max_files_per_analysis = 10  # 每次分析的最大文件数
        self.max_tokens_estimate = 100000  # 预估的最大token数

    def analyze_error(self, stdout: str, repo_path: str, indexed_repo_data: Dict[str, Any] = None, expected_behavior: str = None) -> Dict[str, Any]:
        """
        多阶段错误分析和任务生成
        
        参数:
            stdout (str): 程序输出（包含错误信息）
            repo_path (str): 代码库路径
            indexed_repo_data (dict, optional): 代码库索引数据
            expected_behavior (str, optional): 期望的程序行为描述
        
        返回:
            dict: 多任务分析结果
            {
                "analysis_stages": {
                    "file_identification": {...},
                    "file_reading": {...},
                    "deep_analysis": {...}
                },
                "tasks": [
                    {
                        "task_id": str,
                        "priority": int,
                        "fixing_type": str,
                        "which_file_to_fix": str,
                        "fixing_plan_in_detail": str,
                        "raw_code": str,
                        "dependencies": [str],
                        "estimated_impact": str
                    }
                ],
                "execution_plan": {
                    "total_tasks": int,
                    "execution_order": [str],
                    "risk_assessment": str
                }
            }
        """
        try:
            # 如果没有提供索引数据，尝试生成
            if not indexed_repo_data:
                indexed_repo_data = self._get_or_create_repo_index(repo_path)
            
            print("🔍 开始多阶段错误分析...")
            
            # 阶段1: 识别相关文件
            print("📋 阶段1: 识别需要读取的文件")
            file_identification_result = self._identify_relevant_files(stdout, indexed_repo_data, expected_behavior)
            
            if not file_identification_result.get("files_to_read"):
                return generate_fallback_result("无法识别相关文件")
            
            # 阶段2: 完整读取文件内容
            print("📖 阶段2: 读取完整文件内容")
            file_reading_result = read_files_completely(
                file_identification_result["files_to_read"], 
                repo_path
            )
            
            if not file_reading_result.get("file_contents"):
                return generate_fallback_result("无法读取文件内容")
            
            # 阶段3: 基于完整内容的深度分析
            print("🧠 阶段3: 深度分析和多任务生成")
            deep_analysis_result = self._deep_analysis_with_content(
                stdout, 
                indexed_repo_data, 
                file_reading_result["file_contents"],
                expected_behavior
            )
            
            # 构建最终结果
            result = {
                "analysis_stages": {
                    "file_identification": file_identification_result,
                    "file_reading": file_reading_result,
                    "deep_analysis": deep_analysis_result
                },
                "tasks": deep_analysis_result.get("tasks", []),
                "execution_plan": generate_execution_plan(deep_analysis_result.get("tasks", []))
            }
            
            print(f"✅ 分析完成: 生成了 {len(result['tasks'])} 个修复任务")
            return result
            
        except Exception as e:
            print(f"❌ 分析过程出现异常: {str(e)}")
            return generate_fallback_result(f"分析异常: {str(e)}")

    def _identify_relevant_files(self, stdout: str, indexed_repo_data: Dict[str, Any], expected_behavior: str = None) -> Dict[str, Any]:
        """
        阶段1: 识别需要读取的相关文件
        
        参数:
            stdout (str): 程序错误输出
            indexed_repo_data (dict): 代码库索引数据
            expected_behavior (str, optional): 期望的程序行为描述
        
        返回:
            dict: 文件识别结果
        """
        try:
            system_prompt = """你是一个专业的代码错误分析专家。请基于程序错误输出和项目索引，识别需要读取的相关文件。

                                请严格按照以下JSON格式返回结果：
                                {
                                    "files_to_read": [
                                        {
                                            "file_path": "相对于项目根目录的文件路径",
                                            "reason": "需要读取此文件的原因",
                                            "priority": "high/medium/low",
                                            "analysis_focus": "在此文件中需要重点关注的内容"
                                        }
                                    ],
                                    "analysis_reasoning": "文件选择的整体reasoning"
                                }

                                文件识别策略：
                                1. 直接错误文件：从错误堆栈中提取的直接相关文件
                                2. 依赖文件：根据调用图分析的上下游文件
                                3. 配置文件：可能影响行为的配置文件
                                4. 测试文件：相关的测试文件（如果存在）
                                5. 期望行为相关：与实现期望程序行为直接相关的文件
                                6. 限制数量：最多选择10个最相关的文件

                                优先级说明：
                                - high: 错误直接发生的文件，必须修复
                                - medium: 调用链相关文件，可能需要修改
                                - low: 配置或测试文件，可能需要更新

                                🚫 严格禁止选择以下类型的文件：
                                - 备份文件：*.backup, *.backup_*, *.bak
                                - 调试输出：debug_output/*, debug_report*, modification_history*
                                - 隐藏文件：.*, .*/*
                                - 编译文件：__pycache__/*, *.pyc, *.pyo
                                - 版本控制：.git/*, .svn/*
                                - IDE文件：.vscode/*, .idea/*
                                - 虚拟环境：venv/*, env/*, virtualenv/*
                                - 依赖目录：node_modules/*
                                - 日志文件：*.log, logs/*
                                - 临时文件：*.tmp, *.temp
                                - 系统文件：.DS_Store, Thumbs.db

                                ✅ 只选择真正的源代码文件和配置文件：
                                - Python源码：*.py
                                - 配置文件：*.json, *.yaml, *.yml, *.toml, *.ini, *.cfg
                                - 文档文件：*.md, *.txt
                                - Web文件：*.html, *.css, *.js
                                - 其他源码：*.java, *.cpp, *.c, *.h等"""

                                            # 构建用户提示词
            user_prompt = f"""## 程序错误输出
                                ```
                                {stdout}
                                ```

                                ## 项目索引信息
                                ```json
                                {json.dumps({
                                    "project_name": indexed_repo_data.get("project_name", ""),
                                    "directory_structure": indexed_repo_data.get("directory_structure", [])[:20],
                                    "files": {k: v for i, (k, v) in enumerate(indexed_repo_data.get("files", {}).items()) if i < 8},
                                    "function_dependencies": {
                                        "has_analysis": indexed_repo_data.get("analysis_info", {}).get("has_dependency_analysis", False),
                                        "call_graph_summary": self._summarize_call_graph(indexed_repo_data.get("function_dependencies", {}))
                                    }
                                }, indent=2, ensure_ascii=False)}
                                ```

                                请分析错误输出，识别需要读取的关键文件。重点考虑：
                                1. 错误堆栈中直接提到的文件
                                2. 可能包含错误根因的文件
                                3. 需要修改的相关依赖文件
                                4. 可能需要更新的配置或测试文件"""
            
            # 添加期望行为指导
            expected_behavior_section = ""
            if expected_behavior:
                expected_behavior_section = f"""
                                
                                ## 期望的程序行为
                                ```
                                {expected_behavior}
                                ```
                                
                                在识别相关文件时，请特别关注：
                                5. 与实现期望行为相关的核心文件
                                6. 可能阻止程序达到期望行为的文件
                                7. 需要修改以确保符合期望行为的配置文件
                                
                                请确保选择的文件有助于让程序最终实现期望的行为。"""
            
            user_prompt += expected_behavior_section

            print('\n\n')

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.call_llm(messages, max_tokens=16384, temperature=0.2)
            
            # 解析JSON响应
            result = parse_json_response(response)
            
            # 验证和清理结果
            if "files_to_read" in result:
                # 限制文件数量
                result["files_to_read"] = result["files_to_read"][:self.max_files_per_analysis]
                
                # 验证文件路径格式
                for file_info in result["files_to_read"]:
                    if "priority" not in file_info:
                        file_info["priority"] = "medium"
                    if "analysis_focus" not in file_info:
                        file_info["analysis_focus"] = "整体代码逻辑"
            
            return result
            
        except Exception as e:
            return {
                "files_to_read": [],
                "analysis_reasoning": f"文件识别失败: {str(e)}",
                "error": str(e)
            }

    def _deep_analysis_with_content(self, stdout: str, indexed_repo_data: Dict[str, Any], 
                                file_contents: Dict[str, Any], expected_behavior: str = None) -> Dict[str, Any]:
        """
        阶段3: 基于完整文件内容的深度分析
        
        参数:
            stdout (str): 程序错误输出
            indexed_repo_data (dict): 代码库索引数据
            file_contents (dict): 完整文件内容
            expected_behavior (str, optional): 期望的程序行为描述
        
        返回:
            dict: 深度分析结果，包含多个任务
        """
        try:
            system_prompt = """你是专业的代码错误修复专家。你的首要且唯一目标是：分析错误原因并按照输出格式要求给出详细的修复方案，让程序能够成功运行。
                                你将会根据用户提供的信息来给出你的解决方案，具体形式是给出一个task列表

                                请严格按照以下JSON格式返回结果：
                                {
                                    "root_cause_analysis": "错误的根本原因分析",
                                    "impact_analysis": "错误影响范围和传播路径分析",
                                    "tasks": [
                                        {
                                            "task_id": "任务唯一标识符",
                                            "priority": 1,
                                            "fixing_type": "Add_File 或 Change_File",
                                            "which_file_to_fix": "具体文件路径",
                                            "specific_location": "具体位置（行号、函数名等）",
                                            "fixing_plan_in_detail": "详细的修复计划",
                                            "raw_code": "需要修改的原始代码片段",

                                            "dependencies": ["依赖的其他任务ID"],
                                        }
                                    ],
                                }
                                """

            # 构建包含完整文件内容的用户提示词
            user_prompt = self._build_deep_analysis_prompt(stdout, indexed_repo_data, file_contents, expected_behavior)
            # print('==================user_prompt====================')
            # print(user_prompt)
            # print('==================user_prompt====================')
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            print("🔄 正在调用LLM进行深度分析...")
            print(f"📊 输入大小: {len(user_prompt)} 字符")
            
            response = self.call_llm(messages, max_tokens=16384, temperature=0.3)
            
            print(f"✅ LLM调用完成，响应长度: {len(response)} 字符")
            # print('==================LLM的返回内容====================')
            # print(f"LLM的返回内容: {response}")
            # print('==================LLM的返回内容====================')
            print('\n\n')
            # 检查是否是错误响应
            if response.startswith("调用失败:") or response.startswith("LLM调用异常:"):
                print(f"❌ LLM调用失败: {response}")
                return self._generate_fallback_result(f"LLM调用失败: {response}")
            
            # 调试：打印LLM原始响应的开头和结尾
            # print(f"\n🔍 LLM响应预览:")
            # print("=" * 50)
            # print("开头:")
            # print(response[:500] + "..." if len(response) > 500 else response)
            # if len(response) > 1000:
            #     print("\n结尾:")
            #     print("..." + response[-500:])
            # print("=" * 50)
            
            # 解析JSON响应
            print("🔄 正在解析JSON响应...")
            result = parse_json_response(response)
            
            # 调试：打印解析结果
            print(f"\n📋 解析结果:")
            print(f"tasks数量: {len(result.get('tasks', []))}")
            if result.get('tasks'):
                for i, task in enumerate(result['tasks'][:3]):  # 只显示前3个任务
                    print(f"  任务{i+1}: {task.get('task_id', 'unknown')} - {task.get('fixing_type', 'unknown')}")
            elif result.get('partial_tasks'):
                partial_tasks = result.get('partial_tasks', [])
                print(f"⚠️ 使用恢复的部分任务: {len(partial_tasks)} 个")
                for i, task in enumerate(partial_tasks[:2]):
                    print(f"  恢复任务{i+1}: {task.get('task_id', 'recovered')} - {task.get('fixing_type', 'unknown')}")
            else:
                print("❌ 没有生成任何任务")
                if 'error' in result:
                    print(f"解析错误: {result['error']}")
                if 'raw_response' in result:
                    print(f"原始响应长度: {len(result['raw_response'])} 字符")
            print("=" * 50)
            
            # 验证和增强任务
            if "tasks" in result:
                result["tasks"] = validate_and_enhance_tasks(result["tasks"])
            elif "partial_tasks" in result:
                # 如果JSON解析失败但提取到了部分任务，使用这些任务
                print("🔄 使用从截断响应中恢复的部分任务")
                result["tasks"] = validate_and_enhance_tasks(result["partial_tasks"])
                # 标记这是一个部分恢复的结果
                result["partial_recovery"] = True
            else:
                # 如果完全没有任务，直接报告失败
                print("⚠️ 完全没有任务，LLM分析失败")
                result["tasks"] = []
            
            return result
            
        except Exception as e:
            return {
                "root_cause_analysis": f"深度分析失败: {str(e)}",
                "impact_analysis": "无法分析影响范围",
                "tasks": [],
                "error": str(e)
            }

    def _build_deep_analysis_prompt(self, stdout: str, indexed_repo_data: Dict[str, Any], 
                                   file_contents: Dict[str, Any], expected_behavior: str = None) -> str:
        """构建错误驱动的深度分析提示词"""
        
        expected_behavior_section = ""
        if expected_behavior:
            expected_behavior_section = f"""
        期望的程序行为: {expected_behavior}
        
        请基于期望行为来设计修复方案，确保修复后的程序能够按照期望行为运行。
        """
        
        prompt = f"""
        程序错误详情:{stdout}
        这个错误导致程序无法正常运行，需要立即修复！请立即分析上述错误并提供修复方案。
        {expected_behavior_section}

        项目完整索引信息
        ```json
        {json.dumps(indexed_repo_data, indent=2, ensure_ascii=False)}
        ```

        ## 相关文件分析

        """

        
        # 添加文件内容，但降低视觉权重
        for file_path, file_info in file_contents.items():
            if "content" in file_info:
                prompt += f"""### 📄 {file_path} ({file_info['lines']}行)
                            *分析原因*: {file_info['reason']}
                            *优先级*: {file_info['priority']}

                            ```python
                            {file_info['content']}
                            ```

                            """
            else:
                prompt += f"""### ❌ {file_path}
                                *错误*: {file_info.get('error', '未知错误')}
                                *原因*: {file_info['reason']}

                                """
        
        prompt += f"""
                    # 📋 TASK GENERATION REQUIREMENTS

                    基于上述错误分析和代码上下文，请严格按照以下要求生成修复任务：
                    1. 请严格按照以下JSON格式返回结果：
                    {{
                        "root_cause_analysis": "错误的根本原因分析",
                        "impact_analysis": "错误影响范围和传播路径分析",
                        "tasks": [
                            {{
                                "task_id": "任务唯一标识符",
                                "priority": 1,
                                "fixing_type": "Add_File 或 Change_File",
                                "which_file_to_fix": "具体文件路径",
                                "specific_location": "具体位置（行号、函数名等）",
                                "fixing_plan_in_detail": "详细的修复计划",
                                "raw_code": "需要修改的原始代码片段",

                                "dependencies": ["依赖的其他任务ID"],
                            }}
                        ],
                    }}
                    2. which_file_to_fix必须返回准确的文件地址，不要返回文件名，要返回文件的完整路径
                    3. fixing_plan_in_detail必须返回详细的修复计划，包括修改的代码片段，以及修改的原因
                    4. raw_code必须返回需要修改的原始代码片段
                    5. dependencies必须返回依赖的其他任务ID
                    6. 除了分析终端输出解决直接的问题以外，如果你看到代码还有其他的BUG时，你返回的task里面也应该包含修改其他BUG的task
                    请现在开始分析并生成符合要求的修复任务！"""

        return prompt



    def _summarize_call_graph(self, function_dependencies: Dict) -> Dict:
        """总结函数调用图信息"""
        call_graph = function_dependencies.get("call_graph", {})
        stats = function_dependencies.get("statistics", {})
        
        return {
            "total_functions": stats.get("total_functions", 0),
            "total_files": stats.get("total_files", 0),
            "has_call_relationships": bool(call_graph.get("nodes") or call_graph.get("edges"))
        }

    def _try_fix_truncated_json(self, json_str: str) -> str:
        """尝试修复被截断的JSON"""
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

    def _extract_partial_tasks(self, response: str) -> List[Dict]:
        """从响应中提取部分任务信息，即使JSON不完整"""
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

    def process(self, input_data: Any) -> Any:
        """
        处理输入数据（实现基类抽象方法）
        
        参数:
            input_data: 输入数据，包含stdout和repo_path
        
        返回:
            dict: 多任务分析结果
        """
        if isinstance(input_data, dict):
            stdout = input_data.get("stdout", "")
            repo_path = input_data.get("repo_path", "")
            indexed_repo = input_data.get("indexed_repo")
            return self.analyze_error(stdout, repo_path, indexed_repo)
        else:
            return self._generate_fallback_result("无效的输入数据格式")

    def _get_or_create_repo_index(self, repo_path: str) -> Dict[str, Any]:
        """
        获取或创建代码库索引 - 使用基础结构分析器
        
        参数:
            repo_path (str): 代码库路径
        
        返回:
            dict: 代码库索引数据，包含基础结构信息
        """
        try:
            # 尝试导入分析工具
            import sys
            sys.path.append(str(Path(__file__).parent.parent))
            
            from core_modules.simple_structure_analyzer import SimpleStructureAnalyzer
            
            # 基础结构分析
            structure_analyzer = SimpleStructureAnalyzer(repo_path)
            structure_result = structure_analyzer.analyze_project()
            
            # 构建结果
            combined_result = {
                # 来自SimpleStructureAnalyzer的基础信息
                "project_name": structure_result.get("project_name", ""),
                "project_path": structure_result.get("project_path", ""),
                "directory_structure": structure_result.get("directory_structure", []),
                "files": structure_result.get("files", {}),
                
                # 提供空的依赖信息以保持兼容性
                "function_dependencies": {
                    "call_graph": {},
                    "function_details": {},
                    "statistics": {},
                    "analysis_timestamp": ""
                },
                
                # 分析元信息
                "analysis_info": {
                    "structure_analyzer": "SimpleStructureAnalyzer",
                    "dependency_analyzer": "None",
                    "has_dependency_analysis": False,
                    "analyzer_timestamp": get_current_timestamp()
                }
            }
            
            return combined_result
            
        except Exception as e:
            # 如果分析工具不可用，紧急报错
            raise Exception(f"索引生成失败: {str(e)}")
    
    def _extract_file_from_error(self, stdout: str, repo_path: str) -> str:
        """从错误信息中提取文件路径"""
        
        # 寻找常见的文件路径模式
        patterns = [
            r'File "([^"]+)"',
            r"File '([^']+)'",
            r'in ([a-zA-Z_][a-zA-Z0-9_/\\]*\.py)',
            r'([a-zA-Z_][a-zA-Z0-9_/\\]*\.py)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, stdout)
            if matches:
                file_path = matches[0]
                # 转换为相对路径
                if repo_path in file_path:
                    return file_path.replace(repo_path, '').lstrip('/')
                return file_path
        
        return "" 

    # def _read_files_completely(self, files_to_read: List[Dict], repo_path: str) -> Dict[str, Any]:
    #     """
    #     阶段2: 完整读取所有标识的文件内容
        
    #     参数:
    #         files_to_read (list): 需要读取的文件列表
    #         repo_path (str): 代码库根路径
        
    #     返回:
    #         dict: 文件读取结果
    #     """
    #     file_contents = {}
    #     reading_summary = {
    #         "total_files": len(files_to_read),
    #         "successful_reads": 0,
    #         "failed_reads": 0,
    #         "filtered_out": 0,
    #         "total_lines": 0,
    #         "total_size": 0
    #     }
        
    #     repo_path = Path(repo_path).resolve()
        
    #     for file_info in files_to_read:
    #         file_path = file_info["file_path"]
            
    #         # 过滤不应该读取的文件
    #         if file_tools.should_filter_file(file_path):
    #             print(f"🚫 已过滤: {file_path} (不应该分析的文件)")
    #             reading_summary["filtered_out"] += 1
    #             continue
            
    #         full_path = repo_path / file_path
            
    #         try:
    #             # 检查文件是否存在
    #             if not full_path.exists():
    #                 file_contents[file_path] = {
    #                     "error": "文件不存在",
    #                     "reason": file_info.get("reason", ""),
    #                     "priority": file_info.get("priority", "medium")
    #                 }
    #                 reading_summary["failed_reads"] += 1
    #                 continue
                
    #             # 读取完整文件内容
    #             with open(full_path, 'r', encoding='utf-8') as f:
    #                 content = f.read()
                
    #             lines = content.count('\n') + 1
    #             size = len(content)
                
    #             file_contents[file_path] = {
    #                 "content": content,
    #                 "size": size,
    #                 "lines": lines,
    #                 "reason": file_info.get("reason", ""),
    #                 "priority": file_info.get("priority", "medium"),
    #                 "analysis_focus": file_info.get("analysis_focus", "整体代码逻辑"),
    #                 "encoding": "utf-8"
    #             }
                
    #             reading_summary["successful_reads"] += 1
    #             reading_summary["total_lines"] += lines
    #             reading_summary["total_size"] += size
                
    #             print(f"📄 已读取: {file_path} ({lines}行, {size}字符)")
                
    #         except Exception as e:
    #             file_contents[file_path] = {
    #                 "error": f"读取失败: {str(e)}",
    #                 "reason": file_info.get("reason", ""),
    #                 "priority": file_info.get("priority", "medium")
    #             }
    #             reading_summary["failed_reads"] += 1
    #             print(f"❌ 读取失败: {file_path} - {str(e)}")
        
    #     if reading_summary["filtered_out"] > 0:
    #         print(f"📋 读取总结: {reading_summary['successful_reads']} 成功, {reading_summary['failed_reads']} 失败, {reading_summary['filtered_out']} 过滤")
        
    #     return {
    #         "file_contents": file_contents,
    #         "reading_summary": reading_summary,
    #         "context_estimation": estimate_context_usage(file_contents)
    #     }

    # def _should_filter_file(self, file_path: str) -> bool:
    #     """
    #     判断是否应该过滤某个文件
        
    #     参数:
    #         file_path (str): 文件路径
        
    #     返回:
    #         bool: True表示应该过滤，False表示可以读取
    #     """
    #     # 规范化路径
    #     path = Path(file_path)
    #     path_str = str(path).lower()
        
    #     # 过滤规则
    #     filter_patterns = [
    #         # 调试输出目录
    #         "debug_output",
    #         "debug_report",
            
    #         # 备份文件
    #         ".backup",
    #         ".bak",
            
    #         # 隐藏文件和目录
    #         "/.",
            
    #         # 编译输出
    #         "__pycache__",
    #         ".pyc",
    #         ".pyo",
    #         ".pyd",
            
    #         # 版本控制
    #         ".git",
    #         ".svn",
    #         ".hg",
            
    #         # IDE/编辑器文件
    #         ".vscode",
    #         ".idea",
    #         ".vs",
            
    #         # 虚拟环境
    #         "/venv",
    #         "/env",
    #         "/virtualenv",
            
    #         # 依赖目录
    #         "node_modules",
            
    #         # 日志文件
    #         ".log",
    #         "logs/",
            
    #         # 临时文件
    #         ".tmp",
    #         ".temp",
    #         "~$",
            
    #         # 系统文件
    #         ".ds_store",
    #         "thumbs.db",
            
    #         # 修改历史文件
    #         "modification_history",
            
    #         # 测试覆盖率
    #         ".coverage",
    #         "htmlcov",
            
    #         # 打包文件
    #         ".egg-info",
    #         "dist/",
    #         "build/"
    #     ]
        
    #     # 检查是否匹配任何过滤模式
    #     for pattern in filter_patterns:
    #         if pattern in path_str:
    #             return True
        
    #     # 检查文件扩展名（只读取代码文件）
    #     allowed_extensions = {
    #         '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
    #         '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt',
    #         '.html', '.css', '.scss', '.vue', '.jsx', '.tsx',
    #         '.json', '.yaml', '.yml', '.xml', '.toml', '.ini',
    #         '.sql', '.sh', '.bat', '.ps1', '.md', '.txt', '.cfg'
    #     }
        
    #     if path.suffix and path.suffix.lower() not in allowed_extensions:
    #         return True
        
    #     return False




    # def _validate_and_enhance_tasks(self, tasks: List[Dict]) -> List[Dict]:
    #     """验证和增强任务列表"""
    #     enhanced_tasks = []
        
    #     for i, task in enumerate(tasks):
    #         # 确保必要字段存在
    #         enhanced_task = {
    #             "task_id": task.get("task_id", f"task_{i+1}"),
    #             "priority": task.get("priority", i+1),
    #             "fixing_type": task.get("fixing_type", "Change_File"),
    #             "which_file_to_fix": task.get("which_file_to_fix", ""),
    #             "specific_location": task.get("specific_location", ""),
    #             "fixing_plan_in_detail": task.get("fixing_plan_in_detail", ""),
    #             "raw_code": task.get("raw_code", ""),
    #             "new_code": task.get("new_code", ""),
    #             "dependencies": task.get("dependencies", []),
    #             "estimated_impact": task.get("estimated_impact", ""),
    #             "risk_level": task.get("risk_level", "medium"),
    #             "verification_method": task.get("verification_method", "运行程序验证")
    #         }
            
    #         enhanced_tasks.append(enhanced_task)
        
    #     return enhanced_tasks

    # def _generate_execution_plan(self, tasks: List[Dict]) -> Dict[str, Any]:
    #     """生成任务执行计划"""
    #     if not tasks:
    #         return {
    #             "total_tasks": 0,
    #             "execution_order": [],
    #             "risk_assessment": "无任务需要执行"
    #         }
        
    #     # 简单的拓扑排序，按优先级和依赖关系排序
    #     execution_order = []
    #     remaining_tasks = {task["task_id"]: task for task in tasks}
        
    #     while remaining_tasks:
    #         # 找到没有未满足依赖的任务
    #         ready_tasks = []
    #         for task_id, task in remaining_tasks.items():
    #             dependencies = task.get("dependencies", [])
    #             if all(dep_id not in remaining_tasks for dep_id in dependencies):
    #                 ready_tasks.append((task_id, task.get("priority", 999)))
            
    #         if not ready_tasks:
    #             # 存在循环依赖，按优先级强制执行
    #             task_id = min(remaining_tasks.keys(), 
    #                         key=lambda tid: remaining_tasks[tid].get("priority", 999))
    #             ready_tasks = [(task_id, remaining_tasks[task_id].get("priority", 999))]
            
    #         # 按优先级排序并添加到执行顺序
    #         ready_tasks.sort(key=lambda x: x[1])
    #         for task_id, _ in ready_tasks:
    #             execution_order.append(task_id)
    #             del remaining_tasks[task_id]
        
    #     # 风险评估
    #     high_risk_tasks = [t for t in tasks if t.get("risk_level") == "high"]
    #     risk_assessment = f"总共{len(tasks)}个任务，{len(high_risk_tasks)}个高风险任务"
        
    #     return {
    #         "total_tasks": len(tasks),
    #         "execution_order": execution_order,
    #         "risk_assessment": risk_assessment
    #     }

    # def _generate_fallback_result(self, error_message: str) -> Dict[str, Any]:
    #     """生成备用结果"""
    #     return {
    #         "analysis_stages": {
    #             "file_identification": {"error": error_message},
    #             "file_reading": {"error": "未执行"},
    #             "deep_analysis": {"error": "未执行"}
    #         },
    #         "tasks": [{
    #             "task_id": "fallback_task",
    #             "priority": 1,
    #             "fixing_type": "Change_File",
    #             "which_file_to_fix": "",
    #             "fixing_plan_in_detail": f"分析失败: {error_message}",
    #             "raw_code": "",
    #             "dependencies": [],
    #             "estimated_impact": "无法评估"
    #         }],
    #         "execution_plan": {
    #             "total_tasks": 1,
    #             "execution_order": ["fallback_task"],
    #             "risk_assessment": "分析失败，无法评估风险"
    #         }
    #     }

    # def _estimate_context_usage(self, file_contents: Dict) -> Dict:
    #     """估算上下文使用量"""
    #     total_chars = sum(
    #         info.get("size", 0) for info in file_contents.values() 
    #         if "content" in info
    #     )
    #     total_lines = sum(
    #         info.get("lines", 0) for info in file_contents.values() 
    #         if "content" in info
    #     )
        
    #     # 粗略估算token数（假设4个字符=1个token）
    #     estimated_tokens = total_chars // 4
        
    #     return {
    #         "total_characters": total_chars,
    #         "total_lines": total_lines,
    #         "estimated_tokens": estimated_tokens,
    #         "context_status": "acceptable" if estimated_tokens < 80000 else "large"
    #     }

    # def _parse_json_response(self, response: str) -> Dict:
    #     """解析LLM的JSON响应 - 增强版本，更robust地处理各种格式问题"""
    #     try:
    #         # 1. 首先尝试直接解析
    #         try:
    #             return json.loads(response.strip())
    #         except json.JSONDecodeError:
    #             pass
            
    #         # 2. 尝试提取JSON代码块
    #         if "```json" in response:
    #             json_start = response.find("```json") + 7
    #             json_end = response.find("```", json_start)
    #             if json_end == -1:
    #                 # 没有结束标记，可能被截断
    #                 json_str = response[json_start:].strip()
    #             else:
    #                 json_str = response[json_start:json_end].strip()
    #         elif "{" in response and "}" in response:
    #             # 3. 寻找第一个{到最后一个}
    #             json_start = response.find("{")
    #             json_end = response.rfind("}") + 1
    #             json_str = response[json_start:json_end]
    #         else:
    #             json_str = response.strip()
            
    #         # 4. 尝试解析提取的JSON
    #         try:
    #             return json.loads(json_str)
    #         except json.JSONDecodeError as e:
    #             # 5. 如果解析失败，尝试修复常见问题
    #             print(f"⚠️ JSON解析失败，尝试修复: {str(e)}")
                
    #             # 尝试修复截断的JSON
    #             fixed_json = self._try_fix_truncated_json(json_str)
    #             if fixed_json:
    #                 try:
    #                     return json.loads(fixed_json)
    #                 except json.JSONDecodeError:
    #                     pass
                
    #             # 如果所有尝试都失败，返回错误信息但包含部分数据
    #             return {
    #                 "error": f"JSON解析失败: {str(e)}",
    #                 "raw_response": response,
    #                 "partial_tasks": self._extract_partial_tasks(response)
    #             }

    #     except Exception as e:
    #         return {
    #             "error": f"响应处理异常: {str(e)}",
    #             "raw_response": response
    #         }

        # def _get_current_timestamp(self) -> str:
    #     """获取当前时间戳"""
    #     from datetime import datetime
    #     return datetime.now().isoformat()

    # def _get_basic_file_list(self, repo_path: str) -> list:
    #     """
    #     获取基本的文件列表
        
    #     参数:
    #         repo_path (str): 代码库路径
        
    #     返回:
    #         list: 文件列表
    #     """
    #     try:
    #         files = []
    #         for root, dirs, filenames in os.walk(repo_path):
    #             # 排除隐藏目录
    #             dirs[:] = [d for d in dirs if not d.startswith('.')]
                
    #             level = root.replace(repo_path, '').count(os.sep)
    #             indent = '  ' * level
    #             rel_root = os.path.relpath(root, repo_path)
                
    #             if rel_root != '.':
    #                 files.append(f"{indent}{os.path.basename(root)}/")
                
    #             sub_indent = '  ' * (level + 1)
    #             for filename in filenames:
    #                 if not filename.startswith('.'):
    #                     files.append(f"{sub_indent}{filename}")
            
    #         return files
            
    #     except Exception as e:
    #         return [f"错误: 无法列出文件 - {str(e)}"]

'''
===========================================================
系统提示词备份：    def _deep_analysis_with_content
===========================================================

            system_prompt = """🎯 你是专业的代码错误修复专家。你的首要且唯一目标是：让程序能够成功运行。

                                ## 错误驱动的分析原则

                                **优先级策略**：直接修复stdout错误 > 防御性编程 > 系统优化 > 测试增强

                                **错误类型处理策略**：
                                🔥 临界错误（程序无法启动）- 立即修复
                                ├── ImportError/ModuleNotFoundError → 创建缺失模块或修复导入路径
                                ├── SyntaxError → 修复语法错误
                                ├── IndentationError → 修复缩进问题
                                └── FileNotFoundError → 创建缺失文件或修复路径

                                ⚠️ 运行时错误（程序崩溃）- 优先处理
                                ├── AttributeError → 修复属性访问错误
                                ├── NameError → 修复未定义变量
                                ├── TypeError → 修复类型错误
                                └── ValueError → 修复值错误

                                🐛 逻辑错误（结果不正确）- 后续优化
                                ├── AssertionError → 修复断言逻辑
                                ├── IndexError → 修复索引访问
                                └── 业务逻辑错误 → 改进算法逻辑

                                ## 强制任务生成规则

                                **CRITICAL要求**：
                                1. 第一个任务必须：priority=1，直接解决stdout中显示的具体错误
                                2. 第一个任务必须：提供能让程序立即运行的最小修复方案
                                3. 后续任务：priority=2+，依赖于核心修复任务的完成
                                4. 每个任务：必须说明具体的验证方法

                                **任务分层策略**：
                                - Priority 1: 直接修复stderr/stdout中的错误，让程序能运行
                                - Priority 2: 验证修复效果，确保程序稳定
                                - Priority 3: 防御性编程，添加错误处理
                                - Priority 4+: 代码优化、重构、测试增强

                                请严格按照以下JSON格式返回结果：
                                {
                                    "root_cause_analysis": "错误的根本原因分析",
                                    "impact_analysis": "错误影响范围和传播路径分析",
                                    "tasks": [
                                        {
                                            "task_id": "任务唯一标识符",
                                            "priority": 1,
                                            "fixing_type": "Add_File 或 Change_File",
                                            "which_file_to_fix": "具体文件路径",
                                            "specific_location": "具体位置（行号、函数名等）",
                                            "fixing_plan_in_detail": "详细的修复计划",
                                            "raw_code": "需要修改的原始代码片段",
                                            "new_code": "建议的新代码（如果适用）",
                                            "dependencies": ["依赖的其他任务ID"],
                                            "estimated_impact": "修复的预期影响",
                                            "risk_level": "low/medium/high",
                                            "verification_method": "如何验证修复是否成功"
                                        }
                                    ],
                                    "prevention_recommendations": "防止类似问题的建议",
                                    "testing_recommendations": "建议添加的测试"
                                }
                                ## 分析执行要求
                                1. **错误聚焦**：立即识别stdout错误类型和严重程度
                                2. **最小修复**：优先提供能让程序运行的最小改动
                                3. **渐进改进**：在程序能运行的基础上再考虑优化
                                4. **验证导向**：每个修复都要可验证和可测试"""

===========================================================
系统提示词备份：    def _deep_analysis_with_content
===========================================================
'''


if __name__ == "__main__":
    #测试_get_or_create_repo_index功能，打印详细信息，并优雅展示JSON内容使其在终端的可读性强
    # project_path = "/Users/wwchdemac/python_projects/debug_agent/test_input/webpage"
    # analyzer = AnalyzerAgent(project_path = project_path)
    # result = analyzer._get_or_create_repo_index(project_path)
    # print(json.dumps(result, indent=4, ensure_ascii=False))
    # #最后统计终端打印的字符数量，要仔细统计JSON里面的所有字符
    # print(f"终端打印的字符数量: {len(json.dumps(result, indent=4, ensure_ascii=False))}")
    #测试_build_deep_analysis_prompt

    # 测试 _build_deep_analysis_prompt 方法的正确方式
    project_path = "/Users/wwchdemac/python_projects/debug_agent/test_input/webpage"
    expected_behavior = "程序能够正常运行，并且能够正确显示网页"
    # 1. 创建分析器实例
    analyzer = AnalyzerAgent(project_path=project_path)

    # 2. 准备模拟的错误输出（stdout参数）
    stdout = """
    Traceback (most recent call last):
    File "/Users/wwchdemac/python_projects/debug_agent/test_input/webpage/backend/__init__.py", line 10, in <module>
        from .config import Config
    ImportError: attempted relative import with no known parent package
    """

    # 3. 获取项目索引数据
    indexed_repo_data = analyzer._get_or_create_repo_index(project_path)

    # 4. 准备文件内容数据 (这里需要模拟或者通过正确的方式获取)
    # 方式1: 手动构建文件内容字典
    # file_contents = {
    #     "backend/__init__.py": {
    #         "content": "文件内容...",
    #         "lines": 24,
    #         "size": 625,
    #         "encoding": "utf-8"
    #     },
    #     "backend/config.py": {
    #         "content": "文件内容...", 
    #         "lines": 70,
    #         "size": 2200,
    #         "encoding": "utf-8"
    #     }
    # }

    # 方式2: 或者通过完整的分析流程获取
    from agents.utils.file_tools import read_files_completely

    print('==================analyzer._identify_relevant_files====================')
    # 首先识别相关文件
    file_identification_result = analyzer._identify_relevant_files(stdout, indexed_repo_data, expected_behavior)

    print('\n\n')
    files_to_read = file_identification_result.get("files_to_read", [])
    
    print('==================files_to_read====================')
    # 然后读取文件内容
    file_reading_result = read_files_completely(files_to_read, project_path)
    file_contents = file_reading_result.get("file_contents", {})
    print('==================file_reading_result====================')
    # 5. 现在可以正确调用方法了
    result = analyzer._build_deep_analysis_prompt(stdout, indexed_repo_data, file_contents, expected_behavior)
    #我还希望看看将提示词发送之后LLM的返回内容，_deep_analysis_with_content
    result = analyzer._deep_analysis_with_content(stdout, indexed_repo_data, file_contents, expected_behavior)
    print('==================result====================')
    #优雅打印JSON
    print(json.dumps(result, indent=4, ensure_ascii=False))
    print('==================result====================')


