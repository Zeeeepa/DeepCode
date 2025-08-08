"""
判断Agent

判断程序输出是否正确，如果错误则生成解释原因。
能够读取修改历史，区分程序错误和有意的优雅错误处理。

主要功能:
- judge_output(): 判断程序输出是否正确
- _read_modification_history(): 读取修改历史
- _analyze_error_evolution(): 分析错误演化过程
"""

import json
import os
import re
from typing import Dict, Any, List
from .base_agent import BaseAgent
from .utils import (
    analysis_tools,
    code_tools,
    analyze_output_patterns,
    analyze_error_evolution,
    read_modification_history
)


class JudgerAgent(BaseAgent):
    """
    判断Agent
    
    像人类一样理解程序输出内容，判断是否正确。
    能够读取修改历史，区分程序错误和有意的优雅错误处理。
    """
    
    def __init__(self, **kwargs):
        """
        初始化判断Agent
        
        参数:
            **kwargs: 配置参数
        """
        super().__init__(**kwargs)
        
        # 异常模式检测
        self.exception_patterns = [
            r'错误[:：]\s*.*异常[:：]',
            r'Exception[:：]',
            r'Error[:：]',
            r'异常[:：]',
            r'错误[:：]',
            r'Traceback \(most recent call last\)',
            r'File ".*", line \d+',
            r'\w+Error:',
            r'\w+Exception:',
            r'发生异常',
            r'出现错误',
        ]
        
        # 严重错误模式（未处理的程序错误）
        self.critical_patterns = [
            r'Traceback',
            r'Fatal',
            r'Critical',
            r'程序崩溃',
            r'无法继续',
            r'系统错误',
            r'Segmentation fault',
            r'Core dumped'
        ]
        
        # 优雅错误处理模式（有意的业务错误处理）
        self.graceful_error_patterns = [
            r'错误[:：].*除数不能为零',
            r'.*÷.*=.*错误[:：]',
            r'.*除法.*错误[:：]',
            r'错误[:：].*时发生.*错误[:：]',
            r'警告[:：]',
            r'提示[:：]',
            r'注意[:：]'
        ]

    def judge_output(self, stdout: str, expected_behavior: str = None, output_dir: str = None, iteration: int = 1) -> Dict[str, Any]:
        """
        判断程序输出是否正确
        
        参数:
            stdout (str): 程序的标准输出
            expected_behavior (str, optional): 期望的行为描述
            output_dir (str, optional): 输出目录，用于读取修改历史
            iteration (int): 当前迭代次数
        
        返回:
            dict: JSON格式的判断结果
            {
                "is_correct": bool,
                "reason": str,
                "error_type": str,
                "trigger_analyzer": bool,
                "quality_issues": list,
                "severity": str,
                "error_category": str  # 新增：程序错误/业务错误/改进中
            }
        """
        if not stdout:
            return {
                "is_correct": False,
                "reason": "程序没有任何输出",
                "error_type": "no_output",
                "trigger_analyzer": True,
                "quality_issues": ["程序无输出"],
                "severity": "high",
                "error_category": "程序错误"
            }
        
        try:
            # 预分析：检测输出模式
            analysis_result = analyze_output_patterns(stdout)
            
            # 读取修改历史（如果可用）
            modification_history = None
            if output_dir:
                modification_history = read_modification_history(output_dir)
            
            # 分析错误演化过程
            error_evolution = analyze_error_evolution(stdout, modification_history, iteration)
            
            # 构建增强的智能判断提示词
            system_prompt = """你是一个专业的程序输出质量评估专家。你需要综合分析程序输出、修改历史和错误演化，进行智能判断。

                                判断标准（分层评估）：
                                1. 程序错误：未处理的异常、程序崩溃、系统错误 → 必须修复
                                2. 改进中错误：正在优化的错误处理，但仍需完善 → 继续改进
                                3. 业务错误：有意的、优雅的错误处理输出 → 可以接受

                                错误分类规则：
                                - 程序错误：Traceback、未捕获异常、程序崩溃
                                - 改进中错误：已捕获但处理不够优雅的错误（如第1-2次修复）
                                - 业务错误：经过多次优化后的优雅错误处理（如第3次及以后）

                                严格按照以下JSON格式返回结果：
                                {
                                    "is_correct": true/false,
                                    "reason": "详细说明判断原因",
                                    "error_type": "错误类型：logic_error/exception_handling/runtime_error/output_quality/no_output/graceful_handling",
                                    "trigger_analyzer": true/false,
                                    "quality_issues": ["具体的质量问题列表"],
                                    "severity": "严重程度：low/medium/high/critical",
                                    "error_category": "错误分类：程序错误/改进中错误/业务错误"
                                }

                                智能判断策略：
                                - 如果是第1次迭代且有异常 → 程序错误，需要修复
                                - 如果是第2-3次迭代，错误处理在优化 → 改进中错误，继续优化
                                - 如果是第4次及以后，错误处理已经优雅 → 业务错误，可接受
                                - 如果修改历史显示错误处理逐步改进 → 根据改进程度判断"""

            user_prompt = f"""程序输出：
                                ```
                                {stdout}
                                ```

                                当前迭代：第{iteration}次

                                预分析结果：
                                - 检测到异常模式: {analysis_result['has_exceptions']}
                                - 检测到的异常: {analysis_result['exception_details']}
                                - 检测到优雅错误处理: {analysis_result['has_graceful_errors']}
                                - 严重错误: {analysis_result['has_critical_errors']}
                                - 输出行数: {len(stdout.splitlines())}

                                错误演化分析：
                                {error_evolution}"""

            if expected_behavior:
                user_prompt += f"""期望行为：{expected_behavior}"""

            user_prompt += """
                            请根据智能判断策略进行评估，特别注意：
                            1. 区分程序错误、改进中错误和业务错误
                            2. 考虑修改历史和错误演化过程
                            3. 评估错误处理的优雅程度
                            4. 判断是否需要继续优化

                            按照指定的JSON格式返回判断结果。"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.call_llm(messages, max_tokens=16384, temperature=0.1)
            
            # 尝试解析JSON响应
            try:
                # 提取JSON部分
                if "```json" in response:
                    json_start = response.find("```json") + 7
                    json_end = response.find("```", json_start)
                    json_str = response[json_start:json_end].strip()
                elif "{" in response and "}" in response:
                    json_start = response.find("{")
                    json_end = response.rfind("}") + 1
                    json_str = response[json_start:json_end]
                else:
                    json_str = response
                
                result = json.loads(json_str)
                
                # 验证和补全必要字段
                required_fields = {
                    "is_correct": False,
                    "reason": "unknown",
                    "error_type": "unknown",
                    "trigger_analyzer": True,
                    "quality_issues": [],
                    "severity": "medium",
                    "error_category": "程序错误"
                }
                
                for field, default in required_fields.items():
                    if field not in result:
                        result[field] = default
                
                # 智能纠正：基于预分析和演化分析进行最终判断
                result = self._apply_intelligent_correction(result, analysis_result, error_evolution, iteration)
                
                return result
                
            except json.JSONDecodeError:
                # JSON解析失败，返回基于预分析的智能判断
                return self._fallback_intelligent_judgment(response, analysis_result, error_evolution, iteration)
                
        except Exception as e:
            return {
                "is_correct": False,
                "reason": f"判断过程出现异常: {str(e)}",
                "error_type": "system_error",
                "trigger_analyzer": True,
                "quality_issues": [f"系统异常: {str(e)}"],
                "severity": "high",
                "error_category": "程序错误"
            }

    def _apply_intelligent_correction(self, result: Dict[str, Any], analysis_result: Dict[str, Any], 
                                    error_evolution: str, iteration: int) -> Dict[str, Any]:
        """
        应用智能纠正逻辑
        
        参数:
            result (Dict[str, Any]): LLM判断结果
            analysis_result (Dict[str, Any]): 预分析结果
            error_evolution (str): 错误演化分析
            iteration (int): 当前迭代次数
        
        返回:
            Dict[str, Any]: 纠正后的结果
        """
        # 严重错误必须修复
        if analysis_result['has_critical_errors']:
            result['is_correct'] = False
            result['error_category'] = '程序错误'
            result['severity'] = 'critical'
            result['trigger_analyzer'] = True
            return result
        
        # 基于迭代次数和优雅错误检测进行智能判断
        if analysis_result['has_graceful_errors'] and iteration >= 3:
            # 第3次及以后的迭代，如果是优雅错误处理，可能是业务错误
            if '多次优化' in error_evolution or '趋向优雅' in error_evolution:
                result['is_correct'] = True
                result['error_category'] = '业务错误'
                result['error_type'] = 'graceful_handling'
                result['trigger_analyzer'] = False
                result['severity'] = 'low'
                result['reason'] = f"经过{iteration}次迭代优化，错误处理已经优雅，属于有意的业务错误处理"
        
        elif analysis_result['has_exceptions'] and iteration >= 2:
            # 第2-3次迭代，错误处理在改进中
            if '改进' in error_evolution:
                result['error_category'] = '改进中错误'
                result['trigger_analyzer'] = True
                result['severity'] = 'medium'
        
        elif analysis_result['has_exceptions'] and iteration == 1:
            # 第1次迭代有异常，肯定是程序错误
            result['is_correct'] = False
            result['error_category'] = '程序错误'
            result['trigger_analyzer'] = True
        
        return result

    def _fallback_intelligent_judgment(self, response: str, analysis_result: Dict[str, Any], 
                                     error_evolution: str, iteration: int) -> Dict[str, Any]:
        """
        LLM响应解析失败时的智能回退判断
        
        参数:
            response (str): LLM原始响应
            analysis_result (Dict[str, Any]): 预分析结果
            error_evolution (str): 错误演化分析
            iteration (int): 当前迭代次数
        
        返回:
            Dict[str, Any]: 智能判断结果
        """
        # 基于规则的智能判断
        is_correct = False
        error_category = "程序错误"
        trigger_analyzer = True
        severity = "medium"
        error_type = "parse_error"
        
        if analysis_result['has_critical_errors']:
            is_correct = False
            error_category = "程序错误"
            severity = "critical"
        elif analysis_result['has_graceful_errors'] and iteration >= 3 and '优雅' in error_evolution:
            is_correct = True
            error_category = "业务错误"
            trigger_analyzer = False
            severity = "low"
            error_type = "graceful_handling"
        elif analysis_result['has_exceptions'] and iteration >= 2:
            error_category = "改进中错误"
            severity = "medium"
        
        return {
            "is_correct": is_correct,
            "reason": f"LLM解析失败，基于智能规则判断: {response[:200]}",
            "error_type": error_type,
            "trigger_analyzer": trigger_analyzer,
            "quality_issues": analysis_result['exception_details'] if analysis_result['has_exceptions'] else [],
            "severity": severity,
            "error_category": error_category
        }

    def process(self, input_data: Any) -> Any:
        """
        处理输入数据（实现基类抽象方法）
        
        参数:
            input_data: 输入数据，可以是字符串或字典
        
        返回:
            dict: 判断结果
        """
        if isinstance(input_data, str):
            return self.judge_output(input_data)
        elif isinstance(input_data, dict):
            stdout = input_data.get("stdout", "")
            expected = input_data.get("expected_behavior")
            output_dir = input_data.get("output_dir")
            iteration = input_data.get("iteration", 1)
            return self.judge_output(stdout, expected, output_dir, iteration)
        else:
            return {
                "is_correct": False,
                "reason": "无效的输入数据格式",
                "error_type": "input_error",
                "trigger_analyzer": False,
                "quality_issues": ["输入格式错误"],
                "severity": "low",
                "error_category": "程序错误"
            } 



    # def _analyze_output_patterns(self, stdout: str) -> Dict[str, Any]:
    #     """
    #     预分析输出模式，检测异常和错误
        
    #     参数:
    #         stdout (str): 程序输出
        
    #     返回:
    #         dict: 分析结果
    #     """
    #     result = {
    #         'has_exceptions': False,
    #         'has_critical_errors': False,
    #         'has_graceful_errors': False,
    #         'exception_details': [],
    #         'critical_details': [],
    #         'graceful_details': []
    #     }
        
    #     lines = stdout.splitlines()
        
    #     for line in lines:
    #         # 检测异常模式
    #         for pattern in self.exception_patterns:
    #             if re.search(pattern, line, re.IGNORECASE):
    #                 result['has_exceptions'] = True
    #                 result['exception_details'].append(line.strip())
    #                 break
            
    #         # 检测严重错误模式
    #         for pattern in self.critical_patterns:
    #             if re.search(pattern, line, re.IGNORECASE):
    #                 result['has_critical_errors'] = True
    #                 result['critical_details'].append(line.strip())
    #                 break
            
    #         # 检测优雅错误处理模式
    #         for pattern in self.graceful_error_patterns:
    #             if re.search(pattern, line, re.IGNORECASE):
    #                 result['has_graceful_errors'] = True
    #                 result['graceful_details'].append(line.strip())
    #                 break
        
    #     # 去重
    #     result['exception_details'] = list(set(result['exception_details']))
    #     result['critical_details'] = list(set(result['critical_details']))
    #     result['graceful_details'] = list(set(result['graceful_details']))
        
    #     return result

    # def _read_modification_history(self, output_dir: str) -> Dict[str, Any]:
    #     """
    #     读取修改历史
        
    #     参数:
    #         output_dir (str): 输出目录
        
    #     返回:
    #         dict: 修改历史，如果读取失败则返回None
    #     """
    #     try:
    #         history_file = os.path.join(output_dir, "modification_history.json")
    #         if os.path.exists(history_file):
    #             with open(history_file, 'r', encoding='utf-8') as f:
    #                 return json.load(f)
    #     except Exception:
    #         pass
    #     return None

    # def _analyze_error_evolution(self, current_stdout: str, modification_history: Dict[str, Any], iteration: int) -> str:
    #     """
    #     分析错误演化过程
        
    #     参数:
    #         current_stdout (str): 当前输出
    #         modification_history (Dict[str, Any]): 修改历史
    #         iteration (int): 当前迭代次数
        
    #     返回:
    #         str: 错误演化分析结果
    #     """
    #     if not modification_history:
    #         return f"第{iteration}次迭代，无修改历史可供分析"
        
    #     modifications = modification_history.get('modifications', [])
    #     total_iterations = modification_history.get('total_iterations', 0)
        
    #     if not modifications:
    #         return f"第{iteration}次迭代，修改历史为空"
        
    #     # 分析修改进展
    #     evolution_analysis = []
        
    #     # 统计异常处理相关的改进
    #     exception_improvements = 0
    #     error_handling_improvements = 0
        
    #     for mod in modifications:
    #         summary = mod.get('modification_summary', '').lower()
    #         changes = mod.get('changes_made', [])
            
    #         if any('异常' in change or 'exception' in change.lower() for change in changes):
    #             exception_improvements += 1
    #         if any('错误处理' in change or 'error' in change.lower() for change in changes):
    #             error_handling_improvements += 1
        
    #     evolution_analysis.append(f"总修改次数: {total_iterations}")
    #     evolution_analysis.append(f"异常处理改进次数: {exception_improvements}")
    #     evolution_analysis.append(f"错误处理改进次数: {error_handling_improvements}")
        
    #     # 判断改进趋势
    #     if exception_improvements >= 2:
    #         evolution_analysis.append("错误处理已经过多次优化，趋向优雅")
    #     elif exception_improvements >= 1:
    #         evolution_analysis.append("错误处理正在改进中")
    #     else:
    #         evolution_analysis.append("错误处理改进较少")
        
    #     # 分析最近的修改
    #     if modifications:
    #         latest_mod = modifications[-1]
    #         latest_summary = latest_mod.get('modification_summary', '')
    #         evolution_analysis.append(f"最新修改: {latest_summary}")
        
    #     return "\n".join(evolution_analysis)
