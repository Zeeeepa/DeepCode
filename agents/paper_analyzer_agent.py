"""
论文分析Agent

继承自AnalyzerAgent，专门用于论文复现的分析需求。
针对论文复现的5个层级（L0-L4）提供专门的分析逻辑和提示词。

主要功能:
- PaperAnalyzerAgent: 专门的论文分析Agent
- analyze_paper_level: 针对特定层级的分析方法
- 每个层级有专门的分析策略和提示词
- 完全不影响原有的AnalyzerAgent功能
"""

import json
from typing import Dict, Any, List
from .analyzer_agent import AnalyzerAgent
from .utils import (
    parse_json_response, 
    get_current_timestamp,
    create_repo_index,
    generate_execution_plan,
    get_colored_logger,
    log_detailed,
    log_llm_call,
    log_operation_start,
    log_operation_success,
    log_operation_error,
    load_paper_guide
)


class PaperAnalyzerAgent(AnalyzerAgent):
    """
    论文分析Agent
    
    继承自AnalyzerAgent，专门用于论文复现的分析。
    针对5个复现层级提供专门的分析逻辑：
    - L0: 环境搭建分析
    - L1: 核心算法分析  
    - L2: 训练流程分析
    - L3: 实验复现分析
    - L4: 结果对齐分析
    """
    
    def __init__(self, **kwargs):
        """
        初始化论文分析Agent
        
        参数:
            **kwargs: 传递给父类AnalyzerAgent的参数
        """
        super().__init__(**kwargs)
        
        # 初始化专门的彩色日志记录器
        self.paper_logger = get_colored_logger("PaperAnalyzer")
        self.paper_logger.info("🎯 PaperAnalyzerAgent 初始化完成")
        
        # 论文复现层级的分析策略
        self.level_strategies = {
            "L0": {
                "name": "环境搭建",
                "focus": ["依赖安装", "环境配置", "数据准备", "基础运行能力"],
                "max_tasks": 5,
                "priority_keywords": ["import", "requirement", "dependency", "environment", "setup", "install"]
            },
            "L1": {
                "name": "核心算法", 
                "focus": ["算法实现", "模型架构", "关键公式", "核心组件"],
                "max_tasks": 6,
                "priority_keywords": ["algorithm", "model", "network", "layer", "function", "class", "formula"]
            },
            "L2": {
                "name": "训练流程",
                "focus": ["训练循环", "损失计算", "优化器", "数据流水线", "推理流程"],
                "max_tasks": 5,
                "priority_keywords": ["train", "loss", "optimizer", "forward", "backward", "epoch", "batch"]
            },
            "L3": {
                "name": "实验复现",
                "focus": ["实验脚本", "评估指标", "数据集处理", "结果输出"],
                "max_tasks": 4,
                "priority_keywords": ["experiment", "eval", "test", "metric", "dataset", "benchmark"]
            },
            "L4": {
                "name": "结果对齐", 
                "focus": ["结果优化", "参数调整", "性能改进", "数值一致性"],
                "max_tasks": 4,
                "priority_keywords": ["performance", "accuracy", "result", "output", "metric", "score"]
            }
        }
    
    def analyze_paper_level(self, 
                           level_code: str,
                           level_description: str,
                           paper_guide: str,
                           repo_path: str,
                           additional_content: str = "",
                           target_metrics: Dict[str, Any] = None,
                           iteration: int = 1) -> Dict[str, Any]:
        """
        分析论文复现的特定层级
        
        参数:
            level_code (str): 层级代码 (L0, L1, L2, L3, L4)
            level_description (str): 层级描述
            paper_guide (str): 论文复现指南
            repo_path (str): 代码库路径
            additional_content (str): 补充信息内容（可选）
            target_metrics (dict): 目标指标
            iteration (int): 当前迭代次数
        
        返回:
            dict: 层级分析结果
        """
        try:
            # 开始层级分析
            log_operation_start(self.paper_logger, f"{level_code}层级分析")
            self.paper_logger.info(f"🔍 开始{level_code}层级分析...")
            
            # 获取层级策略
            strategy = self.level_strategies.get(level_code, {})
            level_name = strategy.get("name", level_code)
            focus_areas = strategy.get("focus", [])
            max_tasks = strategy.get("max_tasks", 5)
            
            # 记录分析配置
            analysis_config = {
                "层级代码": level_code,
                "层级名称": level_name,
                "专注领域": ", ".join(focus_areas),
                "最大任务数": max_tasks,
                "迭代次数": iteration
            }
            log_detailed(self.paper_logger, "📋 分析配置", analysis_config)
            
            # 处理paper_guide参数（支持markdown文件路径或直接内容）
            log_operation_start(self.paper_logger, "Paper Guide处理")
            processed_paper_guide = load_paper_guide(paper_guide)
            if not processed_paper_guide:
                self.paper_logger.warning("⚠️ Paper Guide为空，将使用空指南进行分析")
            else:
                guide_info = {
                    "内容长度": f"{len(processed_paper_guide)} 字符",
                    "行数": len(processed_paper_guide.splitlines()),
                    "是否从文件加载": paper_guide.strip().endswith(('.md', '.markdown'))
                }
                log_detailed(self.paper_logger, "📄 Paper Guide信息", guide_info)
            log_operation_success(self.paper_logger, "Paper Guide处理")
            
            # 创建或获取代码库索引
            log_operation_start(self.paper_logger, "代码库索引创建/获取")
            
            indexed_repo_data = self._get_or_create_repo_index(repo_path)
            if not indexed_repo_data:
                log_operation_error(self.paper_logger, "代码库索引创建", "索引数据为空")
                return self._generate_fallback_result(level_code, "无法创建代码库索引")
            
            log_operation_success(self.paper_logger, "代码库索引创建/获取")
            
            # 记录索引统计
            # index_stats = {
            #     "项目名称": indexed_repo_data.get("project_name", "未知"),
            #     "文件数量": len(indexed_repo_data.get("files", {})),
            #     "目录数量": len(indexed_repo_data.get("directory_structure", [])),
            #     "是否有依赖分析": indexed_repo_data.get("analysis_info", {}).get("has_dependency_analysis", "否")
            # }
            # log_detailed(self.paper_logger, "📊 代码库索引统计", index_stats)
            
            # 构建层级专门的系统提示词
            system_prompt = self._build_level_system_prompt(level_code, level_name, focus_areas, max_tasks)
            
            # 构建用户提示词
            user_prompt = self._build_level_user_prompt(
                level_code, level_description, processed_paper_guide, 
                indexed_repo_data, additional_content, target_metrics, iteration
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # 记录LLM调用信息
            llm_info = {
                "系统提示词长度": f"{len(system_prompt)} 字符",
                "用户提示词长度": f"{len(user_prompt)} 字符",
                "总输入长度": f"{len(system_prompt) + len(user_prompt)} 字符",
                "最大输出Token": "16384",
                "温度参数": "0.3"
            }
            log_detailed(self.paper_logger, "📡 LLM调用参数", llm_info)
            
            log_operation_start(self.paper_logger, f"{level_code}层级LLM分析")
            self.paper_logger.info("🔄 正在调用LLM进行层级分析...")
            
            # 使用现有的log_llm_call函数
            log_llm_call(self.paper_logger, "层级分析模型", 16384, len(user_prompt))
            
            # 调用LLM
            response = self.call_llm(messages, max_tokens=16384, temperature=0.3)
            log_operation_success(self.paper_logger, f"{level_code}层级LLM分析")
            
            # 记录响应统计
            response_stats = {
                "响应长度": f"{len(response)} 字符",
                "是否包含JSON": "是" if "{" in response and "}" in response else "否"
            }
            log_detailed(self.paper_logger, "📥 LLM响应统计", response_stats)
            
            # 解析响应
            log_operation_start(self.paper_logger, "JSON响应解析")
            result = parse_json_response(response)
            
            if result:
                log_operation_success(self.paper_logger, "JSON响应解析")
                self.paper_logger.info("✅ JSON解析成功")
            else:
                log_operation_error(self.paper_logger, "JSON响应解析", "解析失败或结果为空")
                self.paper_logger.error("❌ JSON解析失败")
            
            # 验证和增强结果
            log_operation_start(self.paper_logger, "任务验证和增强")
            
            if "tasks" in result and result["tasks"]:
                original_task_count = len(result["tasks"])
                # 限制任务数量
                result["tasks"] = result["tasks"][:max_tasks]
                
                if len(result["tasks"]) < original_task_count:
                    self.paper_logger.warning(f"⚠️  任务数量从 {original_task_count} 限制为 {len(result['tasks'])}")
                
                # 记录任务类型统计
                task_types = {}
                for task in result["tasks"]:
                    fixing_type = task.get("fixing_type", "未知")
                    task_types[fixing_type] = task_types.get(fixing_type, 0) + 1
                
                # 为每个任务添加层级信息
                for i, task in enumerate(result["tasks"]):
                    if "task_id" not in task:
                        task["task_id"] = f"{level_code}_task_{i+1}"
                    task["level_code"] = level_code
                    task["level_name"] = level_name
                    task["iteration"] = iteration
                    
                    # 设置默认值
                    if "priority" not in task:
                        task["priority"] = 5  # 中等优先级
                    if "estimated_impact" not in task:
                        task["estimated_impact"] = "中等影响"
                
                # 生成执行计划
                result["execution_plan"] = generate_execution_plan(result["tasks"])
                
                # 记录任务统计
                task_stats = {
                    "任务数量": len(result["tasks"]),
                    "任务类型分布": ", ".join([f"{k}:{v}" for k, v in task_types.items()]),
                    "平均优先级": f"{sum(t.get('priority', 5) for t in result['tasks']) / len(result['tasks']):.1f}",
                    "执行计划": "已生成" if result.get("execution_plan") else "未生成"
                }
                log_detailed(self.paper_logger, "📊 任务生成统计", task_stats)
                
                log_operation_success(self.paper_logger, "任务验证和增强")
            else:
                self.paper_logger.warning("⚠️  未生成任何任务")
                result["tasks"] = []
            
            # 添加层级分析信息
            result["level_analysis"] = {
                "level_code": level_code,
                "level_name": level_name,
                "focus_areas": focus_areas,
                "analysis_timestamp": get_current_timestamp(),
                "iteration": iteration,
                "tasks_generated": len(result.get("tasks", []))
            }
            
            # 记录分析完成
            log_operation_success(self.paper_logger, f"{level_code}层级分析")
            self.paper_logger.info(f"✅ {level_code}层级分析完成: 生成了 {len(result.get('tasks', []))} 个任务")
            
            # 记录分析结果摘要
            analysis_summary = {
                "层级": f"{level_code} - {level_name}",
                "生成任务数": len(result.get("tasks", [])),
                "分析状态": "成功",
                "耗时": "已完成"
            }
            if result.get("level_summary"):
                analysis_summary["AI总结"] = result["level_summary"][:100] + "..." if len(result.get("level_summary", "")) > 100 else result.get("level_summary", "")
            
            log_detailed(self.paper_logger, "📋 分析结果摘要", analysis_summary)
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            log_operation_error(self.paper_logger, f"{level_code}层级分析", error_msg)
            self.paper_logger.error(f"❌ {level_code}层级分析出现异常: {error_msg}")
            
            # 记录异常详情
            error_details = {
                "异常类型": type(e).__name__,
                "异常消息": error_msg,
                "层级代码": level_code,
                "迭代次数": iteration,
                "处理方式": "返回后备结果"
            }
            log_detailed(self.paper_logger, "🚨 异常详情", error_details)
            
            return self._generate_fallback_result(level_code, f"分析异常: {error_msg}")
    
    def _build_level_system_prompt(self, level_code: str, level_name: str, 
                                 focus_areas: List[str], max_tasks: int) -> str:
        """构建层级专门的系统提示词"""
        
        focus_list = "、".join(focus_areas) if focus_areas else "代码改进"
        
        return f"""你是一个专业的论文复现分析专家。你正在分析{level_code}层级（{level_name}）的代码改进需求。

                    专注领域: {focus_list}

                    请严格按照以下JSON格式返回结果：
                    {{
                        "tasks": [
                            {{
                                "task_id": "任务唯一标识",
                                "priority": 优先级数字(1-10, 10最高),
                                "fixing_type": "add_file 或 change_file",
                                "which_file_to_fix": "需要修改的文件路径",
                                "fixing_plan_in_detail": "详细的修复计划，说明具体要做什么",
                                "raw_code": "需要修改的原始代码片段（如果是修改文件）",
                                "dependencies": ["依赖的其他任务ID"],
                                "estimated_impact": "预估影响: 高影响/中等影响/低影响"
                            }}
                        ],
                        "level_summary": "本层级的整体分析总结",
                        "improvement_strategy": "针对{level_name}的改进策略"
                    }}

                    任务生成要求：
                    1. 最多生成{max_tasks}个最重要的任务
                    2. 专注于{level_name}相关的改进，不要涉及其他层级
                    3. 任务要具体可执行，有明确的文件和修改计划
                    4. 优先级要合理，关键任务优先级更高
                    5. 如果当前层级已经较好，可以生成少量优化任务或返回空任务列表"""
    #用try-except处理
    def _build_level_user_prompt(self, level_code: str, level_description: str, 
                               paper_guide: str, indexed_repo_data: Dict[str, Any],
                               additional_content: str, target_metrics: Dict[str, Any], iteration: int) -> str:
        """构建层级专门的用户提示词"""
        try:
            # 目标指标部分
            metrics_section = ""
            if target_metrics:
                metrics_section = f"""目标指标
                                    ```json
                                    {self._safe_json_dumps(target_metrics)}
                                    ```
                                    """
            
            # 补充信息部分
            additional_section = ""
            if additional_content and additional_content.strip():
                additional_section = f"""
                ## 补充信息
                ```
                {additional_content}
                ```
                """
            
            # 构建提示词
            prompt = f"""这是第{iteration}次迭代，专注于{level_code}层级的改进。

                ## 当前层级信息
                - 层级: {level_code}
                - 名称: {level_description}
                - 迭代: 第{iteration}次

                ## 论文复现指南
                ```
                {paper_guide}
                ```{additional_section}
                {metrics_section}
                ## 代码库索引信息
                ```json
                {self._get_safe_repo_summary(indexed_repo_data)}
                ```

                ## 分析要求

                请专注于{level_code}层级，分析当前代码库在以下方面的改进需求：

                """
            # 根据不同层级添加具体要求
            if level_code == "L0":
                prompt += """L0环境搭建 - 重点分析：
                            1. **依赖管理**: requirements.txt是否完整？缺少哪些包？
                            2. **环境配置**: 是否有配置文件？环境变量设置？
                            3. **数据准备**: 数据加载代码是否正确？路径是否配置？
                            4. **基础运行**: main文件能否正常导入和执行？
                            5. **错误修复**: 修复阻止程序运行的基础错误

                            生成的任务应该让程序能够**成功运行起来**。"""
                
            elif level_code == "L1":
                prompt += """L1核心算法 - 重点分析：
                            1. **算法实现**: 核心算法是否按论文描述实现？
                            2. **模型架构**: 神经网络结构是否正确？
                            3. **关键公式**: 重要的数学公式是否正确实现？
                            4. **核心组件**: 关键的类和函数是否存在且完整？
                            5. **算法逻辑**: 算法流程是否符合论文描述？

                            生成的任务应该让**核心算法实现正确**。"""
                
            elif level_code == "L2":
                prompt += """L2训练流程 - 重点分析：
                            1. **训练循环**: 训练过程是否完整？
                            2. **损失计算**: 损失函数是否正确实现？
                            3. **优化器**: 优化器配置是否合适？
                            4. **数据流水线**: 数据加载和预处理是否正确？
                            5. **推理流程**: 推理过程是否完整？

                            生成的任务应该让**整个训练推理流程能够顺利运行**。"""
                
            elif level_code == "L3":
                prompt += """L3实验复现 - 重点分析：
                            1. **实验脚本**: 是否有完整的实验运行脚本？
                            2. **评估指标**: 评估代码是否完整？指标计算是否正确？
                            3. **多个实验**: 论文中的各个实验是否都能运行？
                            4. **结果输出**: 实验结果是否正确保存和展示？
                            5. **数据集支持**: 是否支持论文中使用的所有数据集？

                            生成的任务应该让**所有论文实验都能正常运行**。"""
                
            elif level_code == "L4":
                prompt += """L4结果对齐 - 重点分析：
                            1. **结果比较**: 当前结果与论文结果的差距？
                            2. **参数调优**: 超参数是否需要调整？
                            3. **性能优化**: 是否有性能瓶颈需要优化？
                            4. **数值精度**: 数值计算精度是否足够？
                            5. **实验设置**: 实验设置是否与论文完全一致？

                            生成的任务应该让**实验结果尽可能接近论文报告的数值**。"""
            
            prompt += """请基于上述分析要求，生成针对当前层级的具体改进任务。一定要结合复现指南对每一个task给出详细的执行方案"""

            return prompt
        except Exception as e:
            self.paper_logger.error(f"❌ 构建用户提示词失败: {str(e)}")
            raise e
    
    def _generate_fallback_result(self, level_code: str, reason: str) -> Dict[str, Any]:
        """生成层级分析的后备结果"""
        return {
            "tasks": [],
            "level_analysis": {
                "level_code": level_code,
                "level_name": self.level_strategies.get(level_code, {}).get("name", level_code),
                "analysis_timestamp": get_current_timestamp(),
                "tasks_generated": 0,
                "fallback_reason": reason
            },
            "level_summary": f"{level_code}层级分析失败: {reason}",
            "improvement_strategy": "建议手动检查代码库状态",
            "execution_plan": {
                "total_tasks": 0,
                "execution_order": [],
                "risk_assessment": "分析失败，无法评估风险"
            }
        }
    
    def _get_safe_repo_summary(self, indexed_repo_data: Dict[str, Any]) -> str:
        """
        安全地获取代码库摘要信息，避免复杂嵌套字典序列化问题
        
        参数:
            indexed_repo_data (Dict[str, Any]): 原始的代码库索引数据
        
        返回:
            str: 安全的JSON字符串
        """
        try:
            # 安全地处理files字段，简化复杂的嵌套结构
            files_summary = {}
            original_files = indexed_repo_data.get("files", {})
            
            # 只取前10个文件，并简化每个文件的信息
            file_count = 0
            for file_path, file_data in original_files.items():
                if file_count >= 10:
                    break
                
                # 简化文件信息，只保留基本结构
                simplified_file_info = {
                    "functions": len(file_data.get("functions", [])),
                    "classes": list(file_data.get("classes", {}).keys())[:5],  # 只保留前5个类名
                    "has_functions": len(file_data.get("functions", [])) > 0,
                    "has_classes": len(file_data.get("classes", {})) > 0
                }
                
                files_summary[file_path] = simplified_file_info
                file_count += 1
            
            # 构建安全的数据结构
            safe_data = {
                "project_name": indexed_repo_data.get("project_name", ""),
                "directory_structure": indexed_repo_data.get("directory_structure", [])[:15],
                "files_summary": files_summary,
                "total_files": len(original_files),
                "analysis_info": {
                    "has_dependency_analysis": indexed_repo_data.get("analysis_info", {}).get("has_dependency_analysis", False),
                    "analyzer_timestamp": indexed_repo_data.get("analysis_info", {}).get("analyzer_timestamp", "")
                }
            }
            
            return self._safe_json_dumps(safe_data)
            
        except Exception as e:
            # 如果预处理失败，返回基本信息
            return self._safe_json_dumps({
                "error": f"代码库信息处理失败: {str(e)}",
                "project_name": indexed_repo_data.get("project_name", ""),
                "total_files": len(indexed_repo_data.get("files", {})),
                "has_directory_structure": len(indexed_repo_data.get("directory_structure", [])) > 0
            })
    
    def _safe_json_dumps(self, data: Any) -> str:
        """安全的JSON序列化，避免unhashable type错误"""
        try:
            return json.dumps(data, indent=2, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            # 如果序列化失败，返回字符串表示
            return f'{{"error": "JSON序列化失败: {str(e)}", "data_type": "{type(data).__name__}"}}' 

if __name__ == "__main__":
    # 导入必要的工具函数
    from .utils import load_additional_guides
    
    analyzer = PaperAnalyzerAgent()
    
    # 使用补充信息文档路径
    additional_guides_paths = [
        "/Users/wwchdemac/python_projects/debug_agent/test_papers/paper_test_1_addendum.md"
    ]
    
    # 加载补充信息内容
    additional_result = load_additional_guides(additional_guides_paths)
    additional_content = additional_result["additional_content"] if additional_result["success"] else ""
    
    result = analyzer.analyze_paper_level(
        level_code="L1",
        level_description="环境搭建",
        #用test paper 1 的guide
        paper_guide="/Users/wwchdemac/python_projects/debug_agent/test_papers/paper_test_1_reproduction_guide.md",
        #用RICE的repo地址
        repo_path="/Users/wwchdemac/python_projects/debug_agent/test_input/rice/submission",
        target_metrics={},
        additional_content=additional_content,  # 使用处理后的补充信息内容
        iteration=1
    )
    print('==================result====================')
    print(json.dumps(result, indent=4, ensure_ascii=False))
    print('==================result====================')
    print('\n\n')