"""
论文复现系统

基于现有的调试系统架构，专门用于论文复现的迭代工作流。
采用5次固定迭代，每次专注一个层级：
L0: 环境搭建 -> L1: 核心算法 -> L2: 训练流程 -> L3: 实验复现 -> L4: 结果对齐

主要功能:
- PaperReproductionSystem: 论文复现主系统
- PaperAnalyzerAgent: 专门的论文分析Agent
- PaperCoderAgent: 专门的论文代码修改Agent
- 5次固定迭代，无需复杂判断逻辑  
- 专注于论文复现的特定需求
- 完全不影响原有调试工作流
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List
from .base_agent import BaseAgent
from .utils import (
    get_colored_logger, 
    get_timestamp,
    log_detailed,
    log_llm_call,
    log_operation_start,
    log_operation_success,
    log_operation_error,
    load_additional_guides
)


class PaperReproductionSystem:
    """
    论文复现系统
    
    采用5次固定迭代的方式，逐步改进代码库以达到论文复现的要求：
    - 第1次迭代 (L0): 环境搭建，确保程序能运行
    - 第2次迭代 (L1): 核心算法，实现关键算法组件
    - 第3次迭代 (L2): 训练流程，完善训练和推理流程
    - 第4次迭代 (L3): 实验复现，实现所有论文实验
    - 第5次迭代 (L4): 结果对齐，优化结果向论文靠拢
    """
    
    def __init__(self, **kwargs):
        """
        初始化论文复现系统
        
        参数:
            **kwargs: 传递给BaseAgent的参数
        """
        self.logger = get_colored_logger("PaperReproduction")
        
        # 导入时才初始化，避免循环导入
        from .paper_analyzer_agent import PaperAnalyzerAgent
        from .paper_coder_agent import PaperCoderAgent
        
        self.analyzer = PaperAnalyzerAgent(**kwargs)
        self.coder = PaperCoderAgent(**kwargs)
        
        # 论文复现的5个层级
        self.reproduction_levels = {
            1: {
                "code": "L0",
                "name": "环境搭建",
                "description": "确保程序能正常运行，依赖安装完整，数据能正确加载",
                "focus": "基础环境和运行能力"
            },
            2: {
                "code": "L1", 
                "name": "核心算法",
                "description": "实现论文的核心算法组件，确保关键公式和模型架构正确",
                "focus": "算法实现的正确性"
            },
            3: {
                "code": "L2",
                "name": "训练流程", 
                "description": "完善训练和推理流程，确保整个pipeline能正常工作",
                "focus": "端到端流程的完整性"
            },
            4: {
                "code": "L3",
                "name": "实验复现",
                "description": "实现论文中的所有实验，确保能跑出实验结果",
                "focus": "实验的可重现性"
            },
            5: {
                "code": "L4",
                "name": "结果对齐", 
                "description": "优化实验结果，使其尽可能接近论文报告的数值",
                "focus": "结果的准确性和一致性"
            }
        }
    
    def reproduce_paper(self, 
                       repo_path: str,
                       paper_guide: str, 
                       additional_guides: List[str] = None,
                       target_metrics: Dict[str, Any] = None,
                       max_iterations: int = 5,
                       output_dir: str = None) -> Dict[str, Any]:
        """
        执行论文复现的主要流程
        
        参数:
            repo_path (str): 代码库路径
            paper_guide (str): 论文复现指南（包含算法描述、实验设置等）
            additional_guides (List[str]): 补充信息文档路径列表（可选）
            target_metrics (dict): 目标指标（期望达到的结果）
            max_iterations (int): 最大迭代次数，默认5次
            output_dir (str): 输出目录，用于保存复现过程和结果
        
        返回:
            dict: 复现结果
        """
        
        # 创建输出目录
        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(repo_path, f"paper_reproduction_output_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 开始论文复现流程
        log_operation_start(self.logger, "论文复现系统初始化")
        self.logger.info("🚀 开始论文复现流程")
        
        # 记录配置详情
        config_details = {
            "代码库路径": repo_path,
            "输出目录": output_dir,
            "计划迭代": f"{max_iterations} 次",
            "论文指南长度": f"{len(paper_guide)} 字符",
            "目标指标数量": f"{len(target_metrics or {})} 项"
        }
        
        # 处理补充信息文档
        additional_content = ""
        additional_info = {}
        if additional_guides:
            log_operation_start(self.logger, "处理补充信息文档")
            self.logger.info(f"📄 开始处理 {len(additional_guides)} 个补充信息文档...")
            
            additional_result = load_additional_guides(additional_guides)
            if additional_result["success"]:
                additional_content = additional_result["additional_content"]
                additional_info = additional_result["metadata"]
                
                # 更新配置详情
                config_details["补充文档数量"] = f"{additional_info['processed_count']}/{additional_info['total_files']} 个"
                config_details["补充信息长度"] = f"{additional_info['final_char_count']} 字符"
                
                log_operation_success(self.logger, "处理补充信息文档")
                self.logger.info(f"✅ 成功处理补充信息: {additional_info['processed_count']} 个文档")
            else:
                config_details["补充文档状态"] = "处理失败"
                log_operation_error(self.logger, "处理补充信息文档", "所有文档都无法处理")
                self.logger.warning("⚠️  补充信息文档处理失败，将仅使用主要论文指南")
        else:
            config_details["补充文档数量"] = "未提供"
        
        log_detailed(self.logger, "📋 论文复现配置信息", config_details)
        
        # 系统组件初始化日志
        self.logger.info("🔧 初始化系统组件...")
        self.logger.info(f"  • Analyzer: PaperAnalyzerAgent")
        self.logger.info(f"  • Coder: PaperCoderAgent")
        log_operation_success(self.logger, "系统组件初始化")
        
        # 初始化复现结果
        reproduction_result = {
            "success": False,
            "repo_path": repo_path,
            "paper_guide": paper_guide,
            "additional_guides": additional_guides or [],
            "additional_content": additional_content,
            "additional_info": additional_info,
            "target_metrics": target_metrics or {},
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_iterations": 0,
            "iterations_log": [],
            "final_level_achieved": "L0",
            "improvements_made": []
        }
        
        try:
            # 执行5次固定迭代
            for iteration in range(1, max_iterations + 1):
                level_info = self.reproduction_levels[iteration]
                
                # 迭代开始日志
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"🔄 第 {iteration}/{max_iterations} 次迭代 - {level_info['code']}: {level_info['name']}")
                
                # 迭代详情日志
                iteration_details = {
                    "层级代码": level_info['code'],
                    "层级名称": level_info['name'],
                    "专注点": level_info['focus'],
                    "目标描述": level_info['description']
                }
                log_detailed(self.logger, "📋 迭代配置信息", iteration_details)
                self.logger.info(f"{'='*60}")
                
                # 记录迭代开始操作
                log_operation_start(self.logger, f"第{iteration}次迭代 - {level_info['code']}层级")
                
                # 记录迭代开始
                iteration_log = {
                    "iteration": iteration,
                    "level": level_info,
                    "start_time": datetime.now().isoformat(),
                    "analyzer_result": None,
                    "tasks_executed": [],
                    "success": False,
                    "improvements": []
                }
                
                try:
                    # 1. Analyzer分析当前层级
                    log_operation_start(self.logger, f"Analyzer分析{level_info['code']}层级")
                    self.logger.info(f"🔍 [STEP] Analyzer -> 分析{level_info['code']}层级")
                    
                    # 记录分析参数
                    analyzer_params = {
                        "层级代码": level_info['code'],
                        "层级描述": level_info['description'][:100] + "..." if len(level_info['description']) > 100 else level_info['description'],
                        "迭代次数": iteration,
                        "论文指南长度": f"{len(paper_guide)} 字符",
                        "补充信息长度": f"{len(additional_content)} 字符" if additional_content else "无补充信息"
                    }
                    log_detailed(self.logger, "📊 Analyzer分析参数", analyzer_params)
                    
                    analyzer_result = self.analyzer.analyze_paper_level(
                        level_code=level_info['code'],
                        level_description=level_info['description'], 
                        paper_guide=paper_guide,
                        repo_path=repo_path,
                        additional_content=additional_content,
                        target_metrics=target_metrics,
                        iteration=iteration
                    )
                    
                    iteration_log["analyzer_result"] = analyzer_result
                    log_operation_success(self.logger, f"Analyzer分析{level_info['code']}层级")
                    
                    # 检查是否生成了任务
                    tasks = analyzer_result.get("tasks", [])
                    if not tasks:
                        self.logger.warning(f"⚠️  {level_info['code']}层级未生成任何改进任务")
                        self.logger.info(f"📊 可能原因: 当前层级已经达到预期状态")
                        iteration_log["success"] = True  # 可能已经完成了
                        iteration_log["improvements"].append(f"{level_info['code']}层级无需改进")
                    else:
                        self.logger.info(f"📝 生成了 {len(tasks)} 个改进任务")
                        
                        # 记录任务概览
                        task_overview = {}
                        for i, task in enumerate(tasks, 1):
                            task_overview[f"任务{i}"] = f"{task.get('fixing_type', '未知类型')} - {task.get('which_file_to_fix', '未知文件')}"
                        log_detailed(self.logger, "📋 任务概览", task_overview)
                        
                        # 2. 执行所有任务
                        log_operation_start(self.logger, f"执行{len(tasks)}个任务")
                        for task_idx, task in enumerate(tasks, 1):
                            self.logger.info(f"🔧 执行任务 {task_idx}/{len(tasks)}: {task.get('task_id', 'unknown')}")
                            
                            # 记录任务详情
                            task_details = {
                                "任务ID": task.get('task_id', 'unknown'),
                                "修复类型": task.get('fixing_type', '未知'),
                                "目标文件": task.get('which_file_to_fix', '未知'),
                                "优先级": task.get('priority', '未设置'),
                                "预估影响": task.get('estimated_impact', '未知')
                            }
                            log_detailed(self.logger, f"📋 任务{task_idx}详情", task_details)
                            
                            # 显示修复计划摘要
                            fixing_plan = task.get('fixing_plan_in_detail', '未知')
                            plan_summary = fixing_plan[:100] + "..." if len(fixing_plan) > 100 else fixing_plan
                            self.logger.info(f"📝 修复计划: {plan_summary}")
                            
                            # 构建包含补充信息的期望行为描述
                            expected_behavior = f"实现论文{level_info['code']}层级: {level_info['description']}"
                            if additional_content:
                                expected_behavior += f" (请参考已提供的补充信息文档以获得更多实现细节)"
                            
                            # 调用Coder执行任务
                            log_operation_start(self.logger, f"Coder执行任务{task_idx}")
                            coder_result = self.coder.fix_code(
                                task_dict=task,
                                repo_path=repo_path,
                                iteration=iteration,
                                output_dir=output_dir,
                                expected_behavior=expected_behavior,
                                paper_guide=paper_guide,
                                additional_content=additional_content
                            )
                            
                            # 记录任务结果
                            task_result = {
                                "task_id": task.get("task_id", f"task_{task_idx}"),
                                "task_info": task,
                                "coder_result": coder_result,
                                "success": coder_result.get("success", False)
                            }
                            
                            iteration_log["tasks_executed"].append(task_result)
                            
                            if coder_result.get("success"):
                                improvement = f"成功{task.get('fixing_type', '修改')}文件: {task.get('which_file_to_fix', '未知')}"
                                iteration_log["improvements"].append(improvement)
                                log_operation_success(self.logger, f"任务{task_idx}执行")
                                self.logger.info(f"✅ 任务完成: {improvement}")
                                
                                # 记录文件修改详情
                                if coder_result.get("changes_made"):
                                    changes_summary = {}
                                    for i, change in enumerate(coder_result["changes_made"][:3], 1):
                                        changes_summary[f"修改{i}"] = change
                                    log_detailed(self.logger, "🔄 代码修改详情", changes_summary)
                            else:
                                error_msg = coder_result.get('action_taken', '未知错误')
                                log_operation_error(self.logger, f"任务{task_idx}执行", error_msg)
                                self.logger.error(f"❌ 任务失败: {error_msg}")
                        
                        # 判断迭代是否成功
                        successful_tasks = sum(1 for task_result in iteration_log["tasks_executed"] 
                                             if task_result.get("success", False))
                        iteration_log["success"] = successful_tasks > 0
                        
                        # 记录迭代结果统计
                        iteration_stats = {
                            "成功任务": f"{successful_tasks}/{len(tasks)}",
                            "成功率": f"{(successful_tasks/len(tasks)*100):.1f}%" if len(tasks) > 0 else "0%",
                            "改进项数": len(iteration_log["improvements"])
                        }
                        
                        if iteration_log["success"]:
                            log_operation_success(self.logger, f"第{iteration}次迭代")
                            self.logger.info(f"✅ 第{iteration}次迭代完成")
                            log_detailed(self.logger, "📊 迭代结果统计", iteration_stats)
                        else:
                            self.logger.warning(f"⚠️  第{iteration}次迭代部分完成")
                            log_detailed(self.logger, "📊 迭代结果统计", iteration_stats)
                            self.logger.warning(f"🔍 建议检查失败任务的具体原因")
                
                except Exception as e:
                    error_msg = str(e)
                    log_operation_error(self.logger, f"第{iteration}次迭代", error_msg)
                    self.logger.error(f"❌ 第{iteration}次迭代出现异常: {error_msg}")
                    
                    # 记录异常详情
                    error_details = {
                        "异常类型": type(e).__name__,
                        "异常消息": error_msg,
                        "迭代阶段": f"{level_info['code']}层级",
                        "已完成任务": len(iteration_log.get("tasks_executed", []))
                    }
                    log_detailed(self.logger, "🚨 异常详情", error_details)
                    
                    iteration_log["error"] = error_msg
                    iteration_log["success"] = False
                
                # 完成迭代记录
                iteration_log["end_time"] = datetime.now().isoformat()
                reproduction_result["iterations_log"].append(iteration_log)
                reproduction_result["total_iterations"] = iteration
                reproduction_result["final_level_achieved"] = level_info['code']
                
                # 收集总体改进
                if iteration_log.get("improvements"):
                    reproduction_result["improvements_made"].extend(iteration_log["improvements"])
            
            # 复现流程完成
            reproduction_result["success"] = True
            reproduction_result["end_time"] = datetime.now().isoformat()
            
            # 总结结果
            total_improvements = len(reproduction_result["improvements_made"])
            successful_iterations = sum(1 for log in reproduction_result["iterations_log"] 
                                      if log.get("success", False))
            
            # 计算总体统计
            total_tasks = sum(len(log.get("tasks_executed", [])) for log in reproduction_result["iterations_log"])
            successful_tasks = sum(len([t for t in log.get("tasks_executed", []) if t.get("success", False)]) 
                                 for log in reproduction_result["iterations_log"])
            
            # 记录最终成功
            log_operation_success(self.logger, "论文复现流程")
            self.logger.info(f"\n🎉 论文复现流程完成!")
            
            # 详细的最终统计
            final_stats = {
                "成功迭代": f"{successful_iterations}/{max_iterations} 次",
                "迭代成功率": f"{(successful_iterations/max_iterations*100):.1f}%",
                "总计任务": f"{successful_tasks}/{total_tasks}",
                "任务成功率": f"{(successful_tasks/total_tasks*100):.1f}%" if total_tasks > 0 else "0%",
                "总计改进": f"{total_improvements} 项",
                "最终层级": reproduction_result['final_level_achieved'],
                "耗时": self._calculate_duration(reproduction_result['start_time'], reproduction_result['end_time'])
            }
            log_detailed(self.logger, "📊 最终统计报告", final_stats)
            
            if total_improvements > 0:
                self.logger.info("🚀 主要改进项目:")
                for improvement in reproduction_result["improvements_made"][:5]:  # 显示前5个
                    self.logger.info(f"   • {improvement}")
                if total_improvements > 5:
                    self.logger.info(f"   ... 以及其他 {total_improvements - 5} 项改进")
            
            # 按层级显示完成情况
            level_completion = {}
            for log in reproduction_result["iterations_log"]:
                level_code = log.get("level", {}).get("code", "未知")
                level_name = log.get("level", {}).get("name", "未知")
                success = "✅ 完成" if log.get("success", False) else "❌ 未完成"
                level_completion[f"{level_code} {level_name}"] = success
            log_detailed(self.logger, "📋 层级完成情况", level_completion)
            
            # 保存结果到文件
            result_file = os.path.join(output_dir, "reproduction_result.json")
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(reproduction_result, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📁 详细结果已保存到: {result_file}")
            
            return reproduction_result
            
        except Exception as e:
            self.logger.error(f"❌ 论文复现过程出现严重异常: {str(e)}")
            reproduction_result["success"] = False
            reproduction_result["error"] = str(e)
            reproduction_result["end_time"] = datetime.now().isoformat()
            return reproduction_result
    
    def get_reproduction_summary(self, result: Dict[str, Any]) -> str:
        """
        生成复现结果摘要
        
        参数:
            result (dict): 复现结果
            
        返回:
            str: 格式化的摘要字符串
        """
        if not result:
            return "❌ 无复现结果"
        
        success = "✅ 成功" if result.get("success", False) else "❌ 失败"
        total_iterations = result.get("total_iterations", 0)
        final_level = result.get("final_level_achieved", "未知")
        improvements_count = len(result.get("improvements_made", []))
        
        summary = f"""
📊 论文复现摘要 {success}
{'='*40}
🎯 最终层级: {final_level}
🔄 完成迭代: {total_iterations}/5 次
🔧 总计改进: {improvements_count} 项
⏰ 开始时间: {result.get('start_time', '未知')}
⏰ 结束时间: {result.get('end_time', '未知')}
"""
        
        if result.get("error"):
            summary += f"\n❌ 错误信息: {result['error']}"
        
        return summary
    
    def _calculate_duration(self, start_time: str, end_time: str) -> str:
        """计算持续时间"""
        try:
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            duration = end - start
            
            total_seconds = int(duration.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            
            if hours > 0:
                return f"{hours}小时{minutes}分钟{seconds}秒"
            elif minutes > 0:
                return f"{minutes}分钟{seconds}秒"
            else:
                return f"{seconds}秒"
        except Exception:
            return "未知" 