"""
调试系统

协调Judger、Analyzer和Coder三个Agent实现自动调试。
支持修改历史记录和智能错误分类。

主要功能:
- debug_program(): 自动调试和修复程序错误
- _save_modification_history(): 保存修改历史
- _display_modification_summary(): 展示修改概述
"""

import os
import json
import subprocess
import shutil
from datetime import datetime
from typing import Dict, Any, Optional
from .judger_agent import JudgerAgent
from .analyzer_agent import AnalyzerAgent  
from .coder_agent import CoderAgent
from .utils import (
    get_colored_logger,
    execution_tools,
    run_program,
    get_timestamp,
    display_modification_summary,
    create_repo_index
)


class DebugSystem:
    """
    调试系统
    
    协调三个Agent的工作流程，实现自动化的程序调试和修复。
    支持修改历史记录、智能错误分类和渐进式代码改进。
    """
    
    def __init__(self, auto_mode: bool = False, **kwargs):
        """
        初始化调试系统
        
        参数:
            auto_mode (bool): 自动执行模式，默认为False。如果为True，将跳过用户交互，自动执行程序并迭代调试
            **kwargs: 配置参数
        """
        self.judger = JudgerAgent(**kwargs)
        self.analyzer = AnalyzerAgent(**kwargs)
        self.coder = CoderAgent(**kwargs)
        self.logger = get_colored_logger("DebugSystem")
        self.max_attempts = 10
        self.auto_mode = auto_mode

    def debug_program(self,
                     repo_path: str,
                     main_file: str,
                     expected_behavior: str = None,
                     output_dir: str = None) -> Dict[str, Any]:
        """
        自动调试和修复程序错误
        
        参数:
            repo_path (str): 代码仓库路径
            main_file (str): 主程序文件名
            expected_behavior (str, optional): 期望的程序行为描述
            output_dir (str, optional): 输出目录，默认为 repo_path/debug_output
        
        返回:
            dict: 调试结果
            {
                "success": bool,
                "attempts": int,
                "final_output": str,
                "debug_log": list,
                "repo_index_path": str,
                "fixed_repo_path": str,
                "modification_history_path": str,  # 新增：修改历史文件路径
                "final_error_category": str        # 新增：最终错误分类
            }
        """
        try:
            # 设置输出目录
            if not output_dir:
                output_dir = os.path.join(repo_path, "debug_output")
            os.makedirs(output_dir, exist_ok=True)
            
            self.logger.info(f"项目类型: Python Application Project")
            self.logger.info(f"推荐入口: {main_file}")
            self.logger.info(f"输出目录: {output_dir}")
            
            # 生成代码库索引
            self.logger.info("[STEP] 索引生成 -> 分析项目结构")
            repo_index_path = create_repo_index(repo_path, output_dir)
            self.logger.info("CHECKPOINT: 代码库索引生成完成")
            
            # 初始化调试结果
            debug_result = {
                "success": False,
                "attempts": 0,
                "final_output": "",
                "debug_log": [],
                "repo_index_path": repo_index_path,
                "fixed_repo_path": repo_path,
                "modification_history_path": os.path.join(output_dir, "modification_history.json"),
                "final_error_category": "未知"
            }
            
            self.logger.info("CHECKPOINT: 开始调试循环")
            
            # 调试循环
            for attempt in range(1, self.max_attempts + 1):
                debug_result["attempts"] = attempt
                
                self.logger.info(f"[STEP] 第{attempt}次尝试 -> 运行程序")
                
                # 根据模式决定是否询问用户输入
                if self.auto_mode:
                    # 自动模式：直接运行程序
                    self.logger.info(f"🤖 自动模式 - 第{attempt}次调试迭代")
                    self.logger.info("🚀 自动运行程序...")
                    stdout, stderr, return_code = run_program(repo_path, main_file)
                    print(f"📤 程序实际输出stdout:\n{stdout}")
                    print(f"📤 程序实际输出stderr:\n{stderr}")
                    print(f"📤 程序实际输出return_code:\n{return_code}")
                    self.logger.info(f"📤 程序输出 - stdout: {len(stdout)} 字符, stderr: {len(stderr)} 字符, return_code: {return_code}")
                else:
                    # 交互模式：询问用户是否提供自定义终端输出
                    print(f"\n{'='*60}")
                    print(f"🔄 第{attempt}次调试迭代 - 程序运行阶段")
                    print(f"{'='*60}")
                    print("💡 支持输入方式:")
                    print("   1. 直接输入错误信息")
                    print("   2. 输入txt文件路径 (如: /path/to/error.txt)")
                    print("   3. 直接回车则自动运行程序")
                    user_input = input("\n请选择输入方式: ")
                    
                    if user_input.strip():
                        # 检查是否是文件路径
                        if user_input.strip().endswith('.txt') and os.path.exists(user_input.strip()):
                            # 从文件读取内容
                            try:
                                with open(user_input.strip(), 'r', encoding='utf-8') as f:
                                    stdout = f.read().strip()
                                self.logger.info(f"📁 从文件读取终端输出: {user_input.strip()}")
                                print(f"📝 从文件读取的输出:\n{stdout}")
                            except Exception as e:
                                self.logger.error(f"❌ 读取文件失败: {e}")
                                stdout = f"文件读取错误: {str(e)}"
                                print(f"❌ 文件读取失败，使用错误信息: {stdout}")
                        else:
                            # 直接使用用户输入
                            stdout = user_input.strip()
                            self.logger.info("✏️ 使用用户直接输入的终端输出")
                            print(f"📝 用户输入的终端输出:\n{stdout}")
                        
                        stderr = ""
                        return_code = 1  # 假设有错误需要修复
                    else:
                        # 自动运行程序
                        self.logger.info("🚀 自动运行程序...")
                        stdout, stderr, return_code = run_program(repo_path, main_file)
                        print(f"📤 程序实际输出stdout:\n{stdout}")
                        print(f"📤 程序实际输出stderr:\n{stderr}")
                        print(f"📤 程序实际输出return_code:\n{return_code}")
                
                debug_result["final_output"] = stdout
                
                # 记录调试步骤
                step_log = {
                    "attempt": attempt,
                    "timestamp": datetime.now().isoformat(),
                    "stdout": stdout,
                    "stderr": stderr,
                    "return_code": return_code,
                    "judge_result": None,
                    "analyzer_result": None,
                    "coder_result": None
                }
                
                # Judger判断
                self.logger.info("[STEP] Judger -> 判断程序输出")
                # 如果stdout为空但stderr有内容，将stderr作为输出传递给judger
                output_for_judge = stdout if stdout.strip() else stderr
                judge_result = self.judger.judge_output(
                    stdout=output_for_judge,
                    expected_behavior=expected_behavior,
                    output_dir=output_dir,
                    iteration=attempt
                )
                step_log["judge_result"] = judge_result
                
                # 显示判断结果和错误分类
                error_category = judge_result.get("error_category", "未知")
                is_correct = judge_result.get("is_correct", False)
                reason = judge_result.get("reason", "")
                
                debug_result["final_error_category"] = error_category
                
                if is_correct:
                    if error_category == "业务错误":
                        self.logger.info(f"✅ 判断结果: 程序正确 ({error_category})")
                        self.logger.info(f"🎯 错误处理已优雅，属于业务逻辑: {reason}")
                    else:
                        self.logger.info("✅ 判断结果: 程序正确")
                    
                    debug_result["success"] = True
                    debug_result["debug_log"].append(step_log)
                    break
                else:
                    self.logger.warning(f"❌ 判断结果: {reason}")
                    self.logger.info(f"📊 错误分类: {error_category}")
                
                # 如果是业务错误但被判为错误，可能需要调整判断标准
                if error_category == "业务错误":
                    self.logger.info("💡 提示: 这可能是有意的业务错误处理，但仍需优化")
                
                trigger_analyzer = judge_result.get("trigger_analyzer", True)
                if not trigger_analyzer:
                    self.logger.info("🎯 Judger建议停止分析，错误处理已足够优雅")
                    debug_result["success"] = True
                    debug_result["debug_log"].append(step_log)
                    break
                
                # Analyzer分析
                self.logger.info("[STEP] Analyzer -> 多阶段错误分析")
                analyzer_result = self.analyzer.analyze_error(
                    stdout=stdout,
                    repo_path=repo_path,
                    indexed_repo_data=None,  # 将使用内部缓存
                    expected_behavior=expected_behavior
                )
                step_log["analyzer_result"] = analyzer_result
                
                # 检查分析是否成功
                if not analyzer_result.get("tasks"):
                    self.logger.error("❌ Analyzer未生成任何修复任务")
                    break
                
                # 显示分析结果概览
                tasks = analyzer_result.get("tasks", [])
                execution_plan = analyzer_result.get("execution_plan", {})
                
                self.logger.info(f"📋 分析完成: 生成了 {len(tasks)} 个修复任务")
                self.logger.info(f"🎯 执行计划: {execution_plan.get('risk_assessment', '未知风险')}")
                
                # 按执行顺序处理任务
                execution_order = execution_plan.get("execution_order", [task["task_id"] for task in tasks])
                task_results = []
                all_tasks_successful = True
                
                for task_id in execution_order:
                    # 找到对应的任务
                    current_task = None
                    for task in tasks:
                        if task["task_id"] == task_id:
                            current_task = task
                            break
                    
                    if not current_task:
                        self.logger.error(f"❌ 找不到任务: {task_id}")
                        all_tasks_successful = False
                        break
                    
                    # 执行当前任务
                    self.logger.info(f"🔧 执行任务 {current_task.get('priority', '?')}: {task_id}")
                    self.logger.info(f"   修复文件: {current_task.get('which_file_to_fix', '未知')}")
                    self.logger.info(f"   修复计划: {current_task.get('fixing_plan_in_detail', '未知')[:100]}...")
                    
                    # 调用Coder修复
                    coder_result = self.coder.fix_code(
                        task_dict=current_task,
                        repo_path=repo_path,
                        iteration=attempt,
                        output_dir=output_dir,
                        expected_behavior=expected_behavior
                    )
                    
                    # 记录任务结果
                    task_result = {
                        "task_id": task_id,
                        "task_info": current_task,
                        "coder_result": coder_result,
                        "success": coder_result.get("success", False)
                    }
                    task_results.append(task_result)
                    
                    # 显示任务修复结果
                    if coder_result.get("success", False):
                        action = coder_result.get("action_taken", "")
                        modification_summary = coder_result.get("modification_summary", "")
                        changes_made = coder_result.get("changes_made", [])
                        
                        self.logger.info(f"   ✅ 任务成功: {action}")
                        if coder_result.get("backup_created", False):
                            self.logger.info("   📁 已创建文件备份")
                        
                        # 显示修改概述（简化版）
                        if modification_summary:
                            self.logger.info(f"   📝 修改概述: {modification_summary[:80]}...")
                    else:
                        self.logger.error(f"   ❌ 任务失败: {coder_result.get('action_taken', '未知错误')}")
                        all_tasks_successful = False
                        
                        # 根据风险级别决定是否继续
                        risk_level = current_task.get("risk_level", "medium")
                        if risk_level == "high":
                            self.logger.error("   🚨 高风险任务失败，停止后续任务执行")
                            break
                        else:
                            self.logger.warning("   ⚠️ 任务失败但继续执行后续任务")
                
                # 更新step_log记录多任务结果
                step_log["multi_task_execution"] = {
                    "total_tasks": len(tasks),
                    "executed_tasks": len(task_results),
                    "successful_tasks": sum(1 for tr in task_results if tr["success"]),
                    "task_results": task_results,
                    "all_successful": all_tasks_successful
                }
                
                # 显示多任务执行总结
                successful_count = sum(1 for tr in task_results if tr["success"])
                if all_tasks_successful:
                    self.logger.info(f"🎉 所有任务执行完成: {successful_count}/{len(task_results)} 成功")
                else:
                    self.logger.warning(f"⚠️ 任务执行完成: {successful_count}/{len(task_results)} 成功")
                
                debug_result["debug_log"].append(step_log)
            
            # 如果达到最大尝试次数仍未成功
            if not debug_result["success"]:
                self.logger.warning(f"达到最大尝试次数 ({self.max_attempts})，调试结束")
                
                # 根据模式获取最终输出
                if self.auto_mode:
                    # 自动模式：直接运行程序获取最终输出
                    self.logger.info("🤖 自动模式 - 获取最终程序输出")
                    self.logger.info("🚀 自动运行程序获取最终输出...")
                    stdout, _, _ = run_program(repo_path, main_file)
                    self.logger.info(f"📤 最终程序输出: {len(stdout)} 字符")
                else:
                    # 交互模式：获取最终输出
                    print(f"\n{'='*60}")
                    print(f"🏁 调试结束 - 获取最终程序输出")
                    print(f"{'='*60}")
                    print("💡 支持输入方式:")
                    print("   1. 直接输入错误信息")
                    print("   2. 输入txt文件路径 (如: /path/to/error.txt)")
                    print("   3. 直接回车则自动运行程序")
                    user_final_input = input("\n请选择输入方式: ")
                    
                    if user_final_input.strip():
                        # 检查是否是文件路径
                        if user_final_input.strip().endswith('.txt') and os.path.exists(user_final_input.strip()):
                            # 从文件读取内容
                            try:
                                with open(user_final_input.strip(), 'r', encoding='utf-8') as f:
                                    stdout = f.read().strip()
                                self.logger.info(f"📁 从文件读取最终终端输出: {user_final_input.strip()}")
                                print(f"📝 从文件读取的最终输出:\n{stdout}")
                            except Exception as e:
                                self.logger.error(f"❌ 读取文件失败: {e}")
                                stdout = f"文件读取错误: {str(e)}"
                                print(f"❌ 文件读取失败，使用错误信息: {stdout}")
                        else:
                            # 直接使用用户输入
                            stdout = user_final_input.strip()
                            self.logger.info("✏️ 使用用户直接输入的最终终端输出")
                            print(f"📝 用户输入的最终输出:\n{stdout}")
                    else:
                        # 自动运行程序获取最终输出
                        self.logger.info("🚀 自动运行程序获取最终输出...")
                        stdout, _, _ = run_program(repo_path, main_file)
                        print(f"📤 程序最终输出:\n{stdout}")
                
                debug_result["final_output"] = stdout
            
            # 保存调试报告
            report_path = os.path.join(output_dir, "debug_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(debug_result, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"调试报告已保存: {report_path}")
            
            return debug_result
            
        except Exception as e:
            self.logger.error(f"调试系统出现异常: {str(e)}")
            return {
                "success": False,
                "attempts": 0,
                "final_output": "",
                "debug_log": [],
                "repo_index_path": "",
                "fixed_repo_path": repo_path,
                "modification_history_path": "",
                "final_error_category": "系统错误",
                "error": str(e)
            }

    def _display_modification_summary(self, modification_summary: str, changes_made: list, iteration: int) -> None:
        """
        显示修改概述
        
        参数:
            modification_summary (str): 修改概述
            changes_made (list): 具体变化列表
            iteration (int): 迭代次数
        """
        self.logger.info(f"📝 第{iteration}次修改概述: {modification_summary}")
        if changes_made:
            for change in changes_made[:3]:  # 最多显示3个主要变化
                self.logger.info(f"   └─ {change}")
            if len(changes_made) > 3:
                self.logger.info(f"   └─ ... 还有 {len(changes_made) - 3} 项变化")

    def _create_repo_index(self, repo_path: str, output_dir: str) -> str:
        """
        创建代码库索引 - 使用基础结构分析
        
        参数:
            repo_path (str): 代码库路径
            output_dir (str): 输出目录
        
        返回:
            str: 索引文件路径
        """
        from core_modules import SimpleStructureAnalyzer
        
        print(f"分析项目: {repo_path}")
        
        # 基础结构分析 - 获取项目结构、函数签名、类定义
        self.logger.info("🏗️ 结构分析")
        structure_analyzer = SimpleStructureAnalyzer(repo_path)
        structure_result = structure_analyzer.analyze_project()
        
        # 统计分析结果
        python_files = 0
        for item in structure_result.get("directory_structure", []):
            if isinstance(item, str) and item.strip().endswith(".py"):
                python_files += 1
        
        print(f"完成结构分析: {python_files} 个Python文件")
        self.logger.info(f"Found {python_files} Python files")
        
        # 构建分析结果
        self.logger.info("🔧 构建分析结果")
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
                "python_files_count": python_files,
                "has_dependency_analysis": False,
                "combined_timestamp": get_timestamp()
            }
        }
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存索引文件
        index_path = os.path.join(output_dir, "repo_index.json")
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(combined_result, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"代码库索引已保存: {index_path}")
        
        print(f"完成结构分析")
        return index_path
    
    # def _get_timestamp(self) -> str:
    #     """获取当前时间戳"""
    #     from datetime import datetime
    # #     return datetime.now().isoformat()

    # # def _run_program(self, repo_path: str, main_file: str) -> tuple:
    #     """
    #     运行程序
        
    #     参数:
    #         repo_path (str): 代码库路径
    #         main_file (str): 主程序文件名
        
    #     返回:
    #         tuple: (stdout, stderr, return_code)
    #     """
    #     program_path = os.path.join(repo_path, main_file)
        
    #     try:
    #         # 构建执行命令和工作目录
    #         if program_path.endswith('.py'):
    #             work_dir = os.path.dirname(program_path)
    #             file_name = os.path.basename(program_path)
    #             if not work_dir:
    #                 work_dir = os.getcwd()
    #             cmd = ['python', file_name]
    #             execution_dir = work_dir
    #         else:
    #             cmd = [program_path]
    #             execution_dir = os.path.dirname(program_path)
            
    #         self.logger.info(f"执行命令: {' '.join(cmd)}")
    #         self.logger.info(f"工作目录: {execution_dir}")
            
    #         result = subprocess.run(
    #             cmd,
    #             capture_output=True,
    #             text=True,
    #             timeout=30,
    #             cwd=execution_dir
    #         )
            
    #         if result.returncode == 0:
    #             self.logger.info("程序执行成功")
    #         else:
    #             self.logger.warning(f"程序执行返回非零码: {result.returncode}")
            
    #         self.logger.info(f"程序返回码: {result.returncode}")
    #         self.logger.info(f"标准输出长度: {len(result.stdout)} 字符")
            
    #         if result.stderr:
    #             self.logger.warning(f"错误输出: {result.stderr}")
            
    #         return result.stdout, result.stderr, result.returncode
            
    #     except subprocess.TimeoutExpired:
    #         return "", "程序执行超时", -1
    #     except FileNotFoundError:
    #         return "", f"找不到程序文件: {program_path}", -1
    #     except Exception as e:
            # return "", f"执行程序时出现异常: {str(e)}", -1 

if __name__ == "__main__":
    # 测试_create_repo_index
    project_path = "/Users/wwchdemac/python_projects/debug_agent/test_input/webpage"
    output_dir = "/Users/wwchdemac/python_projects/debug_agent/test_output/webpage"
    debug_system = DebugSystem()
    index_path = debug_system._create_repo_index(project_path, output_dir)
    
    # 读取并优雅打印结果
    with open(index_path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    print(json.dumps(result, indent=4, ensure_ascii=False))
    
    # 统计JSON字符数量
    json_str = json.dumps(result, indent=4, ensure_ascii=False)
    print(f"终端打印的字符数量: {len(json_str)}")