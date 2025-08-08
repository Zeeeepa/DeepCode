"""
论文代码修改Agent

继承自CoderAgent，专门用于论文复现的代码修改需求。
相比通用的CoderAgent，更注重算法实现的正确性和论文描述的一致性。

主要功能:
- PaperCoderAgent: 专门的论文代码修改Agent
- 增强的代码生成提示词，更注重算法正确性
- 特殊的验证逻辑，确保符合论文描述
- 完全不影响原有的CoderAgent功能
"""

import os
from typing import Dict, Any
from .coder_agent import CoderAgent
from .utils import (
    analyze_code_changes, 
    clean_llm_code_output,
    validate_code_content,
    update_modification_history,
    get_colored_logger,
    log_detailed,
    log_llm_call,
    log_operation_start,
    log_operation_success,
    log_operation_error
)


class PaperCoderAgent(CoderAgent):
    """
    论文代码修改Agent
    
    继承自CoderAgent，专门用于论文复现的代码修改。
    相比通用的代码修复，更注重：
    - 算法实现的准确性和完整性
    - 与论文描述的一致性
    - 科学计算的精度和正确性
    - 实验结果的可重现性
    """
    
    def __init__(self, **kwargs):
        """
        初始化论文代码修改Agent
        
        参数:
            **kwargs: 传递给父类CoderAgent的参数
        """
        super().__init__(**kwargs)
        
        # 初始化专门的彩色日志记录器
        self.paper_logger = get_colored_logger("PaperCoder")
        
        # 论文复现的特殊要求
        self.paper_coding_principles = {
            "algorithm_accuracy": "确保算法实现与论文描述完全一致",
            "numerical_precision": "注重数值计算的精度，避免精度损失", 
            "reproducibility": "确保实验结果的可重现性",
            "scientific_rigor": "遵循科学计算的严谨性要求",
            "documentation": "添加详细的算法说明和公式注释"
        }
        
        self.paper_logger.info("🛠️  PaperCoderAgent 初始化完成")
        
        # 记录编码原则
        log_detailed(self.paper_logger, "📋 论文编码原则", self.paper_coding_principles)
    
    def fix_code(self, task_dict: Dict[str, Any], repo_path: str, iteration: int = 1, 
                output_dir: str = None, expected_behavior: str = None,
                paper_guide: str = "", additional_content: str = "") -> Dict[str, Any]:
        """
        执行论文复现相关的代码修改
        
        参数:
            task_dict (Dict[str, Any]): 从PaperAnalyzerAgent获得的任务字典
            repo_path (str): 仓库路径
            iteration (int): 迭代次数
            output_dir (str): 输出目录
            expected_behavior (str): 期望的程序行为
            paper_guide (str): 论文复现指南内容（可选）
            additional_content (str): 补充信息内容（可选）
        
        返回:
            dict: 修复结果
        """
        # 提取论文相关信息
        level_code = task_dict.get("level_code", "未知")
        level_name = task_dict.get("level_name", "未知层级")
        task_id = task_dict.get("task_id", "未知任务")
        fixing_type = task_dict.get("fixing_type", "未知类型")
        target_file = task_dict.get("which_file_to_fix", "未知文件")
        
        # 开始任务执行日志
        log_operation_start(self.paper_logger, f"论文代码修改任务 - {task_id}")
        self.paper_logger.info(f"🛠️  开始执行论文代码修改任务: {task_id}")
        
        # 记录任务详情
        task_info = {
            "任务ID": task_id,
            "层级": f"{level_code} - {level_name}",
            "修复类型": fixing_type,
            "目标文件": target_file,
            "迭代次数": iteration,
            "优先级": task_dict.get("priority", "未设置")
        }
        log_detailed(self.paper_logger, "📋 任务信息", task_info)
        
        # 构建论文专用的期望行为描述
        paper_expected_behavior = self._build_paper_expected_behavior(
            task_dict, expected_behavior, level_code, level_name
        )
        
        self.paper_logger.info(f"🎯 期望行为: {paper_expected_behavior[:100]}..." if len(paper_expected_behavior) > 100 else paper_expected_behavior)
        
        # 直接调用论文专用的代码修改方法
        log_operation_start(self.paper_logger, f"调用论文专用代码修改引擎")
        
        # 提取任务信息
        fixing_type = task_dict.get("fixing_type", "change_file")
        file_path = task_dict.get("which_file_to_fix", "")
        fixing_plan = task_dict.get("fixing_plan_in_detail", "")
        raw_code = task_dict.get("raw_code", "")
        
        # 构建完整的文件路径
        if not os.path.isabs(file_path):
            full_file_path = os.path.join(repo_path, file_path)
        else:
            full_file_path = file_path
        
        # 根据修复类型调用相应方法
        if fixing_type == "add_file":
            result = self._add_new_file(
                file_path=full_file_path,
                fixing_plan=fixing_plan,
                iteration=iteration,
                expected_behavior=paper_expected_behavior,
                paper_guide=paper_guide,
                additional_content=additional_content
            )
        else:  # change_file 或其他类型默认为修改文件
            result = self._modify_existing_file(
                file_path=full_file_path,
                fixing_plan=fixing_plan,
                raw_code=raw_code,
                iteration=iteration,
                output_dir=output_dir,
                expected_behavior=paper_expected_behavior,
                paper_guide=paper_guide,
                additional_content=additional_content
            )
        
        # 记录执行结果
        if result.get("success"):
            log_operation_success(self.paper_logger, f"论文代码修改任务 - {task_id}")
            self.paper_logger.info(f"✅ 任务 {task_id} 执行成功")
            
            # 增强结果信息
            result["paper_level"] = level_code
            result["paper_level_name"] = level_name
            result["coding_principles_applied"] = self._get_applied_principles(task_dict)
            
            # 记录应用的编码原则
            applied_principles = result["coding_principles_applied"]
            if applied_principles:
                principles_info = {}
                for principle in applied_principles:
                    principles_info[principle] = self.paper_coding_principles.get(principle, "未知原则")
                log_detailed(self.paper_logger, "📏 应用的编码原则", principles_info)
            
            # 记录修改统计
            modification_stats = {
                "修改文件": result.get("file_path", "未知"),
                "修改类型": result.get("action_taken", "未知"),
                "是否备份": "是" if result.get("backup_created") else "否",
                "修改项数": len(result.get("changes_made", []))
            }
            log_detailed(self.paper_logger, "📊 修改统计", modification_stats)
            
        else:
            error_msg = result.get("action_taken", "未知错误")
            log_operation_error(self.paper_logger, f"论文代码修改任务 - {task_id}", error_msg)
            self.paper_logger.error(f"❌ 任务 {task_id} 执行失败: {error_msg}")
            
            # 记录失败详情
            failure_details = {
                "任务ID": task_id,
                "失败原因": error_msg,
                "目标文件": target_file,
                "修复类型": fixing_type
            }
            log_detailed(self.paper_logger, "🚨 任务失败详情", failure_details)
        
        return result
    
    def _modify_existing_file(self, file_path: str, fixing_plan: str, raw_code: str, 
                            iteration: int, output_dir: str = None, expected_behavior: str = None,
                            paper_guide: str = "", additional_content: str = "") -> Dict[str, Any]:
        """
        修改现有文件 - 论文复现版本
        
        重写父类方法，使用论文专用的提示词和验证逻辑
        
        参数:
            file_path (str): 文件路径
            fixing_plan (str): 修复计划
            raw_code (str): 原始代码
            iteration (int): 迭代次数
            output_dir (str): 输出目录
            expected_behavior (str): 期望行为
            paper_guide (str): 论文复现指南内容
            additional_content (str): 补充信息内容
        """
        try:
            log_operation_start(self.paper_logger, f"修改文件 - {os.path.basename(file_path)}")
            
            if not os.path.exists(file_path):
                error_msg = f"文件不存在: {file_path}"
                log_operation_error(self.paper_logger, "文件检查", error_msg)
                return self._create_error_result(file_path, error_msg, iteration)
            
            self.paper_logger.info(f"📄 找到目标文件: {file_path}")
            
            # 读取现有文件内容
            log_operation_start(self.paper_logger, "读取原始文件内容")
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            file_stats = {
                "文件路径": file_path,
                "文件大小": f"{len(original_content)} 字符",
                "行数": len(original_content.split('\n')),
                "编码": "UTF-8"
            }
            log_detailed(self.paper_logger, "📊 文件信息", file_stats)
            log_operation_success(self.paper_logger, "读取原始文件内容")
            
            # 创建备份
            log_operation_start(self.paper_logger, "创建文件备份")
            backup_path = self._create_backup(file_path, iteration, output_dir)
            if backup_path:
                self.paper_logger.info(f"📁 已创建备份: {backup_path}")
                log_operation_success(self.paper_logger, "创建文件备份")
            else:
                self.paper_logger.warning("⚠️  备份创建失败，但继续执行修改")
            
            # 使用论文专用的系统提示词
            log_operation_start(self.paper_logger, "构建论文专用提示词")
            system_prompt = self._build_paper_system_prompt(expected_behavior, paper_guide, additional_content)
            
            # 使用论文专用的用户提示词  
            user_prompt = self._build_paper_user_prompt(
                original_content, raw_code, fixing_plan, expected_behavior, paper_guide, additional_content
            )
            
            # 记录提示词统计
            prompt_stats = {
                "系统提示词长度": f"{len(system_prompt)} 字符",
                "用户提示词长度": f"{len(user_prompt)} 字符",
                "修复计划长度": f"{len(fixing_plan)} 字符",
                "原始代码长度": f"{len(raw_code)} 字符" if raw_code else "0 字符"
            }
            log_detailed(self.paper_logger, "📝 提示词统计", prompt_stats)
            log_operation_success(self.paper_logger, "构建论文专用提示词")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # 调用LLM，使用更保守的参数确保准确性
            log_operation_start(self.paper_logger, "LLM代码生成")
            self.paper_logger.info("🤖 正在调用LLM生成论文级别的代码...")
            log_llm_call(self.paper_logger, "论文代码生成模型", 16384, len(user_prompt))
            
            modified_code = self.call_llm(messages, max_tokens=16384, temperature=0.1)
            log_operation_success(self.paper_logger, "LLM代码生成")
            
            # 清理LLM输出内容，移除markdown标记
            log_operation_start(self.paper_logger, "清理LLM输出内容")
            file_extension = os.path.splitext(file_path)[1]
            cleaned_code = clean_llm_code_output(modified_code, file_extension)
            
            # 验证清理后的内容
            validation_result = validate_code_content(cleaned_code, file_extension)
            if not validation_result["is_valid"]:
                self.paper_logger.warning("⚠️ 代码内容验证失败:")
                for issue in validation_result["issues"]:
                    self.paper_logger.warning(f"   - {issue}")
                for suggestion in validation_result["suggestions"]:
                    self.paper_logger.info(f"   💡 {suggestion}")
                
                # 如果验证失败，尝试使用原始内容但记录警告
                if cleaned_code.strip():  # 如果清理后还有内容，使用清理后的
                    self.paper_logger.warning("⚠️ 使用清理后的内容，但可能存在问题")
                else:  # 如果清理后没有内容，使用原始内容
                    self.paper_logger.warning("⚠️ 使用原始LLM输出，可能包含格式问题")
                    cleaned_code = modified_code
            
            log_operation_success(self.paper_logger, "清理LLM输出内容")
            
            # 记录生成结果统计
            generation_stats = {
                "原始生成长度": f"{len(modified_code)} 字符",
                "清理后长度": f"{len(cleaned_code)} 字符",
                "相对原文件": f"+{len(cleaned_code) - len(original_content)}" if len(cleaned_code) > len(original_content) else f"{len(cleaned_code) - len(original_content)}" + " 字符",
                "温度参数": "0.1 (保守)",
                "生成状态": "成功",
                "内容验证": "通过" if validation_result["is_valid"] else "有警告"
            }
            log_detailed(self.paper_logger, "🎯 代码生成统计", generation_stats)
            
            # 写入修改后的文件
            log_operation_start(self.paper_logger, "保存修改后的文件")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_code)
            
            self.paper_logger.info(f"💾 文件已保存: {file_path} ({len(cleaned_code):,} 字符)")
            log_operation_success(self.paper_logger, "保存修改后的文件")
            
            # 更新modified_code变量为清理后的内容，用于后续分析
            modified_code = cleaned_code
            
            # 分析具体变化
            log_operation_start(self.paper_logger, "分析代码变化")
            changes_made = analyze_code_changes(original_content, modified_code)
            
            if changes_made:
                change_stats = {
                    "变化数量": len(changes_made),
                    "主要变化": ", ".join(changes_made[:3]) if len(changes_made) <= 3 else f"{', '.join(changes_made[:3])}... (共{len(changes_made)}项)"
                }
                log_detailed(self.paper_logger, "🔄 代码变化分析", change_stats)
            else:
                self.paper_logger.warning("⚠️  未检测到明显的代码变化")
            
            log_operation_success(self.paper_logger, "分析代码变化")
            
            # 生成修改概述
            modification_summary = self._generate_paper_modification_summary(
                "论文代码修改", file_path, fixing_plan, modified_code, iteration, changes_made
            )
            
            # 记录最终成功
            log_operation_success(self.paper_logger, f"修改文件 - {os.path.basename(file_path)}")
            self.paper_logger.info(f"✅ 论文代码修改完成: {os.path.basename(file_path)}")
            
            # 记录成功统计
            success_stats = {
                "修改文件": os.path.basename(file_path),
                "备份状态": "已创建" if backup_path else "未创建",
                "变化项数": len(changes_made),
                "迭代次数": iteration,
                "论文编码": "已应用"
            }
            log_detailed(self.paper_logger, "📊 修改完成统计", success_stats)
            
            return {
                "success": True,
                "fixed_code": modified_code,
                "file_path": file_path,
                "action_taken": f"论文复现: 修改文件 {os.path.basename(file_path)}",
                "backup_created": backup_path is not None,
                "modification_summary": modification_summary,
                "iteration": iteration,
                "changes_made": changes_made,
                "paper_coding_applied": True
            }
            
        except Exception as e:
            error_msg = f"修改文件失败: {str(e)}"
            log_operation_error(self.paper_logger, f"修改文件 - {os.path.basename(file_path)}", str(e))
            self.paper_logger.error(f"❌ {error_msg}")
            
            # 记录异常详情
            error_details = {
                "异常类型": type(e).__name__,
                "异常消息": str(e),
                "目标文件": file_path,
                "迭代次数": iteration
            }
            log_detailed(self.paper_logger, "🚨 异常详情", error_details)
            
            return self._create_error_result(file_path, error_msg, iteration)
    
    def _add_new_file(self, file_path: str, fixing_plan: str, iteration: int, 
                     expected_behavior: str = None, paper_guide: str = "", 
                     additional_content: str = "") -> Dict[str, Any]:
        """
        添加新文件 - 论文复现版本
        
        重写父类方法，使用论文专用的代码生成策略
        
        参数:
            file_path (str): 文件路径
            fixing_plan (str): 修复计划
            iteration (int): 迭代次数
            expected_behavior (str): 期望行为
            paper_guide (str): 论文复现指南内容
            additional_content (str): 补充信息内容
        """
        try:
            # 使用论文专用的系统提示词
            system_prompt = self._build_paper_file_creation_prompt(expected_behavior, paper_guide, additional_content)
            
            # 使用论文专用的用户提示词
            user_prompt = self._build_paper_file_user_prompt(fixing_plan, expected_behavior, paper_guide, additional_content)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # 调用LLM，使用更保守的参数
            generated_code = self.call_llm(messages, max_tokens=16384, temperature=0.2)
            
            # 清理LLM输出内容，移除markdown标记
            file_extension = os.path.splitext(file_path)[1]
            cleaned_code = clean_llm_code_output(generated_code, file_extension)
            
            # 验证清理后的内容
            validation_result = validate_code_content(cleaned_code, file_extension)
            if not validation_result["is_valid"]:
                print(f"⚠️ 代码内容验证失败:")
                for issue in validation_result["issues"]:
                    print(f"   - {issue}")
                for suggestion in validation_result["suggestions"]:
                    print(f"   💡 {suggestion}")
                
                # 如果验证失败，尝试使用原始内容但记录警告
                if cleaned_code.strip():  # 如果清理后还有内容，使用清理后的
                    print(f"⚠️ 使用清理后的内容，但可能存在问题")
                else:  # 如果清理后没有内容，使用原始内容
                    print(f"⚠️ 使用原始LLM输出，可能包含格式问题")
                    cleaned_code = generated_code
            
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 写入新文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_code)
            
            print(f"📝 已写入文件: {file_path} ({len(cleaned_code):,} 字符)")
            
            # 生成修改概述
            modification_summary = f"第{iteration}次论文复现: 创建新文件 {os.path.basename(file_path)}"
            
            return {
                "success": True,
                "fixed_code": generated_code,
                "file_path": file_path,
                "action_taken": f"论文复现: 创建文件 {os.path.basename(file_path)}",
                "backup_created": False,
                "modification_summary": modification_summary,
                "iteration": iteration,
                "changes_made": [f"创建新文件: {file_path}"],
                "paper_coding_applied": True
            }
            
        except Exception as e:
            return self._create_error_result(file_path, f"创建文件失败: {str(e)}", iteration)
    
    def _build_paper_expected_behavior(self, task_dict: Dict[str, Any], 
                                     expected_behavior: str, level_code: str, level_name: str) -> str:
        """构建论文专用的期望行为描述"""
        paper_behavior = f"论文复现 - {level_code}层级({level_name}): "
        
        if expected_behavior:
            paper_behavior += expected_behavior
        else:
            paper_behavior += task_dict.get("fixing_plan_in_detail", "改进代码实现")
        
        # 添加层级特定的行为要求
        level_requirements = {
            "L0": "确保程序能够正常运行，依赖完整，环境配置正确",
            "L1": "确保核心算法实现正确，与论文描述一致", 
            "L2": "确保训练推理流程完整，能够端到端运行",
            "L3": "确保所有实验都能正常执行，结果能够输出",
            "L4": "确保实验结果尽可能接近论文报告的数值"
        }
        
        if level_code in level_requirements:
            paper_behavior += f" 特别要求: {level_requirements[level_code]}"
        
        return paper_behavior
    
    def _build_paper_system_prompt(self, expected_behavior: str = None, 
                                  paper_guide: str = "", additional_content: str = "") -> str:
        """构建论文复现专用的系统提示词"""
        
        behavior_section = ""
        if expected_behavior:
            behavior_section = f"""6. 确保修改后的代码符合论文复现的期望行为，期望行为: {expected_behavior}"""
        
        # 构建论文指南部分
        guide_section = ""
        if paper_guide and paper_guide.strip():
            guide_section = f"""
                    
                    论文复现指南参考:
                    请参考以下论文复现指南进行实现，确保与论文算法描述一致：
                    {paper_guide[:2000]}{"..." if len(paper_guide) > 2000 else ""}
                    """
        
        # 构建补充信息部分
        additional_section = ""
        if additional_content and additional_content.strip():
            additional_section = f"""
                    
                    补充实现信息:
                    以下是额外的实现指导和技巧，请在编码时参考：
                    {additional_content[:1500]}{"..." if len(additional_content) > 1500 else ""}
                    """
        
        return f"""你是一个专业的论文复现代码专家。你的任务是修改代码以实现准确的论文复现。

                    论文复现的核心原则:
                    • 算法准确性: 确保实现与论文描述完全一致
                    • 数值精度: 注重科学计算的精度，避免累积误差
                    • 可重现性: 确保实验结果的一致性和可重现性  
                    • 科学严谨: 遵循科学计算的最佳实践
                    • 清晰文档: 添加详细的算法说明和公式注释{guide_section}{additional_section}

                    修改要求:
                    1. 仔细分析原始代码和修复计划
                    2. 参考论文指南和补充信息进行精确实现
                    3. 生成修改后的完整文件内容
                    4. 保持代码结构和风格一致
                    5. 确保修改后的代码是正确的、可运行的
                    6. 添加必要的注释说明修改内容，特别是算法相关部分{behavior_section}

                    特别注意:
                    • 严格按照论文指南中的算法描述实现
                    • 对于数学公式的实现要格外谨慎
                    • 对于模型架构要确保与论文描述一致
                    • 对于超参数设置要参考论文建议
                    • 利用补充信息中的实现技巧和注意事项

                    格式要求：
                    • 直接返回修改后的完整文件内容
                    • 不要使用markdown代码块（如```python或```）
                    • 不要添加任何格式标记或说明文字
                    • 只返回纯Python代码内容
                    • 确保第一行是有效的Python代码（如import语句或注释）"""
    
    def _build_paper_user_prompt(self, original_content: str, raw_code: str, 
                               fixing_plan: str, expected_behavior: str = None,
                               paper_guide: str = "", additional_content: str = "") -> str:
        """构建论文复现专用的用户提示词"""
        
        behavior_section = ""
        if expected_behavior:
            behavior_section = f"""
论文复现目标: {expected_behavior}
请确保修改后的代码能够实现上述论文复现目标。"""
        
        # 构建论文指南参考部分
        guide_reference = ""
        if paper_guide and paper_guide.strip():
            guide_reference = f"""

                    论文指南参考:
                    请仔细参考系统提示中的论文复现指南，确保实现与论文算法完全一致。
                    特别注意论文中的数学公式、算法伪代码和实验设置。"""
        
        # 构建补充信息参考部分
        additional_reference = ""
        if additional_content and additional_content.strip():
            additional_reference = f"""

                    补充实现参考:
                    请参考系统提示中的补充实现信息，利用其中的实现技巧和注意事项。
                    这些信息包含了实践中的最佳做法和常见问题的解决方案。"""
        
        return f"""原始文件内容:
                    ```
                    {original_content}
                    ```

                    原始代码片段（如果提供）:
                    ```
                    {raw_code}
                    ```

                    论文复现修复计划:
                    {fixing_plan}{behavior_section}{guide_reference}{additional_reference}

                    请基于论文复现的要求，生成修改后的完整文件内容。

                    重点关注:
                    1. 严格按照论文指南中的算法描述实现
                    2. 算法实现的正确性和完整性
                    3. 与论文描述的一致性
                    4. 科学计算的精度要求
                    5. 利用补充信息中的实现技巧
                    6. 代码的可读性和可维护性"""
    
    def _build_paper_file_creation_prompt(self, expected_behavior: str = None,
                                          paper_guide: str = "", additional_content: str = "") -> str:
        """构建论文复现专用的文件创建系统提示词"""
        
        behavior_section = ""
        if expected_behavior:
            behavior_section = f"""5. 确保生成的代码符合论文复现的期望行为，期望行为: {expected_behavior}"""
        
        # 构建论文指南部分
        guide_section = ""
        if paper_guide and paper_guide.strip():
            guide_section = f"""
                    
                    论文复现指南参考:
                    请参考以下论文复现指南进行实现，确保与论文算法描述一致：
                    {paper_guide[:2000]}{"..." if len(paper_guide) > 2000 else ""}
                    """
        
        # 构建补充信息部分
        additional_section = ""
        if additional_content and additional_content.strip():
            additional_section = f"""
                    
                    补充实现信息:
                    以下是额外的实现指导和技巧，请在编码时参考：
                    {additional_content[:1500]}{"..." if len(additional_content) > 1500 else ""}
                    """
        
        return f"""你是一个专业的论文复现代码专家。你的任务是创建新的代码文件以实现准确的论文复现。

                    论文复现的代码生成原则:
                    • 算法完整性: 生成完整、正确的算法实现
                    • 论文一致性: 确保与论文描述和公式一致
                    • 科学严谨性: 遵循科学计算的最佳实践
                    • 代码质量: 生成高质量、可维护的代码
                    • 详细注释: 包含算法说明和公式解释{guide_section}{additional_section}

                    生成要求:
                    1. 严格按照论文指南中的算法描述实现
                    2. 生成的代码必须是完整的、可运行的
                    3. 包含必要的注释和文档字符串，特别是算法相关部分
                    4. 遵循Python编码规范和科学计算最佳实践
                    5. 利用补充信息中的实现技巧和注意事项
                    6. 只返回代码内容，不要包含额外的解释{behavior_section}

                    格式要求：
                    • 直接返回完整的代码文件内容
                    • 不要使用markdown代码块（如```python或```）
                    • 不要添加任何格式标记或说明文字
                    • 只返回纯Python代码内容
                    • 确保第一行是有效的Python代码（如import语句或注释）"""
    
    def _build_paper_file_user_prompt(self, fixing_plan: str, expected_behavior: str = None,
                                      paper_guide: str = "", additional_content: str = "") -> str:
        """构建论文复现专用的文件创建用户提示词"""
        
        behavior_section = ""
        if expected_behavior:
            behavior_section = f"""
论文复现目标: {expected_behavior}
请确保生成的代码能够实现上述论文复现目标。"""
        
        # 构建论文指南参考部分
        guide_reference = ""
        if paper_guide and paper_guide.strip():
            guide_reference = f"""

                    论文指南参考:
                    请仔细参考系统提示中的论文复现指南，确保实现与论文算法完全一致。
                    特别注意论文中的数学公式、算法伪代码和实验设置。"""
        
        # 构建补充信息参考部分
        additional_reference = ""
        if additional_content and additional_content.strip():
            additional_reference = f"""

                    补充实现参考:
                    请参考系统提示中的补充实现信息，利用其中的实现技巧和注意事项。
                    这些信息包含了实践中的最佳做法和常见问题的解决方案。"""
        
        return f"""论文复现文件创建计划:
                    {fixing_plan}{behavior_section}{guide_reference}{additional_reference}

                    请生成完整的文件内容。

                    重点要求:
                    1. 严格按照论文指南中的算法描述实现
                    2. 确保算法实现的正确性和完整性
                    3. 添加详细的算法说明注释
                    4. 包含必要的导入和依赖
                    5. 遵循科学计算的最佳实践
                    6. 利用补充信息中的实现技巧
                    7. 确保代码的可读性和可维护性"""
    
    def _generate_paper_modification_summary(self, action_type: str, file_path: str, 
                                           fixing_plan: str, modified_code: str, 
                                           iteration: int, changes_made: list) -> str:
        """生成论文复现专用的修改概述"""
        
        base_summary = f"第{iteration}次论文复现迭代: {action_type} - {os.path.basename(file_path)}"
        
        if changes_made:
            summary_details = f"{base_summary}\n主要修改:\n"
            for i, change in enumerate(changes_made[:3], 1):
                summary_details += f"  {i}. {change}\n"
            if len(changes_made) > 3:
                summary_details += f"  ... 以及其他 {len(changes_made) - 3} 项修改"
        else:
            summary_details = f"{base_summary}\n修改计划: {fixing_plan[:100]}..."
        
        return summary_details
    
    def _get_applied_principles(self, task_dict: Dict[str, Any]) -> list:
        """获取应用的论文复现原则"""
        principles = []
        
        fixing_plan = task_dict.get("fixing_plan_in_detail", "").lower()
        level_code = task_dict.get("level_code", "")
        
        # 根据修复计划和层级判断应用了哪些原则
        if any(keyword in fixing_plan for keyword in ["算法", "公式", "模型"]):
            principles.append("algorithm_accuracy")
        
        if any(keyword in fixing_plan for keyword in ["精度", "数值", "计算"]):
            principles.append("numerical_precision")
        
        if any(keyword in fixing_plan for keyword in ["实验", "结果", "重现"]):
            principles.append("reproducibility")
        
        if level_code in ["L1", "L2"]:
            principles.append("scientific_rigor")
        
        principles.append("documentation")  # 始终包含文档原则
        
        return principles
    
    def _create_backup(self, file_path: str, iteration: int, output_dir: str = None) -> str:
        """创建文件备份"""
        try:
            import shutil
            
            if output_dir:
                backup_dir = os.path.join(output_dir, "backups")
                os.makedirs(backup_dir, exist_ok=True)
                backup_filename = f"{os.path.basename(file_path)}.backup_{iteration}"
                backup_path = os.path.join(backup_dir, backup_filename)
            else:
                file_dir = os.path.dirname(file_path)
                backup_dir = os.path.join(file_dir, ".backups")
                os.makedirs(backup_dir, exist_ok=True)
                backup_filename = f"{os.path.basename(file_path)}.backup_{iteration}"
                backup_path = os.path.join(backup_dir, backup_filename)
            
            shutil.copy2(file_path, backup_path)
            return backup_path
            
        except Exception as e:
            print(f"⚠️  备份创建失败: {str(e)}")
            return None
    
    def _create_error_result(self, file_path: str, error_message: str, iteration: int) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            "success": False,
            "fixed_code": "",
            "file_path": file_path,
            "action_taken": f"论文复现失败: {error_message}",
            "backup_created": False,
            "modification_summary": f"第{iteration}次论文复现失败: {error_message}",
            "iteration": iteration,
            "changes_made": [],
            "paper_coding_applied": False
        } 

if __name__ == "__main__":
    agent = PaperCoderAgent()
    #测试看看提示词的生成效果，用示例original_content和raw_code，以及fixing_plan
    original_content = """
    ------------original content start------------
    
    ------------original content end------------
    """
    raw_code = """
    ------------raw code start------------
    
    ------------raw code end------------
    """
    fixing_plan = """
    ------------fixing plan start------------
    
    ------------fixing plan end------------
    """
    paper_guide = """
    ------------paper guide start------------
    
    ------------paper guide end------------
    """
    additional_content = """
    ------------additional content start------------
    
    ------------additional content end------------
    """
    expected_behavior = """
    ------------expected behavior start------------
    
    ------------expected behavior end------------
    """ 
    print(agent._build_paper_system_prompt(expected_behavior=expected_behavior, paper_guide=paper_guide, additional_content=additional_content))
    print(agent._build_paper_user_prompt(original_content=original_content, raw_code=raw_code, fixing_plan=fixing_plan, expected_behavior=expected_behavior, paper_guide=paper_guide, additional_content=additional_content))
    print(agent._build_paper_file_creation_prompt(expected_behavior=expected_behavior, paper_guide=paper_guide, additional_content=additional_content))
    print(agent._build_paper_file_user_prompt(fixing_plan=fixing_plan, expected_behavior=expected_behavior, paper_guide=paper_guide, additional_content=additional_content))