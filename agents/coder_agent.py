"""
修复Agent

根据任务字典执行代码修复，包括文件的添加和修改操作。

主要功能:
- fix_code(): 根据任务字典执行代码修复
- _generate_modification_summary(): 生成修改概述
"""

import os
import json
import shutil
from typing import Dict, Any
from .base_agent import BaseAgent
from .utils import (
    code_tools,
    analyze_code_changes,
    clean_llm_code_output,
    validate_code_content,
    update_modification_history
)


class CoderAgent(BaseAgent):
    """
    修复Agent
    
    根据Analyzer生成的任务字典执行具体的代码修复操作。
    能够添加新文件、修改现有文件，并确保代码上下文的正确性。
    """
    
    def __init__(self, **kwargs):
        """
        初始化修复Agent
        
        参数:
            **kwargs: 配置参数
        """
        super().__init__(**kwargs)

    def fix_code(self, task_dict: Dict[str, Any], repo_path: str, iteration: int = 1, output_dir: str = None, expected_behavior: str = None) -> Dict[str, Any]:
        """
        根据任务字典执行代码修复
        
        参数:
            task_dict (Dict[str, Any]): 从Analyzer获得的任务字典
            repo_path (str): 仓库路径
            iteration (int): 迭代次数，用于生成修改概述
            output_dir (str): 输出目录，用于保存修改历史
            expected_behavior (str, optional): 期望的程序行为描述
        
        返回:
            dict: 修复结果
            {
                "success": bool,
                "fixed_code": str,
                "file_path": str,
                "action_taken": str,
                "backup_created": bool,
                "modification_summary": str,  # 新增：修改概述
                "iteration": int,             # 新增：迭代次数
                "changes_made": list          # 新增：具体修改列表
            }
        """
        try:
            fixing_type = task_dict.get("fixing_type", "")
            target_file = task_dict.get("which_file_to_fix", "")
            fixing_plan = task_dict.get("fixing_plan_in_detail", "")
            raw_code = task_dict.get("raw_code", "")
            
            if not target_file:
                return {
                    "success": False,
                    "fixed_code": "",
                    "file_path": "",
                    "action_taken": "错误：未指定目标文件",
                    "backup_created": False,
                    "modification_summary": f"第{iteration}次修复失败：未指定目标文件",
                    "iteration": iteration,
                    "changes_made": []
                }
            
            # 确保文件路径是相对于repo_path的
            if os.path.isabs(target_file):
                file_path = target_file
            else:
                file_path = os.path.join(repo_path, target_file)
            
            result = {
                "success": False,
                "fixed_code": "",
                "file_path": file_path,
                "action_taken": "",
                "backup_created": False,
                "modification_summary": "",
                "iteration": iteration,
                "changes_made": []
            }
            
            if fixing_type.lower() == "add_file":
                result = self._add_new_file(file_path, fixing_plan, iteration, expected_behavior)
            elif fixing_type.lower() == "change_file":
                result = self._modify_existing_file(file_path, fixing_plan, raw_code, iteration, output_dir, expected_behavior)
            else:
                result["action_taken"] = f"未知的修复类型: {fixing_type}"
                result["modification_summary"] = f"第{iteration}次修复失败：未知修复类型 {fixing_type}"
            
            # 如果成功且有输出目录，更新修改历史
            if result["success"] and output_dir:
                update_modification_history(output_dir, result)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "fixed_code": "",
                "file_path": target_file,
                "action_taken": f"修复过程中出现异常: {str(e)}",
                "backup_created": False,
                "modification_summary": f"第{iteration}次修复异常：{str(e)}",
                "iteration": iteration,
                "changes_made": []
            }

    def _add_new_file(self, file_path: str, fixing_plan: str, iteration: int, expected_behavior: str = None) -> Dict[str, Any]:
        """
        添加新文件
        
        参数:
            file_path (str): 文件路径
            fixing_plan (str): 修复计划
            iteration (int): 迭代次数
            expected_behavior (str, optional): 期望的程序行为描述
        
        返回:
            dict: 添加结果
        """
        try:
            # 使用LLM生成文件内容
            expected_behavior_section = ""
            if expected_behavior:
                expected_behavior_section = f"""
                                5. 确保生成的代码符合期望的程序行为
                                
                                期望的程序行为：{expected_behavior}
                                """
            
            system_prompt = f"""你是一个专业的论文复现生成专家。根据修复计划生成完整的代码文件内容。

                                要求：
                                1. 生成的代码必须是完整的、可运行的
                                2. 包含必要的注释和文档字符串  
                                3. 遵循Python编码规范
                                4. 只返回代码内容，不要包含额外的解释{expected_behavior_section}

                                格式要求：
                                • 直接返回完整的代码文件内容
                                • 不要使用markdown代码块（如```python或```）
                                • 不要添加任何格式标记或说明文字
                                • 只返回纯代码内容
                                • 确保第一行是有效的代码（如import语句或注释）"""

            expected_behavior_prompt = ""
            if expected_behavior:
                expected_behavior_prompt = f"""
                                期望的程序行为：
                                {expected_behavior}
                                
                                请确保生成的代码能够实现上述期望行为。
                                """

            user_prompt = f"""修复计划：
                                {fixing_plan}
                                {expected_behavior_prompt}
                                请生成完整的文件内容。"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            generated_code = self.call_llm(messages, max_tokens=16384, temperature=0.3)
            
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
            
            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_code)
            
            print(f"📝 已写入文件: {file_path} ({len(cleaned_code):,} 字符)")
            
            # 更新generated_code变量为清理后的内容，用于后续分析
            generated_code = cleaned_code
            
            # 生成修改概述
            modification_summary = self._generate_modification_summary(
                "新增文件", file_path, fixing_plan, generated_code, iteration
            )
            
            return {
                "success": True,
                "fixed_code": generated_code,
                "file_path": file_path,
                "action_taken": f"添加文件: {os.path.basename(file_path)}",
                "backup_created": False,
                "modification_summary": modification_summary,
                "iteration": iteration,
                "changes_made": [f"新增文件: {os.path.basename(file_path)}"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "fixed_code": "",
                "file_path": file_path,
                "action_taken": f"添加文件失败: {str(e)}",
                "backup_created": False,
                "modification_summary": f"第{iteration}次修复失败：添加文件时出现异常 - {str(e)}",
                "iteration": iteration,
                "changes_made": []
            }

    def _modify_existing_file(self, file_path: str, fixing_plan: str, raw_code: str, iteration: int, output_dir: str = None, expected_behavior: str = None) -> Dict[str, Any]:
        """
        修改现有文件
        
        参数:
            file_path (str): 文件路径
            fixing_plan (str): 修复计划
            raw_code (str): 原始代码
            iteration (int): 迭代次数
            output_dir (str): 输出目录路径
            expected_behavior (str, optional): 期望的程序行为描述
        
        返回:
            dict: 修改结果
        """
        try:
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "fixed_code": "",
                    "file_path": file_path,
                    "action_taken": f"文件不存在: {file_path}",
                    "backup_created": False,
                    "modification_summary": f"第{iteration}次修复失败：目标文件不存在",
                    "iteration": iteration,
                    "changes_made": []
                }
            
            # 读取现有文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # 创建备份到专门的备份目录
            backup_dir = None
            backup_path = None
            
            if output_dir:
                # 在output_dir中创建backups子目录
                backup_dir = os.path.join(output_dir, "backups")
                os.makedirs(backup_dir, exist_ok=True)
                
                # 生成备份文件名（保持原始文件的目录结构）
                relative_path = os.path.relpath(file_path)
                backup_filename = f"{os.path.basename(file_path)}.backup_{iteration}"
                backup_path = os.path.join(backup_dir, backup_filename)
            else:
                # 备用方案：在原文件目录创建.backups子目录
                file_dir = os.path.dirname(file_path)
                backup_dir = os.path.join(file_dir, ".backups")
                os.makedirs(backup_dir, exist_ok=True)
                backup_filename = f"{os.path.basename(file_path)}.backup_{iteration}"
                backup_path = os.path.join(backup_dir, backup_filename)
            
            # 执行备份
            shutil.copy2(file_path, backup_path)
            print(f"📁 已创建备份: {backup_path}")
            
            # 使用LLM生成修改后的代码
            expected_behavior_section = ""
            if expected_behavior:
                expected_behavior_section = f"""
                                6. 确保修改后的代码符合期望的程序行为
                                
                                期望的程序行为：{expected_behavior}
                                """
            
            system_prompt = f"""你是一个专业的代码修复专家。根据修复计划修改现有代码。

                                要求：
                                1. 仔细分析原始代码和修复计划
                                2. 生成修改后的完整文件内容
                                3. 保持代码结构和风格一致
                                4. 确保修改后的代码是正确的、可运行的
                                5. 添加必要的注释说明修改内容{expected_behavior_section}

                                格式要求：
                                • 直接返回修改后的完整文件内容
                                • 不要使用markdown代码块（如```python或```）
                                • 不要添加任何格式标记或说明文字
                                • 只返回纯代码内容
                                • 确保第一行是有效的代码（如import语句或注释）"""

            expected_behavior_prompt = ""
            if expected_behavior:
                expected_behavior_prompt = f"""
                                期望的程序行为：
                                {expected_behavior}
                                
                                请确保修改后的代码能够实现上述期望行为。
                                """

            user_prompt = f"""原始文件内容：
                                ```
                                {original_content}
                                ```

                                原始代码片段（如果提供）：
                                ```
                                {raw_code}
                                ```

                                修复计划：
                                {fixing_plan}
                                {expected_behavior_prompt}
                                请生成修改后的完整文件内容。"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            modified_code = self.call_llm(messages, max_tokens=16384, temperature=0.2)
            
            # 清理LLM输出内容，移除markdown标记
            file_extension = os.path.splitext(file_path)[1]
            cleaned_code = clean_llm_code_output(modified_code, file_extension)
            
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
                    cleaned_code = modified_code
            
            # 写入修改后的文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_code)
            
            print(f"📝 已写入文件: {file_path} ({len(cleaned_code):,} 字符)")
            
            # 更新modified_code变量为清理后的内容，用于后续分析
            modified_code = cleaned_code
            
            # 分析具体变化
            changes_made = analyze_code_changes(original_content, modified_code)
            
            # 生成修改概述
            modification_summary = self._generate_modification_summary(
                "修改文件", file_path, fixing_plan, modified_code, iteration, changes_made
            )
            
            return {
                "success": True,
                "fixed_code": modified_code,
                "file_path": file_path,
                "action_taken": f"修改文件: {os.path.basename(file_path)}",
                "backup_created": True,
                "modification_summary": modification_summary,
                "iteration": iteration,
                "changes_made": changes_made
            }
            
        except Exception as e:
            return {
                "success": False,
                "fixed_code": "",
                "file_path": file_path,
                "action_taken": f"修改文件失败: {str(e)}",
                "backup_created": False,
                "modification_summary": f"第{iteration}次修复失败：修改文件时出现异常 - {str(e)}",
                "iteration": iteration,
                "changes_made": []
            }

    def _generate_modification_summary(self, action_type: str, file_path: str, 
                                     fixing_plan: str, code_content: str, 
                                     iteration: int, changes_made: list = None) -> str:
        """
        生成修改概述
        
        参数:
            action_type (str): 操作类型 (新增文件/修改文件)
            file_path (str): 文件路径
            fixing_plan (str): 修复计划
            code_content (str): 代码内容
            iteration (int): 迭代次数
            changes_made (list): 具体变化列表
        
        返回:
            str: 修改概述
        """
        try:
            system_prompt = """你是一个代码修改总结专家。请为代码修改生成简洁的概述。

                                要求：
                                1. 用一句话总结主要修改内容
                                2. 重点关注错误处理改进、逻辑优化、代码质量提升
                                3. 如果是异常处理相关的改进，请特别说明
                                4. 保持概述简洁明了（不超过100字）

                                格式：直接返回概述内容，不要添加额外的格式。"""

            user_prompt = f"""操作类型：{action_type}
                                文件：{os.path.basename(file_path)}
                                修复计划：{fixing_plan}
                                迭代次数：第{iteration}次"""

            if changes_made:
                user_prompt += f"\n具体变化：{', '.join(changes_made)}"

            user_prompt += "\n\n请生成精准干练的修改概述。"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            summary = self.call_llm(messages, max_tokens=16384, temperature=0.3)
            
            # 格式化最终概述
            formatted_summary = f"第{iteration}次修复：{summary.strip()}"
            
            return formatted_summary
            
        except Exception as e:
            return f"第{iteration}次修复：{action_type} {os.path.basename(file_path)} - 概述生成失败：{str(e)}"

    def process(self, input_data: Any) -> Any:
        """
        处理输入数据（实现基类抽象方法）
        
        参数:
            input_data: 输入数据，应该是包含任务字典的字典
        
        返回:
            dict: 修复结果
        """
        if isinstance(input_data, dict):
            task_dict = input_data.get("task_dict", {})
            repo_path = input_data.get("repo_path", "")
            iteration = input_data.get("iteration", 1)
            output_dir = input_data.get("output_dir")
            return self.fix_code(task_dict, repo_path, iteration, output_dir)
        else:
            return {
                "success": False,
                "fixed_code": "",
                "file_path": "",
                "action_taken": "无效的输入数据格式",
                "backup_created": False,
                "modification_summary": "输入格式错误",
                "iteration": 0,
                "changes_made": []
            } 


    # def _analyze_code_changes(self, original: str, modified: str) -> list:
    #     """
    #     分析代码变化
        
    #     参数:
    #         original (str): 原始代码
    #         modified (str): 修改后代码
        
    #     返回:
    #         list: 变化列表
    #     """
    #     changes = []
        
    #     try:
    #         original_lines = original.splitlines()
    #         modified_lines = modified.splitlines()
            
    #         # 简单的变化检测
    #         if len(modified_lines) > len(original_lines):
    #             changes.append(f"增加了 {len(modified_lines) - len(original_lines)} 行代码")
    #         elif len(modified_lines) < len(original_lines):
    #             changes.append(f"删除了 {len(original_lines) - len(modified_lines)} 行代码")
            
    #         # 检测特定关键词的变化
    #         keywords = ['def ', 'class ', 'try:', 'except', 'if ', 'ZeroDivisionError', 'Exception']
    #         for keyword in keywords:
    #             original_count = original.count(keyword)
    #             modified_count = modified.count(keyword)
    #             if modified_count > original_count:
    #                 changes.append(f"新增 {keyword.strip()} 相关代码")
    #             elif modified_count < original_count:
    #                 changes.append(f"移除 {keyword.strip()} 相关代码")
            
    #         # 检测异常处理改进
    #         if 'ZeroDivisionError' in modified and 'ZeroDivisionError' not in original:
    #             changes.append("添加了专门的除零异常处理")
            
    #         if modified.count('try:') > original.count('try:'):
    #             changes.append("增强了异常处理机制")
            
    #         if modified.count('def ') > original.count('def '):
    #             changes.append("新增了函数定义")
            
    #     except Exception:
    #         changes.append("代码结构发生了变化")
        
    #     return changes if changes else ["修改了文件内容"]

    # def _update_modification_history(self, output_dir: str, result: Dict[str, Any]) -> None:
    #     """
    #     更新修改历史记录
        
    #     参数:
    #         output_dir (str): 输出目录
    #         result (Dict[str, Any]): 修复结果
    #     """
    #     try:
    #         history_file = os.path.join(output_dir, "modification_history.json")
            
    #         # 读取现有历史
    #         if os.path.exists(history_file):
    #             with open(history_file, 'r', encoding='utf-8') as f:
    #                 history = json.load(f)
    #         else:
    #             history = {
    #                 "total_iterations": 0,
    #                 "modifications": []
    #             }
            
    #         # 添加新的修改记录
    #         modification_record = {
    #             "iteration": result["iteration"],
    #             "timestamp": str(__import__('datetime').datetime.now()),
    #             "file_path": result["file_path"],
    #             "action_taken": result["action_taken"],
    #             "modification_summary": result["modification_summary"],
    #             "changes_made": result["changes_made"],
    #             "success": result["success"]
    #         }
            
    #         history["modifications"].append(modification_record)
    #         history["total_iterations"] = max(history["total_iterations"], result["iteration"])
            
    #         # 保存更新后的历史
    #         with open(history_file, 'w', encoding='utf-8') as f:
    #             json.dump(history, f, indent=2, ensure_ascii=False)
                
    #     except Exception as e:
    #         print(f"更新修改历史失败: {str(e)}")
