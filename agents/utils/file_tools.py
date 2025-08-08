"""
文件操作工具模块

提供Agent所需的各种文件处理工具函数。

主要功能:
- 文件过滤和筛选
- 文件内容读取
- 目录结构分析
- 文件路径提取
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple


def should_filter_file(file_path: str) -> bool:
    """
    判断是否应该过滤文件
    
    参数:
        file_path (str): 文件路径
    
    返回:
        bool: True表示应该过滤，False表示可以包含
    """
    # 过滤备份文件
    if '.backup' in file_path or file_path.endswith('.bak'):
        return True
    
    # 过滤调试输出
    if 'debug_output' in file_path or 'debug_report' in file_path:
        return True
    
    # 过滤隐藏文件
    if file_path.startswith('.') or '/.git/' in file_path:
        return True
    
    # 过滤编译文件
    if '__pycache__' in file_path or file_path.endswith('.pyc'):
        return True
    
    # 过滤日志文件
    if file_path.endswith('.log') or '/logs/' in file_path:
        return True
    
    # 过滤虚拟环境
    if '/venv/' in file_path or '/env/' in file_path:
        return True
    
    return False


def read_files_completely(files_to_read: List[Dict], repo_path: str) -> Dict[str, Any]:
    """
    完整读取文件内容
    
    参数:
        files_to_read (List[Dict]): 需要读取的文件列表
        repo_path (str): 仓库路径
    
    返回:
        dict: 文件读取结果
    """
    file_contents = {}
    
    try:
        for file_info in files_to_read:
            file_path = file_info.get("file_path", "")
            
            if not file_path:
                continue
            
            # 过滤不合适的文件
            if should_filter_file(file_path):
                print(f"⚠️ 跳过被过滤的文件: {file_path}")
                continue
            
            full_path = os.path.join(repo_path, file_path)
            
            if not os.path.exists(full_path):
                print(f"⚠️ 文件不存在: {file_path}")
                continue
            
            if not os.path.isfile(full_path):
                print(f"⚠️ 不是文件: {file_path}")
                continue
            
            try:
                # 检查文件大小，避免读取过大的文件
                file_size = os.path.getsize(full_path)
                if file_size > 500 * 1024:  # 500KB限制
                    print(f"⚠️ 文件过大，跳过: {file_path} ({file_size} bytes)")
                    continue
                
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    file_contents[file_path] = {
                        "content": content,
                        "size": len(content),
                        "lines": len(content.splitlines()),
                        "priority": file_info.get("priority", "medium"),
                        "analysis_focus": file_info.get("analysis_focus", ""),
                        "reason": file_info.get("reason", "")
                    }
                    print(f"✅ 读取文件: {file_path} ({len(content)} 字符)")
                    
            except UnicodeDecodeError:
                print(f"⚠️ 文件编码错误，跳过: {file_path}")
                continue
            except Exception as e:
                print(f"⚠️ 读取文件失败: {file_path} - {str(e)}")
                continue
        
        return {
            "file_contents": file_contents,
            "files_read": len(file_contents),
            "total_chars": sum(info["size"] for info in file_contents.values()),
            "success": True
        }
        
    except Exception as e:
        return {
            "file_contents": {},
            "error": f"文件读取过程异常: {str(e)}",
            "success": False
        }


def get_basic_file_list(repo_path: str) -> list:
    """
    获取基本的文件列表
    
    参数:
        repo_path (str): 代码库路径
    
    返回:
        list: 文件列表
    """
    try:
        files = []
        for root, dirs, filenames in os.walk(repo_path):
            # 排除隐藏目录
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            level = root.replace(repo_path, '').count(os.sep)
            indent = '  ' * level
            rel_root = os.path.relpath(root, repo_path)
            
            if rel_root != '.':
                files.append(f"{indent}{os.path.basename(root)}/")
            
            sub_indent = '  ' * (level + 1)
            for filename in filenames:
                if not filename.startswith('.'):
                    files.append(f"{sub_indent}{filename}")
        
        return files
        
    except Exception as e:
        return [f"错误: 无法列出文件 - {str(e)}"]


def extract_file_from_error(stdout: str, repo_path: str) -> str:
    """
    从错误信息中提取文件路径
    
    参数:
        stdout (str): 错误输出信息
        repo_path (str): 仓库路径
    
    返回:
        str: 提取到的文件路径
    """
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


def estimate_context_usage(file_contents: Dict) -> Dict:
    """
    估算上下文使用量
    
    参数:
        file_contents (Dict): 文件内容字典
    
    返回:
        Dict: 估算结果
    """
    total_chars = sum(info["size"] for info in file_contents.values())
    total_lines = sum(info["lines"] for info in file_contents.values())
    
    # 简单估算token数量 (大约4个字符 = 1个token)
    estimated_tokens = total_chars // 4
    
    return {
        "total_files": len(file_contents),
        "total_characters": total_chars,
        "total_lines": total_lines,
        "estimated_tokens": estimated_tokens,
        "context_utilization": min(100, (estimated_tokens / 100000) * 100)  # 假设最大100k tokens
    }


def create_repo_index(repo_path: str, output_dir: str) -> str:
    """
    创建代码库索引
    
    参数:
        repo_path (str): 代码库路径
        output_dir (str): 输出目录
    
    返回:
        str: 索引文件路径
    """
    try:
        # 使用现有的core_modules进行索引
        from core_modules import SimpleStructureAnalyzer, PyDepsAnalyzer
        
        # 1. 先生成项目结构分析
        structure_analyzer = SimpleStructureAnalyzer()
        structure_result = structure_analyzer.analyze_project_structure(repo_path)
        
        # 2. 生成依赖关系分析
        deps_analyzer = PyDepsAnalyzer()
        deps_result = deps_analyzer.analyze_dependencies(repo_path)
        
        # 3. 组合结果
        combined_index = {
            "timestamp": get_current_timestamp(),
            "project_name": os.path.basename(repo_path),
            "structure_analysis": structure_result,
            "dependency_analysis": deps_result
        }
        
        # 4. 保存索引文件
        index_path = os.path.join(output_dir, "repo_index.json")
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(combined_index, f, indent=2, ensure_ascii=False)
        
        print(f"完成组合分析: 结构 + 依赖关系")
        return index_path
        
    except Exception as e:
        print(f"创建代码库索引失败: {str(e)}")
        return ""


def get_current_timestamp() -> str:
    """获取当前时间戳"""
    from datetime import datetime
    return datetime.now().isoformat() 


def extract_paper_guide_from_markdown(markdown_path: str) -> Dict[str, Any]:
    """
    从markdown文件中提取paper_guide内容
    
    参数:
        markdown_path (str): markdown文件路径
    
    返回:
        Dict[str, Any]: 包含提取结果的字典
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(markdown_path):
            raise FileNotFoundError(f"Markdown文件不存在: {markdown_path}")
        
        # 检查是否为markdown文件
        if not markdown_path.lower().endswith(('.md', '.markdown')):
            raise ValueError(f"不是有效的Markdown文件: {markdown_path}")
        
        # 读取文件内容
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 基本信息统计
        lines = content.splitlines()
        word_count = len(content.split())
        char_count = len(content)
        
        # 提取标题信息（可选）
        headers = []
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                headers.append(line)
        
        return {
            "success": True,
            "paper_guide": content,  # 完整的markdown内容
            "metadata": {
                "file_path": markdown_path,
                "file_name": os.path.basename(markdown_path),
                "char_count": char_count,
                "word_count": word_count,
                "line_count": len(lines),
                "headers": headers[:10]  # 最多显示前10个标题
            }
        }
        
    except FileNotFoundError as e:
        return {
            "success": False,
            "error": f"文件未找到: {str(e)}",
            "paper_guide": "",
            "metadata": {}
        }
    except UnicodeDecodeError as e:
        return {
            "success": False,
            "error": f"文件编码错误，无法读取: {str(e)}",
            "paper_guide": "",
            "metadata": {}
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"提取markdown内容失败: {str(e)}",
            "paper_guide": "",
            "metadata": {}
        }


def load_paper_guide(guide_source: str) -> str:
    """
    加载paper_guide内容（支持文件路径或直接内容）
    
    参数:
        guide_source (str): markdown文件路径或直接的文本内容
    
    返回:
        str: paper_guide内容
    """
    # 如果是文件路径
    if guide_source.strip().endswith(('.md', '.markdown')) and os.path.exists(guide_source.strip()):
        result = extract_paper_guide_from_markdown(guide_source.strip())
        if result["success"]:
            print(f"✅ 成功从markdown文件加载paper_guide: {result['metadata']['file_name']}")
            print(f"   文件信息: {result['metadata']['char_count']} 字符, {result['metadata']['line_count']} 行")
            return result["paper_guide"]
        else:
            print(f"❌ 加载markdown文件失败: {result['error']}")
            return ""
    
    # 否则当作直接内容返回
    return guide_source


def load_additional_guides(guide_paths: List[str]) -> Dict[str, Any]:
    """
    加载多个补充信息文档并拼接成统一内容
    
    用于论文复现系统中的补充信息处理，将多个markdown文档拼接
    成一个完整的补充信息字符串，方便传递给Analyzer和Coder。
    
    参数:
        guide_paths (List[str]): markdown文档路径列表
    
    返回:
        Dict[str, Any]: 包含以下键值的字典:
            - success (bool): 是否成功处理所有文档
            - additional_content (str): 拼接后的补充信息内容
            - processed_files (List[str]): 成功处理的文件列表
            - failed_files (List[Dict]): 处理失败的文件及错误信息
            - metadata (Dict): 整体统计信息
    """
    if not guide_paths:
        return {
            "success": True,
            "additional_content": "",
            "processed_files": [],
            "failed_files": [],
            "metadata": {
                "total_files": 0,
                "total_char_count": 0,
                "total_line_count": 0
            }
        }
    
    processed_files = []
    failed_files = []
    content_parts = []
    total_char_count = 0
    total_line_count = 0
    
    # 添加补充信息开始标记
    content_parts.append("=== 补充信息汇总 ===\n")
    
    # 逐个处理每个文档
    for i, guide_path in enumerate(guide_paths, 1):
        try:
            # 验证路径格式和文件存在性
            guide_path = guide_path.strip()
            if not guide_path.endswith(('.md', '.markdown')):
                failed_files.append({
                    "file_path": guide_path,
                    "error": "文件格式不支持，仅支持.md和.markdown文件"
                })
                continue
            
            if not os.path.exists(guide_path):
                failed_files.append({
                    "file_path": guide_path,
                    "error": "文件不存在"
                })
                continue
            
            # 提取文档内容
            result = extract_paper_guide_from_markdown(guide_path)
            if result["success"]:
                # 成功提取内容
                file_name = result['metadata']['file_name']
                content = result['paper_guide']
                char_count = result['metadata']['char_count']
                line_count = result['metadata']['line_count']
                
                # 添加文档分隔标记和内容
                content_parts.append(f"\n## 补充文档{i}: {file_name}\n")
                content_parts.append(f"<!-- 来源: {guide_path} -->\n")
                content_parts.append(content)
                content_parts.append("\n")
                
                # 记录成功处理的文件
                processed_files.append({
                    "file_path": guide_path,
                    "file_name": file_name,
                    "char_count": char_count,
                    "line_count": line_count
                })
                
                # 累计统计
                total_char_count += char_count
                total_line_count += line_count
                
                print(f"✅ 成功加载补充文档{i}: {file_name} ({char_count}字符, {line_count}行)")
                
            else:
                # 提取失败
                failed_files.append({
                    "file_path": guide_path,
                    "error": result['error']
                })
                print(f"❌ 加载补充文档{i}失败: {guide_path} - {result['error']}")
                
        except Exception as e:
            # 处理异常
            failed_files.append({
                "file_path": guide_path,
                "error": f"处理异常: {str(e)}"
            })
            print(f"❌ 处理补充文档{i}时出现异常: {guide_path} - {str(e)}")
    
    # 添加补充信息结束标记
    content_parts.append("\n=== 补充信息结束 ===\n")
    
    # 拼接所有内容
    additional_content = "\n".join(content_parts)
    
    # 计算最终统计
    final_char_count = len(additional_content)
    final_line_count = len(additional_content.splitlines())
    
    # 判断整体是否成功
    success = len(processed_files) > 0  # 至少有一个文件成功处理
    
    # 输出处理结果摘要
    if success:
        print(f"📋 补充信息处理完成:")
        print(f"   成功处理: {len(processed_files)}/{len(guide_paths)} 个文件")
        print(f"   总内容: {final_char_count} 字符, {final_line_count} 行")
        if failed_files:
            print(f"   处理失败: {len(failed_files)} 个文件")
    else:
        print(f"❌ 补充信息处理失败: 所有 {len(guide_paths)} 个文件都无法处理")
    
    return {
        "success": success,
        "additional_content": additional_content if success else "",
        "processed_files": processed_files,
        "failed_files": failed_files,
        "metadata": {
            "total_files": len(guide_paths),
            "processed_count": len(processed_files),
            "failed_count": len(failed_files),
            "total_char_count": total_char_count,
            "total_line_count": total_line_count,
            "final_char_count": final_char_count,
            "final_line_count": final_line_count
        }
    }


if __name__ == "__main__":
    #测试获得文件列表
    repo_path = "/Users/zhaoyu/Desktop/test_input/webpage"
    file_list = get_basic_file_list(repo_path)
    print(file_list)

    # 测试markdown提取功能
    test_md_path = "test_papers/paper_test_1_reproduction_guide.md"
    if os.path.exists(test_md_path):
        result = extract_paper_guide_from_markdown(test_md_path)
        if result["success"]:
            print("✅ Markdown提取测试成功")
            print(f"内容长度: {len(result['paper_guide'])} 字符")
            print(f"标题数量: {len(result['metadata']['headers'])}")
            print(result['paper_guide'])
        else:
            print(f"❌ Markdown提取测试失败: {result['error']}")
    else:
        print("⚠️ 测试markdown文件不存在")