"""
程序执行工具模块

提供Agent所需的各种程序执行工具函数。

主要功能:
- 程序运行和监控
- 时间戳生成
- 显示和格式化
"""

import os
import subprocess
from datetime import datetime
from typing import Tuple


def run_program(repo_path: str, main_file: str) -> Tuple[str, str, int]:
    """
    运行程序
    
    参数:
        repo_path (str): 代码库路径
        main_file (str): 主程序文件名
    
    返回:
        tuple: (stdout, stderr, return_code)
    """
    program_path = os.path.join(repo_path, main_file)
    
    try:
        # 构建执行命令和工作目录
        if program_path.endswith('.py'):
            work_dir = os.path.dirname(program_path)
            file_name = os.path.basename(program_path)
            if not work_dir:
                work_dir = os.getcwd()
            cmd = ['python', file_name]
            execution_dir = work_dir
        else:
            cmd = [program_path]
            execution_dir = os.path.dirname(program_path)
        
        print(f"执行命令: {' '.join(cmd)}")
        print(f"工作目录: {execution_dir}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=execution_dir
        )
        
        if result.returncode == 0:
            print("程序执行成功")
        else:
            print(f"程序执行返回非零码: {result.returncode}")
        
        print(f"程序返回码: {result.returncode}")
        print(f"标准输出长度: {len(result.stdout)} 字符")
        
        if result.stderr:
            print(f"错误输出stderr: {result.stderr}")
        
        return result.stdout, result.stderr, result.returncode
        
    except subprocess.TimeoutExpired:
        return "", "程序执行超时", -1
    except FileNotFoundError:
        return "", f"找不到程序文件: {program_path}", -1
    except Exception as e:
        return "", f"执行程序时出现异常: {str(e)}", -1


def get_timestamp() -> str:
    """获取当前时间戳"""
    return datetime.now().isoformat()


def get_formatted_timestamp() -> str:
    """获取格式化的时间戳用于显示"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def display_modification_summary(modification_summary: str, changes_made: list, iteration: int) -> None:
    """
    显示修改概述
    
    参数:
        modification_summary (str): 修改概述
        changes_made (list): 具体修改列表
        iteration (int): 迭代次数
    """
    print("\n" + "="*60)
    print(f"📝 修改概述 (第{iteration}次迭代)")
    print("="*60)
    print(f"概述: {modification_summary}")
    
    if changes_made:
        print("\n具体修改:")
        for i, change in enumerate(changes_made, 1):
            print(f"  {i}. {change}")
    
    print("="*60)


def run_program_with_timeout(repo_path: str, main_file: str, timeout: int = 30) -> Tuple[str, str, int]:
    """
    带超时的程序运行
    
    参数:
        repo_path (str): 代码库路径
        main_file (str): 主程序文件名
        timeout (int): 超时时间（秒）
    
    返回:
        tuple: (stdout, stderr, return_code)
    """
    program_path = os.path.join(repo_path, main_file)
    
    try:
        # 构建执行命令和工作目录
        if program_path.endswith('.py'):
            work_dir = os.path.dirname(program_path)
            file_name = os.path.basename(program_path)
            if not work_dir:
                work_dir = os.getcwd()
            cmd = ['python', file_name]
            execution_dir = work_dir
        else:
            cmd = [program_path]
            execution_dir = os.path.dirname(program_path)
        
        print(f"执行命令: {' '.join(cmd)} (超时: {timeout}秒)")
        print(f"工作目录: {execution_dir}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=execution_dir
        )
        
        return result.stdout, result.stderr, result.returncode
        
    except subprocess.TimeoutExpired:
        return "", f"程序执行超时 ({timeout}秒)", -1
    except FileNotFoundError:
        return "", f"找不到程序文件: {program_path}", -1
    except Exception as e:
        return "", f"执行程序时出现异常: {str(e)}", -1


def validate_program_path(repo_path: str, main_file: str) -> bool:
    """
    验证程序路径是否有效
    
    参数:
        repo_path (str): 代码库路径
        main_file (str): 主程序文件名
    
    返回:
        bool: 路径是否有效
    """
    if not repo_path or not main_file:
        return False
    
    program_path = os.path.join(repo_path, main_file)
    
    # 检查文件是否存在
    if not os.path.exists(program_path):
        return False
    
    # 检查是否是文件
    if not os.path.isfile(program_path):
        return False
    
    # 检查文件扩展名
    if not program_path.endswith(('.py', '.js', '.java', '.cpp', '.c')):
        return False
    
    return True


def get_execution_environment_info() -> dict:
    """
    获取执行环境信息
    
    返回:
        dict: 环境信息
    """
    try:
        python_version = subprocess.run(
            ['python', '--version'], 
            capture_output=True, 
            text=True
        ).stdout.strip()
    except:
        python_version = "未知"
    
    return {
        "python_version": python_version,
        "working_directory": os.getcwd(),
        "timestamp": get_formatted_timestamp(),
        "platform": os.name
    }


def format_execution_result(stdout: str, stderr: str, return_code: int) -> dict:
    """
    格式化执行结果
    
    参数:
        stdout (str): 标准输出
        stderr (str): 错误输出
        return_code (int): 返回码
    
    返回:
        dict: 格式化的结果
    """
    return {
        "success": return_code == 0,
        "return_code": return_code,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_length": len(stdout),
        "stderr_length": len(stderr),
        "has_output": bool(stdout.strip()),
        "has_error": bool(stderr.strip()),
        "timestamp": get_timestamp()
    } 

if __name__ == "__main__":
    #测试执行/Users/wwchdemac/python_projects/debug_agent/complex_library_system /main.py，检查终端输出是否正确
    repo_path = "/Users/wwchdemac/python_projects/debug_agent/complex_library_system"
    main_file = "main.py"
    stdout, stderr, return_code = run_program(repo_path, main_file)
    print(f"stdout: {stdout}")
    print(f"stderr: {stderr}")
    print(f"return_code: {return_code}")