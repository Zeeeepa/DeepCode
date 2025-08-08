"""
Agent工具模块

提供Agent所需的各种工具和客户端。

主要组件:
- LLMClient: LLM API客户端
- ColoredLogger: 彩色日志系统
- file_tools: 文件操作工具
- json_tools: JSON处理工具
- analysis_tools: 错误分析工具
- execution_tools: 程序执行工具
- code_tools: 代码修改分析工具
"""

from .llm_client import LLMClient
from .colored_logging import (
    get_colored_logger, 
    ColoredLogger, 
    log_detailed, 
    log_llm_call,
    log_operation_start,
    log_operation_success,
    log_operation_error,
    log_function_call
)

# 导入工具模块
from . import file_tools
from . import json_tools
from . import analysis_tools
from . import execution_tools
from . import code_tools

# 导出常用函数以便直接使用
from .file_tools import (
    should_filter_file,
    read_files_completely,
    get_basic_file_list,
    extract_file_from_error,
    estimate_context_usage,
    create_repo_index,
    get_current_timestamp,
    extract_paper_guide_from_markdown,
    load_paper_guide,
    load_additional_guides
)

from .json_tools import (
    parse_json_response,
    try_fix_truncated_json,
    extract_partial_tasks,
    validate_json_structure,
    clean_json_response,
    format_json_for_display
)

from .analysis_tools import (
    analyze_output_patterns,
    validate_and_enhance_tasks,
    generate_execution_plan,
    generate_fallback_result,
    analyze_error_evolution,
    summarize_call_graph
)

from .execution_tools import (
    run_program,
    get_timestamp,
    get_formatted_timestamp,
    display_modification_summary,
    run_program_with_timeout,
    validate_program_path,
    get_execution_environment_info,
    format_execution_result
)

from .code_tools import (
    analyze_code_changes,
    clean_llm_code_output,
    validate_code_content,
    update_modification_history,
    read_modification_history,
    generate_code_summary,
    validate_code_syntax,
    extract_imports,
    extract_functions,
    extract_classes,
    calculate_code_complexity
)

__all__ = [
    # 原有组件
    'LLMClient',
    'get_colored_logger',
    'ColoredLogger',
    'log_detailed',
    'log_llm_call',
    'log_operation_start',
    'log_operation_success',
    'log_operation_error',
    'log_function_call',
    
    # 工具模块
    'file_tools',
    'json_tools', 
    'analysis_tools',
    'execution_tools',
    'code_tools',
    
    # 文件操作工具
    'should_filter_file',
    'read_files_completely',
    'get_basic_file_list',
    'extract_file_from_error',
    'estimate_context_usage',
    'create_repo_index',
    'get_current_timestamp',
    'extract_paper_guide_from_markdown',
    'load_paper_guide',
    'load_additional_guides',
    
    # JSON处理工具
    'parse_json_response',
    'try_fix_truncated_json',
    'extract_partial_tasks',
    'validate_json_structure',
    'clean_json_response',
    'format_json_for_display',
    
    # 分析工具
    'analyze_output_patterns',
    'validate_and_enhance_tasks',
    'generate_execution_plan',
    'generate_fallback_result',
    'analyze_error_evolution',
    'summarize_call_graph',
    
    # 执行工具
    'run_program',
    'get_timestamp',
    'get_formatted_timestamp',
    'display_modification_summary',
    'run_program_with_timeout',
    'validate_program_path',
    'get_execution_environment_info',
    'format_execution_result',
    
    # 代码分析工具
    'analyze_code_changes',
    'clean_llm_code_output',
    'validate_code_content',
    'update_modification_history',
    'read_modification_history',
    'generate_code_summary',
    'validate_code_syntax',
    'extract_imports',
    'extract_functions',
    'extract_classes',
    'calculate_code_complexity'
] 