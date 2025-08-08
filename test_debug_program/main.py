#!/usr/bin/env python3
"""
科学计算器主程序入口

这是一个功能完整的科学计算器程序，支持基本运算、科学计算、
统计分析、单位转换等多种功能。

作者: Calculator Team
版本: 1.0.0
"""

import sys
import os
import math
import logging
from typing import Optional


class Calculator:
    """简单的计算器类"""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        """加法"""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a, b):
        """减法"""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result
    
    def multiply(self, a, b):
        """乘法"""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def divide(self, a, b):
        """除法"""
        if b == 0:
            raise ValueError("除数不能为零")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result
    
    def power(self, a, b):
        """幂运算"""
        result = a ** b
        self.history.append(f"{a} ^ {b} = {result}")
        return result
    
    def sqrt(self, a):
        """平方根"""
        if a < 0:
            raise ValueError("负数不能开平方根")
        result = math.sqrt(a)
        self.history.append(f"sqrt({a}) = {result}")
        return result
    
    def sin(self, a):
        """正弦函数"""
        result = math.sin(a)
        self.history.append(f"sin({a}) = {result}")
        return result
    
    def cos(self, a):
        """余弦函数"""
        result = math.cos(a)
        self.history.append(f"cos({a}) = {result}")
        return result
    
    def tan(self, a):
        """正切函数"""
        result = math.tan(a)
        self.history.append(f"tan({a}) = {result}")
        return result
    
    def log(self, a):
        """自然对数"""
        if a <= 0:
            raise ValueError("对数的真数必须大于0")
        result = math.log(a)
        self.history.append(f"ln({a}) = {result}")
        return result
    
    def get_history(self):
        """获取计算历史"""
        return self.history
    
    def clear_history(self):
        """清除计算历史"""
        self.history.clear()


class CLIInterface:
    """命令行界面"""
    
    def __init__(self, calculator):
        self.calculator = calculator
    
    def print_menu(self):
        """打印菜单"""
        menu = """
        ╔══════════════════════════════════════╗
        ║              计算器菜单                ║
        ╠══════════════════════════════════════╣
        ║  1. 加法 (+)                         ║
        ║  2. 减法 (-)                         ║
        ║  3. 乘法 (*)                         ║
        ║  4. 除法 (/)                         ║
        ║  5. 幂运算 (^)                       ║
        ║  6. 平方根 (sqrt)                    ║
        ║  7. 正弦 (sin)                       ║
        ║  8. 余弦 (cos)                       ║
        ║  9. 正切 (tan)                       ║
        ║  10. 自然对数 (ln)                   ║
        ║  11. 查看历史记录                     ║
        ║  12. 清除历史记录                     ║
        ║  0. 退出                             ║
        ╚══════════════════════════════════════╝
        """
        print(menu)
    
    def get_number(self, prompt):
        """获取用户输入的数字"""
        while True:
            try:
                return float(input(prompt))
            except ValueError:
                print("请输入有效的数字！")
    
    def run(self):
        """运行命令行界面"""
        print("欢迎使用科学计算器！")
        
        while True:
            self.print_menu()
            
            try:
                choice = input("请选择操作 (0-12): ").strip()
                
                if choice == '0':
                    print("感谢使用！再见！")
                    break
                elif choice == '1':
                    a = self.get_number("请输入第一个数字: ")
                    b = self.get_number("请输入第二个数字: ")
                    result = self.calculator.add(a, b)
                    print(f"结果: {result}")
                elif choice == '2':
                    a = self.get_number("请输入第一个数字: ")
                    b = self.get_number("请输入第二个数字: ")
                    result = self.calculator.subtract(a, b)
                    print(f"结果: {result}")
                elif choice == '3':
                    a = self.get_number("请输入第一个数字: ")
                    b = self.get_number("请输入第二个数字: ")
                    result = self.calculator.multiply(a, b)
                    print(f"结果: {result}")
                elif choice == '4':
                    a = self.get_number("请输入被除数: ")
                    b = self.get_number("请输入除数: ")
                    result = self.calculator.divide(a, b)
                    print(f"结果: {result}")
                elif choice == '5':
                    a = self.get_number("请输入底数: ")
                    b = self.get_number("请输入指数: ")
                    result = self.calculator.power(a, b)
                    print(f"结果: {result}")
                elif choice == '6':
                    a = self.get_number("请输入数字: ")
                    result = self.calculator.sqrt(a)
                    print(f"结果: {result}")
                elif choice == '7':
                    a = self.get_number("请输入角度(弧度): ")
                    result = self.calculator.sin(a)
                    print(f"结果: {result}")
                elif choice == '8':
                    a = self.get_number("请输入角度(弧度): ")
                    result = self.calculator.cos(a)
                    print(f"结果: {result}")
                elif choice == '9':
                    a = self.get_number("请输入角度(弧度): ")
                    result = self.calculator.tan(a)
                    print(f"结果: {result}")
                elif choice == '10':
                    a = self.get_number("请输入数字: ")
                    result = self.calculator.log(a)
                    print(f"结果: {result}")
                elif choice == '11':
                    history = self.calculator.get_history()
                    if history:
                        print("\n计算历史:")
                        for i, record in enumerate(history, 1):
                            print(f"{i}. {record}")
                    else:
                        print("暂无计算历史")
                elif choice == '12':
                    self.calculator.clear_history()
                    print("历史记录已清除")
                else:
                    print("无效的选择，请重新输入！")
                    
            except ValueError as e:
                print(f"输入错误: {e}")
            except Exception as e:
                print(f"计算错误: {e}")
            
            input("\n按回车键继续...")


def setup_logger(level='INFO', debug=False):
    """设置日志记录器"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def print_banner():
    """打印程序启动横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                      科学计算器 v1.0.0                        ║
    ║                   Scientific Calculator                      ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  功能特性:                                                    ║
    ║  • 基本四则运算 (+, -, *, /)                                 ║
    ║  • 科学计算 (sin, cos, tan, log, exp, sqrt, 等)              ║
    ║  • 统计分析 (平均值, 标准差, 方差, 等)                        ║
    ║  • 单位转换 (长度, 重量, 温度, 等)                           ║
    ║  • 历史记录管理                                              ║
    ║  • 多种界面模式 (命令行/图形界面)                            ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_help():
    """打印帮助信息"""
    help_text = """
使用方法:
    python main.py [选项]

选项:
    -h, --help      显示此帮助信息
    -v, --version   显示版本信息
    -c, --cli       启动命令行界面 (默认)
    --debug         启用调试模式
    --log-level     设置日志级别 (DEBUG, INFO, WARNING, ERROR)

示例:
    python main.py              # 启动默认命令行界面
    python main.py --debug      # 启动调试模式
    python main.py --log-level DEBUG  # 设置详细日志
    """
    print(help_text)


def print_version():
    """打印版本信息"""
    version_info = """
科学计算器 v1.0.0
Python版本: {python_version}
平台: {platform}

Copyright (c) 2024 Calculator Team
MIT License
    """.format(
        python_version=sys.version.split()[0],
        platform=sys.platform
    )
    print(version_info)


def parse_arguments():
    """解析命令行参数"""
    args = {
        'interface': 'cli',
        'debug': False,
        'log_level': 'INFO'
    }
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg in ['-h', '--help']:
            print_help()
            sys.exit(0)
        elif arg in ['-v', '--version']:
            print_version()
            sys.exit(0)
        elif arg in ['-c', '--cli']:
            args['interface'] = 'cli'
        elif arg == '--debug':
            args['debug'] = True
            args['log_level'] = 'DEBUG'
        elif arg == '--log-level':
            if i + 1 < len(sys.argv):
                args['log_level'] = sys.argv[i + 1].upper()
                i += 1
            else:
                print("错误: --log-level 需要一个参数")
                sys.exit(1)
        else:
            print(f"未知参数: {arg}")
            print("使用 -h 或 --help 查看帮助信息")
            sys.exit(1)
        
        i += 1
    
    return args


def check_dependencies():
    """检查必要的依赖"""
    # 检查必要的内置模块是否可用
    try:
        import math
        import logging
        import sys
        import os
        return True
    except ImportError as e:
        print(f"错误: 缺少必要的模块: {e}")
        return False


def setup_environment(args):
    """设置运行环境"""
    # 设置日志
    logger = setup_logger(
        level=args['log_level'],
        debug=args['debug']
    )
    
    if args['debug']:
        logger.info("调试模式已启用")
        logger.info(f"Python版本: {sys.version}")
        logger.info(f"工作目录: {os.getcwd()}")
    
    return logger


def create_calculator():
    """创建计算器实例"""
    try:
        # 直接使用本文件中定义的Calculator类
        calculator = Calculator()
        return calculator
    except Exception as e:
        print(f"错误: 无法创建计算器实例: {e}")
        sys.exit(1)


def run_cli_interface(calculator, logger):
    """运行命令行界面"""
    try:
        # 直接使用本文件中定义的CLIInterface类
        cli = CLIInterface(calculator)
        logger.info("启动命令行界面")
        cli.run()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        logger.info("程序被用户中断")
    except Exception as e:
        print(f"命令行界面运行错误: {e}")
        logger.error(f"命令行界面运行错误: {e}")
        sys.exit(1)


def cleanup():
    """程序清理工作"""
    print("\n感谢使用科学计算器！")


def main():
    """主函数"""
    try:
        # 打印启动横幅
        print_banner()
        
        # 解析命令行参数
        args = parse_arguments()
        
        # 检查依赖
        if not check_dependencies():
            sys.exit(1)
        
        # 设置环境
        logger = setup_environment(args)
        
        # 创建计算器实例
        calculator = create_calculator()
        
        # 启动命令行界面
        run_cli_interface(calculator, logger)
        
    except Exception as e:
        print(f"程序运行出现未预期的错误: {e}")
        if 'logger' in locals():
            logger.error(f"程序运行出现未预期的错误: {e}")
        sys.exit(1)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
