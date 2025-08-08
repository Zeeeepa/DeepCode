import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_module import Calculator


class CLIInterface:
    """命令行界面类，提供计算器的交互功能"""
    
    def __init__(self):
        """初始化CLI界面"""
        self.calculator = Calculator()
        self.running = True
        
    def display_welcome(self):
        """显示欢迎信息"""
        print("=" * 50)
        print("欢迎使用计算器！")
        print("=" * 50)
        print("支持的操作：")
        print("  加法: a + b")
        print("  减法: a - b") 
        print("  乘法: a * b")
        print("  除法: a / b")
        print("  幂运算: a ** b")
        print("  退出: quit 或 exit")
        print("=" * 50)
        
    def display_help(self):
        """显示帮助信息"""
        print("\n使用说明：")
        print("1. 输入数学表达式，如：3 + 5")
        print("2. 支持小数和负数")
        print("3. 输入 'quit' 或 'exit' 退出程序")
        print("4. 输入 'help' 查看帮助")
        print()
        
    def parse_expression(self, expression):
        """解析用户输入的表达式"""
        try:
            # 移除空格
            expression = expression.replace(" ", "")
            
            # 检查是否包含支持的运算符
            operators = ['+', '-', '*', '/', '**']
            operator = None
            
            # 先检查幂运算符（两个字符）
            if '**' in expression:
                operator = '**'
                parts = expression.split('**')
            else:
                # 检查其他运算符
                for op in ['+', '-', '*', '/']:
                    if op in expression:
                        # 处理负数的情况
                        if op == '-' and expression.startswith('-'):
                            # 如果是负数开头，查找下一个运算符
                            temp_expr = expression[1:]
                            if '-' in temp_expr:
                                operator = '-'
                                parts = ['-' + temp_expr.split('-')[0], temp_expr.split('-')[1]]
                                break
                            elif '+' in temp_expr:
                                operator = '+'
                                parts = ['-' + temp_expr.split('+')[0], temp_expr.split('+')[1]]
                                break
                            elif '*' in temp_expr:
                                operator = '*'
                                parts = ['-' + temp_expr.split('*')[0], temp_expr.split('*')[1]]
                                break
                            elif '/' in temp_expr:
                                operator = '/'
                                parts = ['-' + temp_expr.split('/')[0], temp_expr.split('/')[1]]
                                break
                        else:
                            operator = op
                            parts = expression.split(op)
                            break
            
            if operator is None or len(parts) != 2:
                raise ValueError("无效的表达式格式")
                
            # 转换为数字
            num1 = float(parts[0])
            num2 = float(parts[1])
            
            return num1, operator, num2
            
        except Exception as e:
            raise ValueError(f"表达式解析错误: {str(e)}")
    
    def calculate(self, num1, operator, num2):
        """执行计算操作"""
        try:
            if operator == '+':
                return self.calculator.add(num1, num2)
            elif operator == '-':
                return self.calculator.subtract(num1, num2)
            elif operator == '*':
                return self.calculator.multiply(num1, num2)
            elif operator == '/':
                return self.calculator.divide(num1, num2)
            elif operator == '**':
                return self.calculator.power(num1, num2)
            else:
                raise ValueError(f"不支持的运算符: {operator}")
        except Exception as e:
            raise ValueError(f"计算错误: {str(e)}")
    
    def format_result(self, result):
        """格式化计算结果"""
        # 如果结果是整数，显示为整数
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        else:
            return str(result)
    
    def process_input(self, user_input):
        """处理用户输入"""
        user_input = user_input.strip()
        
        # 检查退出命令
        if user_input.lower() in ['quit', 'exit', 'q']:
            self.running = False
            return "再见！"
        
        # 检查帮助命令
        if user_input.lower() in ['help', 'h']:
            self.display_help()
            return None
        
        # 检查空输入
        if not user_input:
            return "请输入一个表达式"
        
        try:
            # 解析表达式
            num1, operator, num2 = self.parse_expression(user_input)
            
            # 执行计算
            result = self.calculate(num1, operator, num2)
            
            # 格式化并返回结果
            formatted_result = self.format_result(result)
            return f"{user_input} = {formatted_result}"
            
        except ValueError as e:
            return f"错误: {str(e)}"
        except Exception as e:
            return f"未知错误: {str(e)}"
    
    def run(self):
        """运行CLI界面主循环"""
        self.display_welcome()
        
        while self.running:
            try:
                # 获取用户输入
                user_input = input("\n请输入表达式 (输入 'help' 查看帮助): ")
                
                # 处理输入
                result = self.process_input(user_input)
                
                # 显示结果
                if result:
                    print(result)
                    
            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                self.running = False
            except EOFError:
                print("\n\n程序结束")
                self.running = False
            except Exception as e:
                print(f"程序错误: {str(e)}")
        
        print("感谢使用计算器！")


def main():
    """主函数"""
    cli = CLIInterface()
    cli.run()


if __name__ == "__main__":
    main()
