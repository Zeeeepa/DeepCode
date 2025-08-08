import tkinter as tk
from tkinter import messagebox
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_module import Calculator


class CalculatorGUI:
    """
    计算器图形用户界面类
    使用tkinter实现基本的计算器功能
    """
    
    def __init__(self):
        """初始化GUI界面"""
        self.root = tk.Tk()
        self.root.title("计算器")
        self.root.geometry("300x400")
        self.root.resizable(False, False)
        
        # 初始化计算器引擎
        self.calculator = Calculator()
        
        # 当前显示的表达式
        self.current_expression = ""
        
        # 创建界面元素
        self.create_display()
        self.create_buttons()
    
    def create_display(self):
        """创建显示屏"""
        # 显示框架
        display_frame = tk.Frame(self.root, bg='black', padx=5, pady=5)
        display_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 显示标签
        self.display_var = tk.StringVar()
        self.display_var.set("0")
        
        self.display = tk.Label(
            display_frame,
            textvariable=self.display_var,
            font=('Arial', 20, 'bold'),
            bg='black',
            fg='white',
            anchor='e',
            padx=10,
            pady=10
        )
        self.display.pack(fill=tk.X)
    
    def create_buttons(self):
        """创建按钮布局"""
        # 按钮框架
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 按钮布局定义
        buttons = [
            ['C', '±', '%', '÷'],
            ['7', '8', '9', '×'],
            ['4', '5', '6', '-'],
            ['1', '2', '3', '+'],
            ['0', '.', '=']
        ]
        
        # 创建按钮
        for i, row in enumerate(buttons):
            for j, text in enumerate(row):
                if text == '0':
                    # 0按钮占两列
                    btn = tk.Button(
                        button_frame,
                        text=text,
                        font=('Arial', 16, 'bold'),
                        command=lambda t=text: self.button_click(t),
                        bg='#f0f0f0',
                        relief='raised',
                        bd=2
                    )
                    btn.grid(row=i, column=j, columnspan=2, sticky='nsew', padx=1, pady=1)
                elif text == '=':
                    # 等号按钮
                    btn = tk.Button(
                        button_frame,
                        text=text,
                        font=('Arial', 16, 'bold'),
                        command=lambda t=text: self.button_click(t),
                        bg='#ff9500',
                        fg='white',
                        relief='raised',
                        bd=2
                    )
                    btn.grid(row=i, column=j+1, sticky='nsew', padx=1, pady=1)
                elif text in ['÷', '×', '-', '+']:
                    # 运算符按钮
                    btn = tk.Button(
                        button_frame,
                        text=text,
                        font=('Arial', 16, 'bold'),
                        command=lambda t=text: self.button_click(t),
                        bg='#ff9500',
                        fg='white',
                        relief='raised',
                        bd=2
                    )
                    btn.grid(row=i, column=j, sticky='nsew', padx=1, pady=1)
                elif text in ['C', '±', '%']:
                    # 功能按钮
                    btn = tk.Button(
                        button_frame,
                        text=text,
                        font=('Arial', 16, 'bold'),
                        command=lambda t=text: self.button_click(t),
                        bg='#a6a6a6',
                        fg='black',
                        relief='raised',
                        bd=2
                    )
                    btn.grid(row=i, column=j, sticky='nsew', padx=1, pady=1)
                else:
                    # 数字按钮
                    btn = tk.Button(
                        button_frame,
                        text=text,
                        font=('Arial', 16, 'bold'),
                        command=lambda t=text: self.button_click(t),
                        bg='#f0f0f0',
                        relief='raised',
                        bd=2
                    )
                    btn.grid(row=i, column=j, sticky='nsew', padx=1, pady=1)
        
        # 配置网格权重
        for i in range(5):
            button_frame.grid_rowconfigure(i, weight=1)
        for j in range(4):
            button_frame.grid_columnconfigure(j, weight=1)
    
    def button_click(self, char):
        """处理按钮点击事件"""
        try:
            if char == 'C':
                # 清除
                self.clear()
            elif char == '=':
                # 计算结果
                self.calculate()
            elif char == '±':
                # 正负号切换
                self.toggle_sign()
            elif char == '%':
                # 百分号
                self.percentage()
            elif char in ['÷', '×', '-', '+']:
                # 运算符
                self.add_operator(char)
            else:
                # 数字和小数点
                self.add_number(char)
        except Exception as e:
            messagebox.showerror("错误", f"操作失败: {str(e)}")
    
    def clear(self):
        """清除显示和表达式"""
        self.current_expression = ""
        self.display_var.set("0")
    
    def add_number(self, number):
        """添加数字或小数点"""
        if self.current_expression == "" and number != ".":
            self.current_expression = number
        elif number == "." and "." in self.current_expression.split()[-1]:
            # 防止重复小数点
            return
        else:
            if self.current_expression == "0" and number != ".":
                self.current_expression = number
            else:
                self.current_expression += number
        
        self.display_var.set(self.current_expression)
    
    def add_operator(self, operator):
        """添加运算符"""
        if self.current_expression == "":
            return
        
        # 转换显示符号为计算符号
        op_map = {'÷': '/', '×': '*', '-': '-', '+': '+'}
        calc_operator = op_map[operator]
        
        # 如果最后一个字符是运算符，替换它
        if self.current_expression and self.current_expression[-1] in "+-*/":
            self.current_expression = self.current_expression[:-1] + calc_operator
        else:
            self.current_expression += calc_operator
        
        self.display_var.set(self.current_expression)
    
    def calculate(self):
        """计算表达式结果"""
        if self.current_expression == "":
            return
        
        try:
            # 使用计算器引擎计算结果
            result = self.calculator.evaluate_expression(self.current_expression)
            
            # 格式化结果显示
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            
            self.display_var.set(str(result))
            self.current_expression = str(result)
            
        except Exception as e:
            messagebox.showerror("计算错误", f"无法计算表达式: {str(e)}")
            self.clear()
    
    def toggle_sign(self):
        """切换正负号"""
        if self.current_expression and self.current_expression != "0":
            if self.current_expression.startswith("-"):
                self.current_expression = self.current_expression[1:]
            else:
                self.current_expression = "-" + self.current_expression
            
            self.display_var.set(self.current_expression)
    
    def percentage(self):
        """百分号操作"""
        if self.current_expression:
            try:
                value = float(self.current_expression)
                result = value / 100
                
                if result.is_integer():
                    result = int(result)
                
                self.current_expression = str(result)
                self.display_var.set(self.current_expression)
                
            except ValueError:
                messagebox.showerror("错误", "无效的数字格式")
    
    def run(self):
        """启动GUI应用"""
        self.root.mainloop()


def main():
    """主函数"""
    try:
        app = CalculatorGUI()
        app.run()
    except Exception as e:
        print(f"启动GUI失败: {e}")
        messagebox.showerror("启动错误", f"无法启动计算器GUI: {str(e)}")


if __name__ == "__main__":
    main()
