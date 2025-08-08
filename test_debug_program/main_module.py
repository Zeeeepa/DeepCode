"""
主计算器模块
提供基本的数学运算功能
"""

import math
from typing import Union, List


class Calculator:
    """基础计算器类，提供基本的数学运算功能"""
    
    def __init__(self):
        """初始化计算器"""
        self.history = []
    
    def add(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """
        加法运算
        
        Args:
            a: 第一个数
            b: 第二个数
            
        Returns:
            两数之和
        """
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """
        减法运算
        
        Args:
            a: 被减数
            b: 减数
            
        Returns:
            两数之差
        """
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result
    
    def multiply(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """
        乘法运算
        
        Args:
            a: 第一个数
            b: 第二个数
            
        Returns:
            两数之积
        """
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def divide(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """
        除法运算
        
        Args:
            a: 被除数
            b: 除数
            
        Returns:
            两数之商
            
        Raises:
            ValueError: 当除数为0时
        """
        if b == 0:
            raise ValueError("除数不能为零")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result
    
    def power(self, base: Union[int, float], exponent: Union[int, float]) -> Union[int, float]:
        """
        幂运算
        
        Args:
            base: 底数
            exponent: 指数
            
        Returns:
            幂运算结果
        """
        result = base ** exponent
        self.history.append(f"{base} ^ {exponent} = {result}")
        return result
    
    def square_root(self, x: Union[int, float]) -> float:
        """
        平方根运算
        
        Args:
            x: 被开方数
            
        Returns:
            平方根结果
            
        Raises:
            ValueError: 当输入负数时
        """
        if x < 0:
            raise ValueError("不能计算负数的平方根")
        result = math.sqrt(x)
        self.history.append(f"√{x} = {result}")
        return result
    
    def get_history(self) -> List[str]:
        """
        获取计算历史记录
        
        Returns:
            计算历史记录列表
        """
        return self.history.copy()
    
    def clear_history(self) -> None:
        """清空计算历史记录"""
        self.history.clear()


class ScientificCalculator(Calculator):
    """科学计算器类，继承基础计算器并添加科学计算功能"""
    
    def sin(self, x: Union[int, float]) -> float:
        """
        正弦函数
        
        Args:
            x: 角度值（弧度）
            
        Returns:
            正弦值
        """
        result = math.sin(x)
        self.history.append(f"sin({x}) = {result}")
        return result
    
    def cos(self, x: Union[int, float]) -> float:
        """
        余弦函数
        
        Args:
            x: 角度值（弧度）
            
        Returns:
            余弦值
        """
        result = math.cos(x)
        self.history.append(f"cos({x}) = {result}")
        return result
    
    def tan(self, x: Union[int, float]) -> float:
        """
        正切函数
        
        Args:
            x: 角度值（弧度）
            
        Returns:
            正切值
        """
        result = math.tan(x)
        self.history.append(f"tan({x}) = {result}")
        return result
    
    def log(self, x: Union[int, float], base: Union[int, float] = math.e) -> float:
        """
        对数函数
        
        Args:
            x: 真数
            base: 底数，默认为自然对数底e
            
        Returns:
            对数值
            
        Raises:
            ValueError: 当真数小于等于0或底数小于等于0且不等于1时
        """
        if x <= 0:
            raise ValueError("真数必须大于0")
        if base <= 0 or base == 1:
            raise ValueError("底数必须大于0且不等于1")
        
        if base == math.e:
            result = math.log(x)
            self.history.append(f"ln({x}) = {result}")
        else:
            result = math.log(x, base)
            self.history.append(f"log_{base}({x}) = {result}")
        return result
    
    def factorial(self, n: int) -> int:
        """
        阶乘函数
        
        Args:
            n: 非负整数
            
        Returns:
            阶乘结果
            
        Raises:
            ValueError: 当输入负数或非整数时
        """
        if not isinstance(n, int) or n < 0:
            raise ValueError("阶乘只能计算非负整数")
        result = math.factorial(n)
        self.history.append(f"{n}! = {result}")
        return result


def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """
    独立的加法函数
    
    Args:
        a: 第一个数
        b: 第二个数
        
    Returns:
        两数之和
    """
    return a + b


def subtract(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """
    独立的减法函数
    
    Args:
        a: 被减数
        b: 减数
        
    Returns:
        两数之差
    """
    return a - b


def multiply(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """
    独立的乘法函数
    
    Args:
        a: 第一个数
        b: 第二个数
        
    Returns:
        两数之积
    """
    return a * b


def divide(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """
    独立的除法函数
    
    Args:
        a: 被除数
        b: 除数
        
    Returns:
        两数之商
        
    Raises:
        ValueError: 当除数为0时
    """
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b


def is_even(n: int) -> bool:
    """
    判断数字是否为偶数
    
    Args:
        n: 整数
        
    Returns:
        如果是偶数返回True，否则返回False
    """
    return n % 2 == 0


def is_prime(n: int) -> bool:
    """
    判断数字是否为质数
    
    Args:
        n: 正整数
        
    Returns:
        如果是质数返回True，否则返回False
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def fibonacci(n: int) -> int:
    """
    计算斐波那契数列的第n项
    
    Args:
        n: 非负整数，表示斐波那契数列的位置
        
    Returns:
        斐波那契数列第n项的值
        
    Raises:
        ValueError: 当输入负数时
    """
    if n < 0:
        raise ValueError("斐波那契数列位置不能为负数")
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def gcd(a: int, b: int) -> int:
    """
    计算两个数的最大公约数
    
    Args:
        a: 第一个整数
        b: 第二个整数
        
    Returns:
        最大公约数
    """
    return math.gcd(abs(a), abs(b))


def lcm(a: int, b: int) -> int:
    """
    计算两个数的最小公倍数
    
    Args:
        a: 第一个整数
        b: 第二个整数
        
    Returns:
        最小公倍数
    """
    return abs(a * b) // gcd(a, b) if a != 0 and b != 0 else 0


if __name__ == "__main__":
    # 示例用法
    calc = Calculator()
    print(f"5 + 3 = {calc.add(5, 3)}")
    print(f"10 - 4 = {calc.subtract(10, 4)}")
    print(f"6 * 7 = {calc.multiply(6, 7)}")
    print(f"15 / 3 = {calc.divide(15, 3)}")
    print(f"2^8 = {calc.power(2, 8)}")
    print(f"√16 = {calc.square_root(16)}")
    
    print("\n计算历史:")
    for record in calc.get_history():
        print(record)
    
    # 科学计算器示例
    sci_calc = ScientificCalculator()
    print(f"\nsin(π/2) = {sci_calc.sin(math.pi/2)}")
    print(f"cos(0) = {sci_calc.cos(0)}")
    print(f"5! = {sci_calc.factorial(5)}")
    
    # 独立函数示例
    print(f"\n独立函数示例:")
    print(f"add(10, 20) = {add(10, 20)}")
    print(f"is_even(4) = {is_even(4)}")
    print(f"is_prime(17) = {is_prime(17)}")
    print(f"fibonacci(10) = {fibonacci(10)}")
    print(f"gcd(48, 18) = {gcd(48, 18)}")
    print(f"lcm(12, 8) = {lcm(12, 8)}")