"""
文件概述：图书管理系统主程序
功能描述：系统主入口，整合所有模块功能，包含剩余的各类错误

错误类型：剩余错误 (46-50) + 综合错误
46. 配置错误
47. 日志记录错误
48. 异常传播错误
49. 资源管理错误
50. 系统集成错误
"""

import sys
import os
from datetime import datetime
from typing import Dict, List, Any

# 错误46：配置错误 - 硬编码配置
CONFIG = {
    'data_file': 'data/library_data.json',
    'log_file': '/tmp/library.log',  # 错误：硬编码路径
    'max_users': 1000,
    'max_books': 5000,
    'debug': True
}

# 错误47：日志记录错误 - 简单的print语句代替专业日志
def log_message(level: str, message: str):
    # 错误47：没有使用标准logging模块
    print(f"[{level}] {message}")  # 错误：日志没有时间戳、没有写入文件

# 错误1：导入错误（与之前的错误呼应）
try:
    from models import Library, Book, User
    from models.book import BookManager  
    from models.user import UserManager
    from utils import FileHandler, Validator
except ImportError as e:
    # 错误48：异常传播错误 - 捕获后没有适当处理
    log_message("ERROR", f"Import error: {e}")
    # 错误：继续执行而不是退出程序

class LibrarySystem:
    """图书管理系统主类"""
    
    def __init__(self):
        # 错误49：资源管理错误 - 没有适当的初始化检查
        self.library = None
        self.file_handler = FileHandler()
        self.validator = Validator()
        self.is_initialized = False
        
    def initialize(self):
        """初始化系统"""
        try:
            log_message("INFO", "Initializing library system...")
            
            # 错误50：系统集成错误 - 组件之间的协调问题
            self.library = Library("Central Library", "Main Street")
            
            # 错误：没有检查数据文件是否存在
            if os.path.exists(CONFIG['data_file']):
                self.library.load_from_file(CONFIG['data_file'])
            
            self.is_initialized = True
            log_message("INFO", "System initialized successfully")
            
        except Exception as e:
            # 错误48：异常传播错误
            log_message("ERROR", f"Initialization failed: {e}")
            # 错误：没有设置适当的错误状态
    
    def run_interactive_mode(self):
        """运行交互模式"""
        # 错误：没有检查系统是否已初始化
        if not self.is_initialized:
            log_message("ERROR", "System not initialized")
            return
            
        log_message("INFO", "Starting interactive mode...")
        
        while True:
            try:
                self.show_menu()
                choice = input("请选择操作 (1-6): ").strip()
                
                if choice == '1':
                    self.add_book_interactive()
                elif choice == '2':
                    self.add_user_interactive()
                elif choice == '3':
                    self.borrow_book_interactive()
                elif choice == '4':
                    self.return_book_interactive()
                elif choice == '5':
                    self.show_report()
                elif choice == '6':
                    log_message("INFO", "Exiting system...")
                    break
                else:
                    print("无效选择，请重试")
                    
            except KeyboardInterrupt:
                log_message("INFO", "User interrupted")
                break
            except Exception as e:
                # 错误48：异常传播错误 - 捕获所有异常但处理不当
                log_message("ERROR", f"Unexpected error: {e}")
                # 错误：继续运行而不是适当处理错误
    
    def show_menu(self):
        """显示菜单"""
        print("\n" + "="*50)
        print("📚 图书管理系统")
        print("="*50)
        print("1. 添加图书")
        print("2. 添加用户")
        print("3. 借阅图书")
        print("4. 归还图书")
        print("5. 生成报告")
        print("6. 退出系统")
        print("="*50)
    
    def add_book_interactive(self):
        """交互式添加图书"""
        try:
            print("\n📖 添加新图书")
            isbn = input("ISBN: ").strip()
            title = input("书名: ").strip()
            author = input("作者: ").strip()
            year = input("出版年份: ").strip()
            
            # 错误：没有使用验证器验证输入
            book = Book(isbn, title, author, int(year))  # 错误：直接转换可能失败
            self.library.book_manager.add_book(book)
            
            print(f"✅ 图书 '{title}' 添加成功！")
            
        except ValueError as e:
            print(f"❌ 输入错误: {e}")
        except Exception as e:
            print(f"❌ 添加图书失败: {e}")
    
    def add_user_interactive(self):
        """交互式添加用户"""
        try:
            print("\n👤 添加新用户")
            user_id = input("用户ID: ").strip()
            name = input("姓名: ").strip()
            email = input("邮箱: ").strip()
            
            # 错误：类型转换错误
            user = User(int(user_id), name, email)  # 如果user_id不是数字会失败
            self.library.user_manager.add_user(user)
            
            print(f"✅ 用户 '{name}' 添加成功！")
            
        except ValueError as e:
            print(f"❌ 输入错误: {e}")
        except Exception as e:
            print(f"❌ 添加用户失败: {e}")
    
    def borrow_book_interactive(self):
        """交互式借阅图书"""
        try:
            print("\n📚 借阅图书")
            user_id = int(input("用户ID: ").strip())
            isbn = input("图书ISBN: ").strip()
            
            if self.library.borrow_book(user_id, isbn):
                print("✅ 借阅成功！")
            else:
                print("❌ 借阅失败！请检查用户ID和ISBN是否正确，或图书是否可借阅。")
                
        except ValueError as e:
            print(f"❌ 输入错误: {e}")
        except Exception as e:
            print(f"❌ 借阅失败: {e}")
    
    def return_book_interactive(self):
        """交互式归还图书"""
        try:
            print("\n📖 归还图书")
            user_id = int(input("用户ID: ").strip())
            isbn = input("图书ISBN: ").strip()
            
            if self.library.return_book(user_id, isbn):
                print("✅ 归还成功！")
            else:
                print("❌ 归还失败！请检查用户ID和ISBN是否正确。")
                
        except ValueError as e:
            print(f"❌ 输入错误: {e}")
        except Exception as e:
            print(f"❌ 归还失败: {e}")
    
    def show_report(self):
        """显示报告"""
        try:
            print("\n📊 图书馆统计报告")
            print("="*50)
            
            report = self.library.generate_report()
            
            for key, value in report.items():
                print(f"{key}: {value}")
                
        except Exception as e:
            print(f"❌ 生成报告失败: {e}")
    
    # 错误49：资源管理错误
    def cleanup(self):
        """清理资源"""
        # 错误：没有适当的资源清理
        if hasattr(self, 'file_handler'):
            # 错误：没有调用清理方法
            pass
        
        # 错误：没有保存数据
        log_message("INFO", "System cleanup completed")

def main():
    """主函数"""
    print("🚀 启动图书管理系统...")
    
    # 错误46：配置错误 - 没有验证配置
    system = LibrarySystem()
    
    try:
        system.initialize()
        
        if len(sys.argv) > 1 and sys.argv[1] == '--demo':
            run_demo(system)
        else:
            system.run_interactive_mode()
            
    except Exception as e:
        log_message("CRITICAL", f"System crashed: {e}")
        # 错误48：异常传播错误 - 没有适当的错误处理
        sys.exit(1)
    finally:
        # 错误49：资源管理错误
        system.cleanup()  # 这个方法本身有问题

def run_demo(system: LibrarySystem):
    """运行演示模式"""
    log_message("INFO", "Running demo mode...")
    
    try:
        # 添加示例数据
        from models.book import Book
        from models.user import User
        
        # 添加图书
        books = [
            Book("9781234567890", "Python编程", "张三", 2020),
            Book("9781234567891", "数据结构", "李四", 2019),
            Book("9781234567892", "算法导论", "王五", 2021)
        ]
        
        for book in books:
            system.library.book_manager.add_book(book)
        
        # 添加用户
        users = [
            User(1, "用户A", "usera@example.com"),
            User(2, "用户B", "userb@example.com")
        ]
        
        for user in users:
            system.library.user_manager.add_user(user)
        
        # 执行一些操作
        system.library.borrow_book(1, "9781234567890")
        system.library.borrow_book(2, "9781234567891")
        
        # 显示报告
        system.show_report()
        
        log_message("INFO", "Demo completed successfully")
        
    except Exception as e:
        log_message("ERROR", f"Demo failed: {e}")

if __name__ == "__main__":
    main()
