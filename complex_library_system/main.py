"""
文件概述：图书管理系统主程序
功能描述：系统主入口，整合所有模块功能，修复了配置、日志、异常处理、资源管理等错误

修复内容：
- 改进配置管理，支持环境变量和配置文件
- 实现专业日志记录系统
- 改进异常处理和传播机制
- 修复资源管理问题
- 改进系统集成和初始化检查
- 添加输入验证和错误处理流程
- 修复模块导入逻辑，增强错误处理机制
- 改进add_book_interactive方法的错误处理，确保正确捕获BookManager异常
"""

import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        # 默认配置
        default_config = {
            'data_file': 'data/library_data.json',
            'log_file': 'logs/library.log',
            'max_users': 1000,
            'max_books': 5000,
            'debug': False,
            'log_level': 'INFO'
        }
        
        # 尝试从配置文件加载
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    default_config.update(file_config)
            except Exception as e:
                print(f"警告：无法加载配置文件 {self.config_file}: {e}")
        
        # 环境变量覆盖
        env_mappings = {
            'LIBRARY_DATA_FILE': 'data_file',
            'LIBRARY_LOG_FILE': 'log_file',
            'LIBRARY_MAX_USERS': 'max_users',
            'LIBRARY_MAX_BOOKS': 'max_books',
            'LIBRARY_DEBUG': 'debug',
            'LIBRARY_LOG_LEVEL': 'log_level'
        }
        
        for env_var, config_key in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                if config_key in ['max_users', 'max_books']:
                    try:
                        default_config[config_key] = int(value)
                    except ValueError:
                        pass
                elif config_key == 'debug':
                    default_config[config_key] = value.lower() in ('true', '1', 'yes')
                else:
                    default_config[config_key] = value
        
        return default_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)
    
    def save_config(self):
        """保存配置到文件"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"警告：无法保存配置文件: {e}")

class LogManager:
    """日志管理器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('LibrarySystem')
        logger.setLevel(getattr(logging, self.config.get('log_level', 'INFO')))
        
        # 清除现有处理器
        logger.handlers.clear()
        
        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器
        log_file = self.config.get('log_file')
        if log_file:
            try:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"无法创建日志文件处理器: {e}")
        
        return logger
    
    def log(self, level: str, message: str, exc_info: bool = False):
        """记录日志"""
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message, exc_info=exc_info)

def safe_import() -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """安全导入模块，返回导入状态、错误信息和导入的模块"""
    modules = {}
    missing_modules = []
    
    # 尝试导入各个模块
    try:
        from models import Library
        modules['Library'] = Library
    except ImportError as e:
        missing_modules.append(f"models.Library: {e}")
    
    try:
        from models.book import Book, BookManager
        modules['Book'] = Book
        modules['BookManager'] = BookManager
    except ImportError as e:
        missing_modules.append(f"models.book: {e}")
    
    try:
        from models.user import User, UserManager
        modules['User'] = User
        modules['UserManager'] = UserManager
    except ImportError as e:
        missing_modules.append(f"models.user: {e}")
    
    try:
        from utils import FileHandler, Validator
        modules['FileHandler'] = FileHandler
        modules['Validator'] = Validator
    except ImportError as e:
        missing_modules.append(f"utils: {e}")
    
    # 检查是否所有必需模块都已导入
    required_modules = ['Library', 'Book', 'User', 'FileHandler', 'Validator']
    missing_required = [mod for mod in required_modules if mod not in modules]
    
    if missing_required:
        error_msg = f"缺少必需模块: {', '.join(missing_required)}"
        if missing_modules:
            error_msg += f"\n详细错误: {'; '.join(missing_modules)}"
        return False, error_msg, None
    
    return True, None, modules

class LibrarySystemError(Exception):
    """图书管理系统自定义异常"""
    pass

class LibrarySystem:
    """图书管理系统主类"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_manager = ConfigManager(config_file)
        self.log_manager = LogManager(self.config_manager)
        self.library = None
        self.file_handler = None
        self.validator = None
        self.modules = None
        self.is_initialized = False
        self._resources_acquired = []
        
    def initialize(self) -> bool:
        """初始化系统"""
        try:
            self.log_manager.log("INFO", "正在初始化图书管理系统...")
            
            # 检查模块导入
            import_success, import_error, modules = safe_import()
            if not import_success:
                self.log_manager.log("ERROR", f"模块导入失败: {import_error}")
                raise LibrarySystemError(f"模块导入失败: {import_error}")
            
            self.modules = modules
            self.log_manager.log("INFO", "所有必需模块导入成功")
            
            # 初始化组件
            try:
                self.file_handler = modules['FileHandler']()
                self.validator = modules['Validator']()
                self.library = modules['Library']("中央图书馆", "主街道")
                
                # 记录已获取的资源
                self._resources_acquired = ['file_handler', 'validator', 'library']
                self.log_manager.log("INFO", "核心组件初始化成功")
                
            except Exception as e:
                self.log_manager.log("ERROR", f"组件初始化失败: {e}", exc_info=True)
                raise LibrarySystemError(f"组件初始化失败: {e}")
            
            # 创建必要的目录
            try:
                data_file = self.config_manager.get('data_file')
                if data_file:
                    data_dir = os.path.dirname(data_file)
                    if data_dir:
                        os.makedirs(data_dir, exist_ok=True)
                        self.log_manager.log("INFO", f"数据目录创建成功: {data_dir}")
            except Exception as e:
                self.log_manager.log("WARNING", f"创建数据目录失败: {e}")
            
            # 加载数据
            try:
                data_file = self.config_manager.get('data_file')
                if data_file and os.path.exists(data_file):
                    self.library.load_from_file(data_file)
                    self.log_manager.log("INFO", f"成功加载数据文件: {data_file}")
                else:
                    self.log_manager.log("INFO", "数据文件不存在，将使用空数据库")
            except Exception as e:
                self.log_manager.log("WARNING", f"加载数据文件失败，将使用空数据库: {e}")
            
            # 验证系统完整性
            try:
                self._validate_system_integrity()
                self.log_manager.log("INFO", "系统完整性验证通过")
            except Exception as e:
                self.log_manager.log("ERROR", f"系统完整性验证失败: {e}")
                raise LibrarySystemError(f"系统完整性验证失败: {e}")
            
            self.is_initialized = True
            self.log_manager.log("INFO", "系统初始化成功")
            return True
            
        except LibrarySystemError:
            # 重新抛出自定义异常
            raise
        except Exception as e:
            self.log_manager.log("ERROR", f"系统初始化失败: {e}", exc_info=True)
            self.is_initialized = False
            raise LibrarySystemError(f"系统初始化失败: {e}")
    
    def _validate_system_integrity(self):
        """验证系统完整性"""
        # 检查核心组件
        if not self.library:
            raise LibrarySystemError("图书馆对象未初始化")
        
        if not self.file_handler:
            raise LibrarySystemError("文件处理器未初始化")
        
        if not self.validator:
            raise LibrarySystemError("验证器未初始化")
        
        # 检查图书馆组件
        if not hasattr(self.library, 'book_manager') or not self.library.book_manager:
            raise LibrarySystemError("图书管理器未初始化")
        
        if not hasattr(self.library, 'user_manager') or not self.library.user_manager:
            raise LibrarySystemError("用户管理器未初始化")
        
        # 检查必要方法
        required_methods = ['borrow_book', 'return_book', 'generate_report', 'save_to_file', 'load_from_file']
        for method in required_methods:
            if not hasattr(self.library, method):
                raise LibrarySystemError(f"图书馆对象缺少必要方法: {method}")
    
    def _check_initialized(self):
        """检查系统是否已初始化"""
        if not self.is_initialized:
            raise LibrarySystemError("系统未初始化，请先调用 initialize() 方法")
    
    def run_interactive_mode(self):
        """运行交互模式"""
        try:
            self._check_initialized()
            self.log_manager.log("INFO", "启动交互模式...")
            
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
                        self.log_manager.log("INFO", "用户选择退出系统")
                        self._save_data()
                        break
                    else:
                        print("❌ 无效选择，请重试")
                        
                except KeyboardInterrupt:
                    self.log_manager.log("INFO", "用户中断操作")
                    print("\n\n👋 感谢使用图书管理系统！")
                    break
                except Exception as e:
                    self.log_manager.log("ERROR", f"交互模式发生错误: {e}", exc_info=True)
                    print(f"❌ 操作失败: {e}")
                    
        except LibrarySystemError as e:
            self.log_manager.log("ERROR", str(e))
            print(f"❌ 系统错误: {e}")
        except Exception as e:
            self.log_manager.log("CRITICAL", f"交互模式严重错误: {e}", exc_info=True)
            print(f"❌ 系统发生严重错误: {e}")
    
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
            year_str = input("出版年份: ").strip()
            
            # 验证输入
            if not all([isbn, title, author, year_str]):
                raise ValueError("所有字段都不能为空")
            
            if not self.validator.validate_isbn(isbn):
                raise ValueError("ISBN格式无效")
            
            try:
                year = int(year_str)
                if year < 1000 or year > datetime.now().year + 1:
                    raise ValueError("出版年份无效")
            except ValueError:
                raise ValueError("出版年份必须是有效的数字")
            
            # 创建图书对象并添加到图书管理器
            try:
                book = self.modules['Book'](isbn, title, author, year)
                self.library.book_manager.add_book(book)
                
                print(f"✅ 图书 '{title}' 添加成功！")
                self.log_manager.log("INFO", f"添加图书成功: {title} (ISBN: {isbn})")
                
            except ValueError as e:
                # 捕获BookManager.add_book可能抛出的ValueError异常
                print(f"❌ 添加图书失败: {e}")
                self.log_manager.log("WARNING", f"添加图书失败: {title} (ISBN: {isbn}) - {e}")
            except Exception as e:
                # 捕获其他可能的异常
                print(f"❌ 添加图书时发生错误: {e}")
                self.log_manager.log("ERROR", f"添加图书时发生错误: {title} (ISBN: {isbn}) - {e}", exc_info=True)
            
        except ValueError as e:
            print(f"❌ 输入错误: {e}")
            self.log_manager.log("WARNING", f"添加图书输入错误: {e}")
        except Exception as e:
            print(f"❌ 添加图书失败: {e}")
            self.log_manager.log("ERROR", f"添加图书失败: {e}", exc_info=True)
    
    def add_user_interactive(self):
        """交互式添加用户"""
        try:
            print("\n👤 添加新用户")
            user_id_str = input("用户ID: ").strip()
            name = input("姓名: ").strip()
            email = input("邮箱: ").strip()
            
            # 验证输入
            if not all([user_id_str, name, email]):
                raise ValueError("所有字段都不能为空")
            
            try:
                user_id = int(user_id_str)
                if user_id <= 0:
                    raise ValueError("用户ID必须是正整数")
            except ValueError:
                raise ValueError("用户ID必须是有效的数字")
            
            if not self.validator.validate_email(email):
                raise ValueError("邮箱格式无效")
            
            user = self.modules['User'](user_id, name, email)
            self.library.user_manager.add_user(user)
            
            print(f"✅ 用户 '{name}' 添加成功！")
            self.log_manager.log("INFO", f"添加用户成功: {name} (ID: {user_id})")
            
        except ValueError as e:
            print(f"❌ 输入错误: {e}")
            self.log_manager.log("WARNING", f"添加用户输入错误: {e}")
        except Exception as e:
            print(f"❌ 添加用户失败: {e}")
            self.log_manager.log("ERROR", f"添加用户失败: {e}", exc_info=True)
    
    def borrow_book_interactive(self):
        """交互式借阅图书"""
        try:
            print("\n📚 借阅图书")
            user_id_str = input("用户ID: ").strip()
            isbn = input("图书ISBN: ").strip()
            
            if not all([user_id_str, isbn]):
                raise ValueError("用户ID和ISBN都不能为空")
            
            try:
                user_id = int(user_id_str)
            except ValueError:
                raise ValueError("用户ID必须是有效的数字")
            
            if self.library.borrow_book(user_id, isbn):
                print("✅ 借阅成功！")
                self.log_manager.log("INFO", f"借阅成功: 用户{user_id} 借阅 {isbn}")
            else:
                print("❌ 借阅失败！请检查用户ID和ISBN是否正确，或图书是否可借阅。")
                self.log_manager.log("WARNING", f"借阅失败: 用户{user_id} 尝试借阅 {isbn}")
                
        except ValueError as e:
            print(f"❌ 输入错误: {e}")
            self.log_manager.log("WARNING", f"借阅图书输入错误: {e}")
        except Exception as e:
            print(f"❌ 借阅失败: {e}")
            self.log_manager.log("ERROR", f"借阅失败: {e}", exc_info=True)
    
    def return_book_interactive(self):
        """交互式归还图书"""
        try:
            print("\n📖 归还图书")
            user_id_str = input("用户ID: ").strip()
            isbn = input("图书ISBN: ").strip()
            
            if not all([user_id_str, isbn]):
                raise ValueError("用户ID和ISBN都不能为空")
            
            try:
                user_id = int(user_id_str)
            except ValueError:
                raise ValueError("用户ID必须是有效的数字")
            
            if self.library.return_book(user_id, isbn):
                print("✅ 归还成功！")
                self.log_manager.log("INFO", f"归还成功: 用户{user_id} 归还 {isbn}")
            else:
                print("❌ 归还失败！请检查用户ID和ISBN是否正确。")
                self.log_manager.log("WARNING", f"归还失败: 用户{user_id} 尝试归还 {isbn}")
                
        except ValueError as e:
            print(f"❌ 输入错误: {e}")
            self.log_manager.log("WARNING", f"归还图书输入错误: {e}")
        except Exception as e:
            print(f"❌ 归还失败: {e}")
            self.log_manager.log("ERROR", f"归还失败: {e}", exc_info=True)
    
    def show_report(self):
        """显示报告"""
        try:
            print("\n📊 图书馆统计报告")
            print("="*50)
            
            report = self.library.generate_report()
            
            for key, value in report.items():
                print(f"{key}: {value}")
            
            self.log_manager.log("INFO", "生成统计报告成功")
                
        except Exception as e:
            print(f"❌ 生成报告失败: {e}")
            self.log_manager.log("ERROR", f"生成报告失败: {e}", exc_info=True)
    
    def _save_data(self):
        """保存数据"""
        try:
            data_file = self.config_manager.get('data_file')
            if data_file and self.library:
                self.library.save_to_file(data_file)
                self.log_manager.log("INFO", f"数据保存成功: {data_file}")
        except Exception as e:
            self.log_manager.log("ERROR", f"保存数据失败: {e}", exc_info=True)
    
    def cleanup(self):
        """清理资源"""
        try:
            self.log_manager.log("INFO", "开始清理系统资源...")
            
            # 保存数据
            if self.is_initialized:
                self._save_data()
            
            # 清理资源
            for resource_name in self._resources_acquired:
                try:
                    resource = getattr(self, resource_name, None)
                    if resource and hasattr(resource, 'cleanup'):
                        resource.cleanup()
                    setattr(self, resource_name, None)
                except Exception as e:
                    self.log_manager.log("WARNING", f"清理资源 {resource_name} 失败: {e}")
            
            self._resources_acquired.clear()
            self.is_initialized = False
            
            self.log_manager.log("INFO", "系统资源清理完成")
            
        except Exception as e:
            self.log_manager.log("ERROR", f"资源清理失败: {e}", exc_info=True)

def main():
    """主函数"""
    print("🚀 启动图书管理系统...")
    
    system = None
    try:
        system = LibrarySystem()
        
        if not system.initialize():
            print("❌ 系统初始化失败，请检查日志文件")
            return 1
        
        if len(sys.argv) > 1 and sys.argv[1] == '--demo':
            run_demo(system)
        else:
            system.run_interactive_mode()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n👋 感谢使用图书管理系统！")
        return 0
    except LibrarySystemError as e:
        if system and system.log_manager:
            system.log_manager.log("CRITICAL", f"系统错误: {e}")
        print(f"❌ 系统错误: {e}")
        return 1
    except Exception as e:
        if system and system.log_manager:
            system.log_manager.log("CRITICAL", f"系统崩溃: {e}", exc_info=True)
        print(f"❌ 系统发生严重错误: {e}")
        return 1
    finally:
        if system:
            system.cleanup()

def run_demo(system: LibrarySystem):
    """运行演示模式"""
    try:
        system.log_manager.log("INFO", "运行演示模式...")
        
        # 添加示例图书
        books_data = [
            ("9787111544937", "Python编程：从入门到实践", "埃里克·马瑟斯", 2020),
            ("9787115428028", "数据结构与算法分析", "马克·艾伦·维斯", 2019),
            ("9787111407010", "算法导论", "托马斯·科尔曼", 2021)
        ]
        
        print("\n📚 添加示例图书...")
        for isbn, title, author, year in books_data:
            try:
                book = system.modules['Book'](isbn, title, author, year)
                system.library.book_manager.add_book(book)
                print(f"✅ 添加图书: {title}")
            except Exception as e:
                print(f"❌ 添加图书失败: {title} - {e}")
        
        # 添加示例用户
        users_data = [
            (1001, "张三", "zhangsan@example.com"),
            (1002, "李四", "lisi@example.com"),
            (1003, "王五", "wangwu@example.com")
        ]
        
        print("\n👤 添加示例用户...")
        for user_id, name, email in users_data:
            try:
                user = system.modules['User'](user_id, name, email)
                system.library.user_manager.add_user(user)
                print(f"✅ 添加用户: {name}")
            except Exception as e:
                print(f"❌ 添加用户失败: {name} - {e}")
        
        # 执行借阅操作
        print("\n📖 执行示例借阅...")
        borrow_operations = [
            (1001, "9787111544937"),
            (1002, "9787115428028"),
            (1003, "9787111407010")
        ]
        
        for user_id, isbn in borrow_operations:
            if system.library.borrow_book(user_id, isbn):
                print(f"✅ 用户 {user_id} 成功借阅图书 {isbn}")
            else:
                print(f"❌ 用户 {user_id} 借阅图书 {isbn} 失败")
        
        # 显示报告
        print("\n📊 生成演示报告...")
        system.show_report()
        
        system.log_manager.log("INFO", "演示模式完成")
        print("\n✅ 演示模式运行完成！")
        
    except Exception as e:
        system.log_manager.log("ERROR", f"演示模式失败: {e}", exc_info=True)
        print(f"❌ 演示模式失败: {e}")

if __name__ == "__main__":
    sys.exit(main())
