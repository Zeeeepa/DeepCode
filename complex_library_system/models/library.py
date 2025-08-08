"""
文件概述：图书馆模型类定义
功能描述：定义Library类，包含完善的数据处理和错误处理

修复内容：
1. 添加JSON处理异常处理
2. 修复数据类型转换错误
3. 添加字典键存在性检查
4. 修复列表操作错误
5. 添加数据验证
6. 修复除零错误
7. 增强文件加载的错误处理和数据验证
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from .book import BookManager
from .user import UserManager

class Library:
    """图书馆类"""
    
    def __init__(self, name: str, location: str):
        self.name = name
        self.location = location
        self.book_manager = BookManager()
        self.user_manager = UserManager()
        self.transactions: List[Dict] = []
        
    def load_from_file(self, file_path: str) -> bool:
        """从文件加载图书馆数据，包含完善的错误处理和数据验证"""
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"警告：文件 {file_path} 不存在，将使用默认配置")
            return self._initialize_default_data()
            
        # 检查文件是否为空
        if os.path.getsize(file_path) == 0:
            print(f"警告：文件 {file_path} 为空，将使用默认配置")
            return self._initialize_default_data()
            
        try:
            # 修复11：添加JSON处理异常处理
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    print(f"警告：文件 {file_path} 内容为空，将使用默认配置")
                    return self._initialize_default_data()
                data = json.loads(content)
                
        except FileNotFoundError:
            print(f"警告：文件 {file_path} 不存在，将使用默认配置")
            return self._initialize_default_data()
        except json.JSONDecodeError as e:
            print(f"错误：JSON格式错误 - {e}，将使用默认配置")
            return self._initialize_default_data()
        except PermissionError:
            print(f"错误：没有权限读取文件 {file_path}")
            return False
        except Exception as e:
            print(f"错误：读取文件失败 - {e}，将使用默认配置")
            return self._initialize_default_data()
            
        # 验证数据结构
        if not isinstance(data, dict):
            print("错误：数据格式不正确，应为字典格式，将使用默认配置")
            return self._initialize_default_data()
            
        try:
            # 修复12：修复数据类型转换错误 - name应该保持为字符串
            if 'name' in data and isinstance(data['name'], str) and data['name'].strip():
                self.name = data['name'].strip()
            
            if 'location' in data and isinstance(data['location'], str) and data['location'].strip():
                self.location = data['location'].strip()
            
            # 修复13：添加字典键存在性检查和数据类型验证
            books_data = data.get('books', [])
            users_data = data.get('users', [])
            transactions_data = data.get('transactions', [])
            
            # 验证数据类型
            if not isinstance(books_data, list):
                print("警告：图书数据格式不正确，跳过图书加载")
                books_data = []
                
            if not isinstance(users_data, list):
                print("警告：用户数据格式不正确，跳过用户加载")
                users_data = []
                
            if not isinstance(transactions_data, list):
                print("警告：交易数据格式不正确，跳过交易记录加载")
                transactions_data = []
            
            # 加载图书数据
            loaded_books = 0
            for i, book_data in enumerate(books_data):
                if self._validate_book_data(book_data):
                    try:
                        from .book import Book
                        book = Book(
                            isbn=str(book_data['isbn']).strip(),
                            title=str(book_data['title']).strip(),
                            author=str(book_data['author']).strip(),
                            year=int(book_data['year'])
                        )
                        if self.book_manager.add_book(book):
                            loaded_books += 1
                    except (ValueError, TypeError) as e:
                        print(f"警告：第{i+1}本图书数据转换失败 - {e}")
                        continue
                else:
                    print(f"警告：第{i+1}本图书数据不完整，跳过")
                    
            print(f"成功加载 {loaded_books} 本图书")
                
            # 加载用户数据  
            loaded_users = 0
            for i, user_data in enumerate(users_data):
                if self._validate_user_data(user_data):
                    try:
                        from .user import User
                        user = User(
                            user_id=int(user_data['id']),
                            name=str(user_data['name']).strip(),
                            email=str(user_data['email']).strip()
                        )
                        # 确保用户有borrowed_books属性
                        if not hasattr(user, 'borrowed_books'):
                            user.borrowed_books = []
                        # 加载用户的借阅记录
                        if 'borrowed_books' in user_data and isinstance(user_data['borrowed_books'], list):
                            user.borrowed_books = [str(isbn) for isbn in user_data['borrowed_books']]
                        if self.user_manager.add_user(user):
                            loaded_users += 1
                    except (ValueError, TypeError) as e:
                        print(f"警告：第{i+1}个用户数据转换失败 - {e}")
                        continue
                else:
                    print(f"警告：第{i+1}个用户数据不完整，跳过")
                    
            print(f"成功加载 {loaded_users} 个用户")
            
            # 加载交易记录
            loaded_transactions = 0
            for transaction in transactions_data:
                if self._validate_transaction_data(transaction):
                    self.transactions.append(transaction)
                    loaded_transactions += 1
                    
            print(f"成功加载 {loaded_transactions} 条交易记录")
            
            return True
            
        except Exception as e:
            print(f"错误：处理数据时发生错误 - {e}")
            return self._initialize_default_data()
    
    def _initialize_default_data(self) -> bool:
        """初始化默认数据"""
        try:
            # 保持当前的名称和位置，或使用默认值
            if not hasattr(self, 'name') or not self.name:
                self.name = "默认图书馆"
            if not hasattr(self, 'location') or not self.location:
                self.location = "默认位置"
                
            # 确保管理器已初始化
            if not hasattr(self, 'book_manager') or self.book_manager is None:
                self.book_manager = BookManager()
            if not hasattr(self, 'user_manager') or self.user_manager is None:
                self.user_manager = UserManager()
            if not hasattr(self, 'transactions') or self.transactions is None:
                self.transactions = []
                
            print("已初始化默认配置")
            return True
        except Exception as e:
            print(f"错误：初始化默认数据失败 - {e}")
            return False
    
    def _validate_book_data(self, book_data: Dict) -> bool:
        """验证图书数据的完整性"""
        if not isinstance(book_data, dict):
            return False
            
        required_fields = ['isbn', 'title', 'author', 'year']
        
        # 检查必需字段是否存在且不为空
        for field in required_fields:
            if field not in book_data:
                return False
            if book_data[field] is None or str(book_data[field]).strip() == '':
                return False
                
        # 验证年份是否为有效数字
        try:
            year = int(book_data['year'])
            if year < 0 or year > datetime.now().year + 10:
                return False
        except (ValueError, TypeError):
            return False
            
        return True
    
    def _validate_user_data(self, user_data: Dict) -> bool:
        """验证用户数据的完整性"""
        if not isinstance(user_data, dict):
            return False
            
        required_fields = ['id', 'name', 'email']
        
        # 检查必需字段是否存在且不为空
        for field in required_fields:
            if field not in user_data:
                return False
            if user_data[field] is None or str(user_data[field]).strip() == '':
                return False
                
        # 验证用户ID是否为有效数字
        try:
            user_id = int(user_data['id'])
            if user_id <= 0:
                return False
        except (ValueError, TypeError):
            return False
            
        # 简单的邮箱格式验证
        email = str(user_data['email']).strip()
        if '@' not in email or '.' not in email:
            return False
            
        return True
    
    def _validate_transaction_data(self, transaction_data: Dict) -> bool:
        """验证交易数据的完整性"""
        if not isinstance(transaction_data, dict):
            return False
            
        required_fields = ['type', 'user_id', 'isbn']
        
        for field in required_fields:
            if field not in transaction_data:
                return False
            if transaction_data[field] is None:
                return False
                
        # 验证交易类型
        if transaction_data['type'] not in ['borrow', 'return']:
            return False
            
        return True
    
    def borrow_book(self, user_id: int, isbn: str) -> bool:
        """借书操作，包含完善的错误处理"""
        # 修复15：添加数据验证
        if not isinstance(user_id, int) or not isinstance(isbn, str):
            print("错误：参数类型不正确")
            return False
            
        if user_id <= 0:
            print("错误：用户ID必须为正整数")
            return False
            
        if not isbn.strip():
            print("错误：ISBN不能为空")
            return False
            
        user = self.user_manager.find_user(user_id)
        book = self.book_manager.find_book(isbn)
        
        if not user:
            print(f"错误：用户 {user_id} 不存在")
            return False
            
        if not book:
            print(f"错误：图书 {isbn} 不存在")
            return False
            
        if book.borrow(user_id):
            # 修复14：完善交易记录，添加时间戳
            self.transactions.append({
                'type': 'borrow',
                'user_id': user_id,
                'isbn': isbn,
                'timestamp': datetime.now().isoformat()
            })
            
            # 修复14：确保borrowed_books列表存在，并安全添加
            if not hasattr(user, 'borrowed_books'):
                user.borrowed_books = []
            
            if isbn not in user.borrowed_books:
                user.borrowed_books.append(isbn)
            
            return True
        return False
    
    def return_book(self, user_id: int, isbn: str) -> bool:
        """还书操作，包含完善的错误处理"""
        # 修复15：添加数据验证
        if not isinstance(user_id, int) or not isinstance(isbn, str):
            print("错误：参数类型不正确")
            return False
            
        if user_id <= 0:
            print("错误：用户ID必须为正整数")
            return False
            
        if not isbn.strip():
            print("错误：ISBN不能为空")
            return False
            
        user = self.user_manager.find_user(user_id)
        book = self.book_manager.find_book(isbn)
        
        if not user:
            print(f"错误：用户 {user_id} 不存在")
            return False
            
        if not book:
            print(f"错误：图书 {isbn} 不存在")
            return False
            
        if book.borrowed_by == user_id:
            book.return_book()
            
            self.transactions.append({
                'type': 'return',
                'user_id': user_id, 
                'isbn': isbn,
                'timestamp': datetime.now().isoformat()
            })
            
            # 修复14：安全的列表操作，避免ValueError
            if hasattr(user, 'borrowed_books') and isbn in user.borrowed_books:
                try:
                    user.borrowed_books.remove(isbn)
                except ValueError:
                    print(f"警告：图书 {isbn} 不在用户 {user_id} 的借阅列表中")
            
            return True
        else:
            print(f"错误：图书 {isbn} 不是由用户 {user_id} 借阅的")
            return False
    
    def generate_report(self) -> Dict[str, Any]:
        """生成统计报告，包含数据验证"""
        try:
            # 修复15：添加数据验证
            total_books = len(self.book_manager.books) if self.book_manager.books else 0
            available_books = len(self.book_manager.get_available_books()) if hasattr(self.book_manager, 'get_available_books') else 0
            total_users = len(self.user_manager.users) if self.user_manager.users else 0
            
            # 确保数据的有效性
            borrowed_books = max(0, total_books - available_books)
            
            # 修复15：避免除零错误
            utilization_rate = 0.0
            if total_books > 0:
                utilization_rate = (borrowed_books / total_books) * 100
            
            return {
                'library_name': self.name,
                'total_books': total_books,
                'available_books': available_books,
                'borrowed_books': borrowed_books,
                'total_users': total_users,
                'transactions_count': len(self.transactions),
                'utilization_rate': round(utilization_rate, 2)
            }
            
        except Exception as e:
            print(f"错误：生成报告时发生错误 - {e}")
            return {
                'library_name': self.name,
                'total_books': 0,
                'available_books': 0,
                'borrowed_books': 0,
                'total_users': 0,
                'transactions_count': 0,
                'utilization_rate': 0.0,
                'error': str(e)
            }
    
    def save_to_file(self, file_path: str) -> bool:
        """保存图书馆数据到文件"""
        try:
            # 确保目录存在
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            data = {
                'name': self.name,
                'location': self.location,
                'books': [
                    {
                        'isbn': book.isbn,
                        'title': book.title,
                        'author': book.author,
                        'year': book.year
                    }
                    for book in self.book_manager.books
                ],
                'users': [
                    {
                        'id': user.user_id,
                        'name': user.name,
                        'email': user.email,
                        'borrowed_books': getattr(user, 'borrowed_books', [])
                    }
                    for user in self.user_manager.users
                ],
                'transactions': self.transactions
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except PermissionError:
            print(f"错误：没有权限写入文件 {file_path}")
            return False
        except Exception as e:
            print(f"错误：保存文件失败 - {e}")
            return False
