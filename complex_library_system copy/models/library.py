"""
文件概述：图书馆模型类定义
功能描述：定义Library类，包含数据处理错误

错误类型：数据处理错误 (11-15)
11. JSON处理错误
12. 数据类型转换错误
13. 字典键访问错误
14. 列表操作错误
15. 数据验证缺失
"""

import json
from datetime import datetime
from typing import List, Dict, Any
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
        
    # 错误11：JSON处理错误 - 没有异常处理
    def load_from_file(self, file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)  # 如果文件格式错误会崩溃
            
        # 错误12：数据类型转换错误
        self.name = int(data['name'])  # name应该是字符串，不应该转换为int
        self.location = data['location']
        
        # 错误13：字典键访问错误 - 直接访问可能不存在的键
        books_data = data['books']  # 如果'books'键不存在会报错
        users_data = data['users']  # 如果'users'键不存在会报错
        
        # 加载图书数据
        for book_data in books_data:
            from .book import Book
            book = Book(
                isbn=book_data['isbn'],
                title=book_data['title'], 
                author=book_data['author'],
                year=book_data['year']
            )
            self.book_manager.add_book(book)
            
        # 加载用户数据  
        for user_data in users_data:
            from .user import User
            user = User(
                user_id=user_data['id'],
                name=user_data['name'],
                email=user_data['email']
            )
            self.user_manager.add_user(user)
    
    # 错误14：列表操作错误
    def borrow_book(self, user_id: int, isbn: str) -> bool:
        user = self.user_manager.find_user(user_id)
        book = self.book_manager.find_book(isbn)
        
        if not user or not book:
            return False
            
        if book.borrow(user_id):
            # 错误14：列表操作 - append操作缺少必要字段
            self.transactions.append({
                'type': 'borrow',
                'user_id': user_id,
                'isbn': isbn
                # 缺少时间戳字段
            })
            
            # 错误：向用户的borrowed_books列表添加，但没检查列表是否存在
            user.borrowed_books.append(isbn)  # 如果borrowed_books不存在会报错
            return True
        return False
    
    def return_book(self, user_id: int, isbn: str) -> bool:
        user = self.user_manager.find_user(user_id)
        book = self.book_manager.find_book(isbn)
        
        if not user or not book:
            return False
            
        if book.borrowed_by == user_id:
            book.return_book()
            
            self.transactions.append({
                'type': 'return',
                'user_id': user_id, 
                'isbn': isbn,
                'timestamp': datetime.now().isoformat()
            })
            
            # 错误14：列表操作错误 - remove可能抛出ValueError
            user.borrowed_books.remove(isbn)  # 如果isbn不在列表中会报错
            return True
        return False
    
    # 错误15：数据验证缺失
    def generate_report(self) -> Dict[str, Any]:
        total_books = len(self.book_manager.books)
        available_books = len(self.book_manager.get_available_books())
        total_users = len(self.user_manager.users)
        
        # 错误：没有验证数据的有效性
        borrowed_books = total_books - available_books
        
        return {
            'library_name': self.name,
            'total_books': total_books,
            'available_books': available_books,
            'borrowed_books': borrowed_books,
            'total_users': total_users,
            'transactions_count': len(self.transactions),
            # 错误15：计算可能导致除零错误
            'utilization_rate': borrowed_books / total_books * 100  # 如果total_books为0会报错
        }
