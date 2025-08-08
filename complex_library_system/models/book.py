"""
文件概述：图书模型类定义
功能描述：定义Book类和BookManager类，修复了多种代码结构错误

修复内容：
1. 添加了缺失的uuid导入
2. 修复了get_info方法缺少self参数的问题
3. 修正了boolean类型注解为bool
4. 修复了__str__方法中的属性名错误
5. 移除了BookManager的错误继承
6. 修复了装饰器拼写错误
7. 修复了add_book方法的返回值问题
"""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional

class Book:
    """图书类"""
    
    def __init__(self, isbn: str, title: str, author: str, year: int):
        self.isbn = isbn
        self.title = title  
        self.author = author
        self.year = year
        self.available = True
        self.borrowed_by = None
        self.id = uuid.uuid4()
    
    def get_info(self) -> Dict:
        """获取图书信息"""
        return {
            'isbn': self.isbn,
            'title': self.title,
            'author': self.author,
            'year': self.year,
            'available': self.available
        }
    
    def borrow(self, user_id: int) -> bool:
        """借书操作"""
        if not self.available:
            return False
        self.available = False
        self.borrowed_by = user_id
        return True
    
    def return_book(self):
        """还书操作"""
        self.available = True
        self.borrowed_by = None
    
    def __str__(self):
        """字符串表示"""
        return f"{self.title} by {self.author}"

class BookManager:
    """图书管理器"""
    
    def __init__(self):
        """初始化图书管理器"""
        self.books: List[Book] = []
        self._book_index: Dict[str, Book] = {}
    
    def add_book(self, book: Book) -> bool:
        """添加图书
        
        Args:
            book: 要添加的图书对象
            
        Returns:
            bool: 添加成功返回True，失败返回False
        """
        try:
            if book.isbn in self._book_index:
                # 图书已存在，返回False而不是抛出异常
                return False
            
            self.books.append(book)
            self._book_index[book.isbn] = book
            return True
        except Exception:
            # 任何异常都返回False
            return False
    
    def find_book(self, isbn: str) -> Optional[Book]:
        """根据ISBN查找图书"""
        return self._book_index.get(isbn)
    
    def find_books_by_author(self, author: str) -> List[Book]:
        """根据作者查找图书（不区分大小写）"""
        result = []
        for book in self.books:
            if book.author.lower() == author.lower():
                result.append(book)
        return result
    
    def get_available_books(self) -> List[Book]:
        """获取所有可借阅的图书"""
        return [book for book in self.books if book.available]
    
    def remove_book(self, isbn: str) -> bool:
        """移除图书"""
        book = self.find_book(isbn)
        if book:
            # 检查图书是否已被借出
            if not book.available:
                return False
            self.books.remove(book)
            del self._book_index[isbn]
            return True
        return False
