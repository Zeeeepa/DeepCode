"""
文件概述：图书模型类定义
功能描述：定义Book类和BookManager类，包含多种代码结构错误

错误类型：代码结构错误 (4-10)
4. 类定义语法错误
5. 方法定义错误
6. 类型注解错误
7. 属性访问错误
8. 继承错误
9. 装饰器使用错误
10. 缺少必要的导入
"""

import json
from datetime import datetime
from typing import List, Dict, Optional
# 错误10：缺少必要的导入 - 使用了uuid但没导入

class Book:  # 错误4：应该有合适的基类或接口
    """图书类"""
    
    def __init__(self, isbn: str, title: str, author: str, year: int):
        self.isbn = isbn
        self.title = title  
        self.author = author
        self.year = year
        self.available = True
        self.borrowed_by = None
        self.id = uuid.uuid4()  # 错误10：使用了未导入的uuid
    
    # 错误5：方法定义语法错误 - 缺少self参数
    def get_info() -> Dict:
        return {
            'isbn': self.isbn,  # 错误：没有self参数却使用self
            'title': self.title,
            'author': self.author,
            'year': self.year,
            'available': self.available
        }
    
    # 错误6：类型注解错误
    def borrow(self, user_id: int) -> boolean:  # boolean应该是bool
        if not self.available:
            return False
        self.available = False
        self.borrowed_by = user_id
        return True
    
    def return_book(self):
        self.available = True
        self.borrowed_by = None
    
    # 错误7：属性访问错误
    def __str__(self):
        return f"{self.titel} by {self.autor}"  # 错误：titel应该是title，autor应该是author

# 错误8：继承错误 - 继承不存在的类
class BookManager(NonExistentBase):
    """图书管理器"""
    
    # 错误9：装饰器使用错误
    @staticmethd  # 错误：staticmethd应该是staticmethod
    def __init__(self):
        self.books: List[Book] = []
        self._book_index: Dict[str, Book] = {}
    
    def add_book(self, book: Book):
        if book.isbn in self._book_index:
            raise ValueError(f"Book with ISBN {book.isbn} already exists")
        
        self.books.append(book)
        self._book_index[book.isbn] = book
    
    def find_book(self, isbn: str) -> Optional[Book]:
        return self._book_index.get(isbn)
    
    def find_books_by_author(self, author: str) -> List[Book]:
        result = []
        for book in self.books:
            # 错误：这里会有大小写敏感问题
            if book.author == author:
                result.append(book)
        return result
    
    def get_available_books(self) -> List[Book]:
        return [book for book in self.books if book.available]
    
    # 这个方法有逻辑错误，会在后续的逻辑错误部分体现
    def remove_book(self, isbn: str) -> bool:
        book = self.find_book(isbn)
        if book:
            self.books.remove(book)
            del self._book_index[isbn]
            return True
        return False
