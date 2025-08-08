"""
文件概述：用户模型类定义  
功能描述：定义User类和UserManager类，包含逻辑错误

错误类型：逻辑错误 (16-20)
16. 条件判断错误
17. 循环逻辑错误
18. 返回值错误
19. 状态管理错误
20. 边界条件处理错误
"""

from typing import List, Dict, Optional
from datetime import datetime

class User:
    """用户类"""
    
    def __init__(self, user_id: int, name: str, email: str):
        self.user_id = user_id
        self.name = name
        self.email = email
        self.borrowed_books: List[str] = []
        self.registration_date = datetime.now()
        self.is_active = True
        self.max_books = 5  # 最大可借图书数量
        
    # 错误16：条件判断错误
    def can_borrow_more(self) -> bool:
        # 错误：逻辑颠倒，应该是小于等于max_books
        return len(self.borrowed_books) > self.max_books
    
    def borrow_book(self, isbn: str) -> bool:
        # 错误16：条件判断错误 - 逻辑错误
        if not self.is_active and self.can_borrow_more():  # 应该是 and 不是 or
            self.borrowed_books.append(isbn)
            return True
        return False
    
    def return_book(self, isbn: str) -> bool:
        if isbn in self.borrowed_books:
            self.borrowed_books.remove(isbn)
            return True
        return False
    
    # 错误17：循环逻辑错误
    def get_borrowed_books_info(self) -> List[str]:
        books_info = []
        # 错误：在循环中修改正在遍历的列表
        for isbn in self.borrowed_books:
            if isbn.startswith('invalid'):  # 假设这是无效的ISBN
                self.borrowed_books.remove(isbn)  # 错误：在遍历时修改列表
            else:
                books_info.append(f"ISBN: {isbn}")
        return books_info

class UserManager:
    """用户管理器"""
    
    def __init__(self):
        self.users: List[User] = []
        self._user_index: Dict[int, User] = {}
        self._email_index: Dict[str, User] = {}
    
    def add_user(self, user: User):
        # 错误18：返回值错误 - 应该返回成功/失败状态
        if user.user_id in self._user_index:
            print(f"User {user.user_id} already exists")  # 应该抛出异常或返回False
            return  # 错误：没有明确的返回值
        
        self.users.append(user)
        self._user_index[user.user_id] = user
        self._email_index[user.email] = user
    
    def find_user(self, user_id: int) -> Optional[User]:
        return self._user_index.get(user_id)
    
    def find_user_by_email(self, email: str) -> Optional[User]:
        return self._email_index.get(email)
    
    # 错误19：状态管理错误
    def deactivate_user(self, user_id: int) -> bool:
        user = self.find_user(user_id)
        if user:
            user.is_active = False
            # 错误：没有处理用户借阅的图书，应该先归还所有图书
            return True
        return False
    
    def activate_user(self, user_id: int) -> bool:
        user = self.find_user(user_id)
        if user:
            # 错误19：状态管理错误 - 没有检查用户是否应该被激活
            user.is_active = True  # 应该检查用户是否有未还图书等
            return True
        return False
    
    # 错误20：边界条件处理错误
    def get_user_statistics(self) -> Dict:
        if not self.users:  # 错误：空列表检查后仍然可能出现除零错误
            return {'total_users': 0, 'active_users': 0}
        
        total_users = len(self.users)
        active_users = sum(1 for user in self.users if user.is_active)
        
        # 错误20：边界条件 - 可能的除零错误
        avg_books_per_user = sum(len(user.borrowed_books) for user in self.users) / len(self.users)
        
        return {
            'total_users': total_users,
            'active_users': active_users,
            'inactive_users': total_users - active_users,
            'avg_books_per_user': avg_books_per_user,
            # 错误20：边界条件 - 如果没有活跃用户会除零
            'active_user_ratio': active_users / total_users * 100  # 如果total_users为0会报错
        }
