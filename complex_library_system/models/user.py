"""
文件概述：用户模型类定义  
功能描述：定义User类和UserManager类，修复逻辑错误

修复内容：
- 修正条件判断错误
- 修复循环逻辑错误
- 改进返回值处理
- 修复状态管理错误
- 添加边界条件检查
- 增强异常处理和输入验证
"""

from typing import List, Dict, Optional
from datetime import datetime

class User:
    """用户类"""
    
    def __init__(self, user_id: int, name: str, email: str):
        # 输入验证
        if not isinstance(user_id, int) or user_id <= 0:
            raise ValueError("用户ID必须是正整数")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("用户姓名不能为空")
        if not isinstance(email, str) or not email.strip():
            raise ValueError("邮箱地址不能为空")
        
        self.user_id = user_id
        self.name = name.strip()
        self.email = email.strip()
        self.borrowed_books: List[str] = []
        self.registration_date = datetime.now()
        self.is_active = True
        self.max_books = 5  # 最大可借图书数量
        
    def can_borrow_more(self) -> bool:
        """检查用户是否可以借阅更多图书"""
        # 修复：正确的逻辑判断，当前借阅数量小于最大限制时可以借阅更多
        return len(self.borrowed_books) < self.max_books
    
    def borrow_book(self, isbn: str) -> bool:
        """借阅图书"""
        try:
            # 输入验证
            if not isinstance(isbn, str) or not isbn.strip():
                return False
            
            isbn = isbn.strip()
            
            # 修复：正确的条件判断，用户必须是活跃状态且可以借阅更多图书
            if self.is_active and self.can_borrow_more():
                # 检查是否已经借阅了这本书
                if isbn not in self.borrowed_books:
                    self.borrowed_books.append(isbn)
                    return True
            return False
        except Exception:
            # 异常情况下返回False，确保方法总是返回布尔值
            return False
    
    def return_book(self, isbn: str) -> bool:
        """归还图书"""
        try:
            # 输入验证
            if not isinstance(isbn, str) or not isbn.strip():
                return False
            
            isbn = isbn.strip()
            
            if isbn in self.borrowed_books:
                self.borrowed_books.remove(isbn)
                return True
            return False
        except Exception:
            # 异常情况下返回False
            return False
    
    def get_borrowed_books_info(self) -> List[str]:
        """获取借阅图书信息，同时清理无效ISBN"""
        try:
            books_info = []
            # 修复：使用副本进行遍历，避免在遍历时修改原列表
            books_to_remove = []
            
            for isbn in self.borrowed_books:
                if isbn.startswith('invalid'):  # 假设这是无效的ISBN
                    books_to_remove.append(isbn)  # 记录需要移除的ISBN
                else:
                    books_info.append(f"ISBN: {isbn}")
            
            # 在遍历完成后移除无效的ISBN
            for isbn in books_to_remove:
                self.borrowed_books.remove(isbn)
                
            return books_info
        except Exception:
            # 异常情况下返回空列表
            return []

class UserManager:
    """用户管理器"""
    
    def __init__(self):
        self.users: List[User] = []
        self._user_index: Dict[int, User] = {}
        self._email_index: Dict[str, User] = {}
    
    def add_user(self, user: User) -> bool:
        """添加用户，返回操作是否成功"""
        try:
            # 输入验证
            if not isinstance(user, User):
                return False
            
            # 修复：明确返回布尔值表示操作结果
            if user.user_id in self._user_index:
                return False  # 用户已存在，返回False
            
            if user.email in self._email_index:
                return False  # 邮箱已被使用，返回False
            
            self.users.append(user)
            self._user_index[user.user_id] = user
            self._email_index[user.email] = user
            return True  # 成功添加用户
            
        except Exception:
            # 异常情况下返回False，确保方法总是返回布尔值
            return False
    
    def find_user(self, user_id: int) -> Optional[User]:
        """根据用户ID查找用户"""
        try:
            if not isinstance(user_id, int):
                return None
            return self._user_index.get(user_id)
        except Exception:
            return None
    
    def find_user_by_email(self, email: str) -> Optional[User]:
        """根据邮箱查找用户"""
        try:
            if not isinstance(email, str):
                return None
            return self._email_index.get(email.strip())
        except Exception:
            return None
    
    def deactivate_user(self, user_id: int) -> bool:
        """停用用户，需要先归还所有图书"""
        try:
            user = self.find_user(user_id)
            if user:
                # 修复：检查用户是否有未归还的图书
                if user.borrowed_books:
                    return False  # 用户还有未归还的图书，不能停用
                
                user.is_active = False
                return True
            return False
        except Exception:
            return False
    
    def activate_user(self, user_id: int) -> bool:
        """激活用户"""
        try:
            user = self.find_user(user_id)
            if user:
                # 修复：添加激活条件检查
                if not user.is_active:  # 只有非活跃用户才需要激活
                    user.is_active = True
                    return True
                return False  # 用户已经是活跃状态
            return False
        except Exception:
            return False
    
    def force_return_all_books(self, user_id: int) -> bool:
        """强制归还用户的所有图书"""
        try:
            user = self.find_user(user_id)
            if user:
                user.borrowed_books.clear()
                return True
            return False
        except Exception:
            return False
    
    def get_user_statistics(self) -> Dict:
        """获取用户统计信息，包含边界条件检查"""
        try:
            # 修复：完善边界条件检查
            if not self.users:
                return {
                    'total_users': 0,
                    'active_users': 0,
                    'inactive_users': 0,
                    'avg_books_per_user': 0.0,
                    'active_user_ratio': 0.0
                }
            
            total_users = len(self.users)
            active_users = sum(1 for user in self.users if user.is_active)
            
            # 计算平均借阅图书数
            total_borrowed_books = sum(len(user.borrowed_books) for user in self.users)
            avg_books_per_user = total_borrowed_books / total_users if total_users > 0 else 0.0
            
            # 修复：添加除零检查
            active_user_ratio = (active_users / total_users * 100) if total_users > 0 else 0.0
            
            return {
                'total_users': total_users,
                'active_users': active_users,
                'inactive_users': total_users - active_users,
                'avg_books_per_user': round(avg_books_per_user, 2),
                'active_user_ratio': round(active_user_ratio, 2)
            }
        except Exception:
            # 异常情况下返回默认统计信息
            return {
                'total_users': 0,
                'active_users': 0,
                'inactive_users': 0,
                'avg_books_per_user': 0.0,
                'active_user_ratio': 0.0
            }
    
    def remove_user(self, user_id: int) -> bool:
        """移除用户（仅当用户没有借阅图书时）"""
        try:
            user = self.find_user(user_id)
            if user:
                if user.borrowed_books:
                    return False  # 用户还有未归还的图书，不能移除
                
                # 从所有索引中移除用户
                self.users.remove(user)
                del self._user_index[user.user_id]
                del self._email_index[user.email]
                return True
            return False
        except Exception:
            return False
