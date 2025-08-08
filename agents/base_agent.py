"""
基础Agent类

所有Agent的基础类，提供统一的接口和通用功能。

主要功能:
- call_llm(): 调用LLM的统一接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from .config import AgentConfig
from .utils import LLMClient


class BaseAgent(ABC):
    """
    Agent基础类
    
    所有Agent都应该继承此类，并实现抽象方法。
    提供了配置管理、LLM客户端等通用功能。
    """
    
    def __init__(self, **kwargs):
        """
        初始化Agent
        
        参数:
            **kwargs: 其他配置参数
        """
        self.config = AgentConfig()
        self.llm_client = None
        self._setup_llm_client()

    def call_llm(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        调用LLM的统一接口（带输入长度检查和错误处理）
        
        参数:
            messages (list): 消息列表，格式如 [{"role": "user", "content": "内容"}]
            **kwargs: LLM参数（temperature, max_tokens等）
        
        返回:
            str: LLM的响应内容
        """
        if not self.llm_client:
            return "LLM客户端未初始化，请检查配置"
        
        # 检查输入长度
        total_length = sum(len(msg.get("content", "")) for msg in messages)
        print(f"📏 输入总长度: {total_length:,} 字符")
        
        # 如果输入过长，尝试截断
        MAX_INPUT_LENGTH = 120000  # 12万字符限制
        if total_length > MAX_INPUT_LENGTH:
            print(f"⚠️ 输入过长 ({total_length:,} > {MAX_INPUT_LENGTH:,})，尝试截断...")
            messages = self._truncate_messages(messages, MAX_INPUT_LENGTH)
            new_length = sum(len(msg.get("content", "")) for msg in messages)
            print(f"📏 截断后长度: {new_length:,} 字符")
        
        # 设置默认参数
        llm_params = {
            "temperature": kwargs.get("temperature", self.config.get_config("temperature", 0.7)),
            "max_tokens": kwargs.get("max_tokens", self.config.get_config("max_tokens", 16384))
        }
        
        # 添加其他参数
        llm_params.update(kwargs)
        
        try:
            print(f"🤖 开始调用LLM...")
            response = self.llm_client.chat_completion(messages, **llm_params)
            
            if "error" in response:
                error_msg = response['error']
                print(f"❌ LLM调用失败: {error_msg}")
                return f"调用失败: {error_msg}"
            
            if "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"]["content"]
                print(f"✅ LLM响应成功，长度: {len(content):,} 字符")
                return content
            else:
                print(f"❌ 无效的API响应格式: {response}")
                return "无效的API响应格式"
                
        except Exception as e:
            error_msg = f"LLM调用异常: {str(e)}"
            print(f"💥 {error_msg}")
            return error_msg
    
    def _truncate_messages(self, messages: List[Dict[str, str]], max_length: int) -> List[Dict[str, str]]:
        """
        截断消息以适应长度限制
        
        优先保留系统消息和最后的用户消息，中间内容适当截断
        """
        if not messages:
            return messages
        
        # 计算当前总长度
        current_length = sum(len(msg.get("content", "")) for msg in messages)
        if current_length <= max_length:
            return messages
        
        # 保留系统消息（通常是第一条）
        truncated_messages = []
        system_msg = None
        user_msgs = []
        
        for msg in messages:
            if msg.get("role") == "system":
                system_msg = msg
            else:
                user_msgs.append(msg)
        
        # 添加系统消息
        if system_msg:
            truncated_messages.append(system_msg)
            remaining_length = max_length - len(system_msg.get("content", ""))
        else:
            remaining_length = max_length
        
        # 处理用户消息，从最后一条开始保留
        for msg in reversed(user_msgs):
            content = msg.get("content", "")
            if len(content) <= remaining_length:
                truncated_messages.insert(-1 if system_msg else 0, msg)
                remaining_length -= len(content)
            else:
                # 截断这条消息
                truncated_content = content[:remaining_length-100] + "\n\n[... 内容已截断 ...]"
                truncated_msg = msg.copy()
                truncated_msg["content"] = truncated_content
                truncated_messages.insert(-1 if system_msg else 0, truncated_msg)
                break
        
        return truncated_messages

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        处理输入数据（抽象方法）
        
        参数:
            input_data: 输入数据
        
        返回:
            Any: 处理结果
        """
        pass

    def _setup_llm_client(self) -> None:
        """设置LLM客户端"""
        try:
            api_key = self.config.get_config("api_key")
            base_url = self.config.get_config("base_url")
            model = self.config.get_config("model", "gpt-3.5-turbo")
            
            if api_key and base_url:
                self.llm_client = LLMClient(api_key, base_url, model)
            else:
                print("⚠️ 警告: 缺少API配置，请设置环境变量 AGENT_API_KEY 和 AGENT_BASE_URL")
        except Exception as e:
            print(f"❌ LLM客户端初始化失败: {e}") 