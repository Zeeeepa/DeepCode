"""
LLM API客户端

支持多种LLM服务提供商，包括OpenRouter等中转服务。

主要功能:
- chat_completion(): 发送对话补全请求
- 支持自动重试机制
- 增强的错误处理和超时控制
"""

from typing import Dict, Any, List
import requests
import time
import json


class LLMClient:
    """
    LLM API客户端
    
    支持OpenRouter、OpenAI等多种API服务。
    通过配置base_url可以轻松切换不同的服务提供商。
    """
    
    def __init__(self, api_key: str, base_url: str, model: str = "gpt-3.5-turbo"):
        """
        初始化LLM客户端
        
        参数:
            api_key (str): API密钥
            base_url (str): API基础URL
            model (str): 模型名称
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = 120  # 增加超时时间到120秒
        self.max_retries = 3  # 最大重试次数
        self.retry_delay = 2  # 重试间隔（秒）

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        发送对话补全请求（带重试机制）
        
        参数:
            messages (list): 消息列表，格式如 [{"role": "user", "content": "你好"}]
            **kwargs: 其他参数（temperature, max_tokens等）
        
        返回:
            dict: API响应
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 16384),
        }
        
        # 添加其他支持的参数
        for param in ["top_p", "frequency_penalty", "presence_penalty", "stop"]:
            if param in kwargs:
                payload[param] = kwargs[param]
        
        # 计算请求大小，用于调试
        request_size = len(json.dumps(payload).encode('utf-8'))
        print(f"🔍 LLM请求大小: {request_size:,} 字节")
        
        last_error = None
        
        # 重试机制
        for attempt in range(self.max_retries):
            try:
                print(f"🔄 LLM请求尝试 {attempt + 1}/{self.max_retries}")
                
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=payload, 
                    timeout=self.timeout,
                    stream=False  # 确保不使用流式响应
                )
                
                # 检查HTTP状态码
                if response.status_code != 200:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    print(f"❌ HTTP错误: {error_msg}")
                    last_error = error_msg
                    
                    # 如果是客户端错误（4xx），不重试
                    if 400 <= response.status_code < 500:
                        break
                    
                    # 服务器错误（5xx）或其他错误，继续重试
                    if attempt < self.max_retries - 1:
                        print(f"⏳ {self.retry_delay}秒后重试...")
                        time.sleep(self.retry_delay)
                    continue
                
                # 尝试解析JSON响应
                try:
                    result = response.json()
                    print(f"✅ LLM请求成功")
                    return result
                except json.JSONDecodeError as e:
                    error_msg = f"JSON解析失败: {str(e)}, 响应内容: {response.text[:200]}..."
                    print(f"❌ JSON解析错误: {error_msg}")
                    last_error = error_msg
                    
                    if attempt < self.max_retries - 1:
                        print(f"⏳ {self.retry_delay}秒后重试...")
                        time.sleep(self.retry_delay)
                    continue
                    
            except requests.exceptions.Timeout as e:
                error_msg = f"请求超时 ({self.timeout}秒): {str(e)}"
                print(f"⏰ 超时错误: {error_msg}")
                last_error = error_msg
                
            except requests.exceptions.ConnectionError as e:
                error_msg = f"连接错误: {str(e)}"
                print(f"🔌 连接错误: {error_msg}")
                last_error = error_msg
                
            except requests.exceptions.RequestException as e:
                error_msg = f"请求异常: {str(e)}"
                print(f"❌ 请求异常: {error_msg}")
                last_error = error_msg
                
            except Exception as e:
                error_msg = f"未知错误: {str(e)}"
                print(f"⚠️ 未知错误: {error_msg}")
                last_error = error_msg
            
            # 如果不是最后一次尝试，等待后重试
            if attempt < self.max_retries - 1:
                print(f"⏳ {self.retry_delay}秒后重试...")
                time.sleep(self.retry_delay)
                # 指数退避：每次重试增加延迟时间
                self.retry_delay *= 1.5
        
        # 所有重试都失败了
        print(f"💥 所有 {self.max_retries} 次重试都失败了")
        return {
            "error": f"API请求失败（{self.max_retries}次重试后）: {last_error}",
            "choices": [{"message": {"content": f"调用失败: {last_error}"}}]
        } 