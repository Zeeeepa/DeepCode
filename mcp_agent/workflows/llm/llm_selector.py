"""
Model selector for LLM workflows.
"""

from typing import Any, Dict, List, Optional


class ModelInfo:
    """
    Information about a model.
    """
    
    def __init__(
        self,
        name: str,
        provider: str,
        input_cost_per_1k: float = 0.0,
        output_cost_per_1k: float = 0.0,
    ):
        """
        Initialize model info.
        
        Args:
            name: Model name
            provider: Provider name
            input_cost_per_1k: Input cost per 1000 tokens
            output_cost_per_1k: Output cost per 1000 tokens
        """
        self.name = name
        self.provider = provider
        self.input_cost_per_1k = input_cost_per_1k
        self.output_cost_per_1k = output_cost_per_1k


class ModelSelector:
    """
    Model selector for LLM workflows.
    """
    
    def __init__(self):
        """Initialize the model selector."""
        self.models: Dict[str, ModelInfo] = {}
    
    def get_model(self, name: str) -> Optional[ModelInfo]:
        """
        Get model info by name.
        
        Args:
            name: Model name
            
        Returns:
            Model info, or None if not found
        """
        return self.models.get(name)
    
    def add_model(self, model_info: ModelInfo) -> None:
        """
        Add a model.
        
        Args:
            model_info: Model info
        """
        self.models[model_info.name] = model_info


def load_default_models() -> Dict[str, ModelInfo]:
    """
    Load default models.
    
    Returns:
        Dictionary of model info
    """
    models = {
        "gpt-4o": ModelInfo(
            name="gpt-4o",
            provider="openai",
            input_cost_per_1k=0.01,
            output_cost_per_1k=0.03,
        ),
        "claude-3-opus-20240229": ModelInfo(
            name="claude-3-opus-20240229",
            provider="anthropic",
            input_cost_per_1k=0.015,
            output_cost_per_1k=0.075,
        ),
    }
    
    return models

