"""
Configuration settings for the photo cutout tool backend.
"""
import os
from pathlib import Path
from typing import Dict, Any

# Try importing BaseSettings with fallback support for different pydantic versions
try:
    # First try pydantic_settings (pydantic v2.x)
    from pydantic_settings import BaseSettings
except ImportError:
    try:
        # Fallback to pydantic (pydantic v1.x)
        from pydantic import BaseSettings
    except ImportError:
        # Provide a basic fallback implementation if neither is available
        class BaseSettings:
            """Basic fallback implementation of BaseSettings."""
            
            def __init__(self, **kwargs):
                # Set default values from class attributes
                for key, value in self.__class__.__dict__.items():
                    if not key.startswith('_') and not callable(value):
                        setattr(self, key, value)
                
                # Override with provided kwargs
                for key, value in kwargs.items():
                    setattr(self, key, value)
                
                # Load from environment variables
                self._load_from_env()
            
            def _load_from_env(self):
                """Load configuration from environment variables."""
                for key, value in self.__class__.__dict__.items():
                    if not key.startswith('_') and not callable(value):
                        env_value = os.getenv(key)
                        if env_value is not None:
                            # Try to convert to the same type as the default value
                            if isinstance(value, bool):
                                setattr(self, key, env_value.lower() in ('true', '1', 'yes', 'on'))
                            elif isinstance(value, int):
                                try:
                                    setattr(self, key, int(env_value))
                                except ValueError:
                                    pass
                            elif isinstance(value, list):
                                # Simple list parsing for comma-separated values
                                setattr(self, key, [item.strip() for item in env_value.split(',')])
                            elif isinstance(value, set):
                                # Simple set parsing for comma-separated values
                                setattr(self, key, {item.strip() for item in env_value.split(',')})
                            else:
                                setattr(self, key, env_value)
            
            class Config:
                case_sensitive = True

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "ID Photo Background Manager"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "A web-based tool for removing and replacing backgrounds in ID photos"
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8000"]
    
    # File Processing Settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {"png", "jpg", "jpeg"}
    
    # Directory Settings - Updated to match usage in image_processor.py
    UPLOAD_DIR: str = "uploads"  # Renamed from UPLOAD_FOLDER for consistency
    PROCESSED_DIR: str = "processed"  # Renamed from OUTPUT_FOLDER for consistency
    CACHE_DIR: str = "cache"  # Added missing cache directory configuration
    
    # Backward compatibility aliases (deprecated)
    UPLOAD_FOLDER: str = "uploads"  # Kept for backward compatibility
    OUTPUT_FOLDER: str = "processed"  # Kept for backward compatibility
    
    # Image Processing Settings
    DEFAULT_BG_COLOR: str = "#FFFFFF"
    MAX_IMAGE_DIMENSION: int = 4096
    JPEG_QUALITY: int = 95
    
    # Cache Settings
    ENABLE_CACHE: bool = True
    CACHE_EXPIRY: int = 3600  # 1 hour
    
    # Security Settings
    RATE_LIMIT: int = 100  # requests per minute
    
    class Config:
        case_sensitive = True

    def create_folders(self):
        """Create necessary folders for file storage."""
        # Updated to use new directory names and include cache directory
        for folder in [self.UPLOAD_DIR, self.PROCESSED_DIR, self.CACHE_DIR]:
            Path(folder).mkdir(parents=True, exist_ok=True)

# Create global settings instance
settings = Settings()

# Ensure storage directories exist
settings.create_folders()

def get_settings() -> Settings:
    """Dependency injection for settings."""
    return settings

def validate_file_extension(filename: str) -> bool:
    """Validate if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in settings.ALLOWED_EXTENSIONS

def get_file_path(folder: str, filename: str) -> str:
    """Get the full path for a file in a specific folder."""
    return os.path.join(folder, filename)