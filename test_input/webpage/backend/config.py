"""
Configuration settings for the ID Photo Tool backend application.
"""
import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Flask application settings
class Config:
    # Secret key for session management
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    
    # File upload settings
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # Temporary file settings
    TEMP_FOLDER = os.path.join(BASE_DIR, 'temp')
    
    # CORS settings
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
    
    # Image processing settings
    DEFAULT_BG_COLOR = (255, 255, 255)  # White background
    OUTPUT_FORMAT = 'PNG'
    
    @staticmethod
    def init_app(app):
        """Initialize application with the config settings."""
        # Create upload and temp directories if they don't exist
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.TEMP_FOLDER, exist_ok=True)
        
        # Set Flask config
        app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
        app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
        app.config['SECRET_KEY'] = Config.SECRET_KEY

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False

class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = False
    TESTING = True
    # Use temporary directories for testing
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'test', 'uploads')
    TEMP_FOLDER = os.path.join(BASE_DIR, 'test', 'temp')

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    # In production, secret key should be set through environment variable
    SECRET_KEY = os.getenv('SECRET_KEY')
    # Stricter CORS settings for production
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'https://yourdomain.com')

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}