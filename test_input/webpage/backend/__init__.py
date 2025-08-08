"""
ID Photo Tool Backend Package

This package provides the backend services for the ID photo background removal and replacement tool.
It includes image processing, storage handling, and API endpoints.
"""

from flask import Flask
from flask_cors import CORS
from .config import Config

def create_app(config_class=Config):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints and configure routes
    from .app import bp as main_bp
    app.register_blueprint(main_bp)
    
    return app