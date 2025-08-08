"""
API package initialization.
Exposes the routes module for the FastAPI application.
"""

from api.routes import (
    upload_image,
    process_remove_background,
    process_replace_background,
    process_optimize_image,
    get_image_details,
    cleanup_files
)

__all__ = [
    'upload_image',
    'process_remove_background',
    'process_replace_background',
    'process_optimize_image',
    'get_image_details',
    'cleanup_files'
]