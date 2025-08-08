"""
Validators module for input validation in the ID Photo Tool application.
"""
import os
import re
from typing import Tuple, Optional, Union
from PIL import Image
from io import BytesIO

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

# Constants for validation
MAX_FILE_SIZE_MB = 10
MIN_IMAGE_DIMENSION = 100
MAX_IMAGE_DIMENSION = 4000
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
HEX_COLOR_PATTERN = re.compile(r'^#(?:[0-9a-fA-F]{3}){1,2}$')
RGB_COLOR_PATTERN = re.compile(r'^\s*\d+\s*,\s*\d+\s*,\s*\d+\s*$')

def validate_file_extension(filename: str) -> bool:
    """
    Validate if the file has an allowed extension.
    
    Args:
        filename (str): Name of the file to validate
        
    Returns:
        bool: True if extension is allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_size(file_size: int) -> bool:
    """
    Validate if the file size is within allowed limits.
    
    Args:
        file_size (int): Size of the file in bytes
        
    Returns:
        bool: True if size is within limits, False otherwise
    """
    return file_size <= (MAX_FILE_SIZE_MB * 1024 * 1024)

def validate_image_dimensions(image_data: bytes) -> Tuple[bool, Optional[str]]:
    """
    Validate if the image dimensions are within allowed limits.
    
    Args:
        image_data (bytes): Raw image data
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    try:
        img = Image.open(BytesIO(image_data))
        width, height = img.size
        
        if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
            return False, f"Image dimensions too small. Minimum size is {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION} pixels."
        
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            return False, f"Image dimensions too large. Maximum size is {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION} pixels."
        
        return True, None
        
    except Exception as e:
        return False, f"Error validating image dimensions: {str(e)}"

def validate_color_code(color: str) -> Tuple[bool, Optional[str]]:
    """
    Validate if the color code is in correct format (hex or RGB).
    
    Args:
        color (str): Color code to validate
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if not color:
        return False, "Color code cannot be empty"
    
    # Check hex color format
    if color.startswith('#'):
        if HEX_COLOR_PATTERN.match(color):
            return True, None
        return False, "Invalid hex color format. Use #RGB or #RRGGBB format."
    
    # Check RGB format
    if RGB_COLOR_PATTERN.match(color):
        try:
            r, g, b = map(int, color.split(','))
            if all(0 <= x <= 255 for x in (r, g, b)):
                return True, None
            return False, "RGB values must be between 0 and 255"
        except ValueError:
            pass
    
    return False, "Invalid color format. Use hex (#RGB/#RRGGBB) or RGB (r,g,b) format."

def validate_upload_request(file_data: bytes, filename: str) -> Tuple[bool, Optional[str]]:
    """
    Comprehensive validation for file uploads.
    
    Args:
        file_data (bytes): Raw file data
        filename (str): Name of the uploaded file
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if not filename:
        return False, "No file provided"
    
    if not validate_file_extension(filename):
        return False, f"Invalid file extension. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
    
    if not validate_file_size(len(file_data)):
        return False, f"File size exceeds maximum limit of {MAX_FILE_SIZE_MB}MB"
    
    is_valid_dims, error_msg = validate_image_dimensions(file_data)
    if not is_valid_dims:
        return False, error_msg
    
    return True, None