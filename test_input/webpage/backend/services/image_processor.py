import os
import io
from PIL import Image
from rembg import remove
import numpy as np
from typing import Tuple, Optional

class ImageProcessor:
    def __init__(self, config):
        """Initialize the image processor with configuration settings."""
        self.config = config
        self.supported_formats = {'.jpg', '.jpeg', '.png'}
        self.max_size = (1920, 1920)  # Maximum image dimensions

    def process_image(self, image_data: bytes, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> bytes:
        """
        Process the image by removing background and replacing it with specified color.
        
        Args:
            image_data: Raw image data in bytes
            bg_color: RGB tuple for background color (default: white)
            
        Returns:
            Processed image data in bytes
        """
        try:
            # Convert bytes to PIL Image
            input_image = Image.open(io.BytesIO(image_data))
            
            # Convert RGBA to RGB if needed
            if input_image.mode == 'RGBA':
                input_image = self._flatten_alpha(input_image, bg_color)
            
            # Resize if image is too large
            input_image = self._resize_if_needed(input_image)
            
            # Remove background
            output_image = remove(input_image)
            
            # Replace transparent background with specified color
            final_image = self._replace_background(output_image, bg_color)
            
            # Convert back to bytes
            output_buffer = io.BytesIO()
            final_image.save(output_buffer, format='PNG')
            return output_buffer.getvalue()
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to process image: {str(e)}")

    def _flatten_alpha(self, image: Image.Image, bg_color: Tuple[int, int, int]) -> Image.Image:
        """Flatten RGBA image onto a background of the specified color."""
        if image.mode != 'RGBA':
            return image
            
        background = Image.new('RGB', image.size, bg_color)
        background.paste(image, mask=image.split()[3])
        return background

    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        """Resize image if it exceeds maximum dimensions while maintaining aspect ratio."""
        if image.size[0] > self.max_size[0] or image.size[1] > self.max_size[1]:
            image.thumbnail(self.max_size, Image.Resampling.LANCZOS)
        return image

    def _replace_background(self, image: Image.Image, bg_color: Tuple[int, int, int]) -> Image.Image:
        """Replace transparent background with specified color."""
        if image.mode != 'RGBA':
            return image
            
        background = Image.new('RGB', image.size, bg_color)
        background.paste(image, mask=image.split()[3])
        return background

    def validate_image(self, image_data: bytes) -> bool:
        """
        Validate image format and dimensions.
        
        Args:
            image_data: Raw image data in bytes
            
        Returns:
            bool: True if image is valid, False otherwise
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            extension = os.path.splitext(image.filename)[1].lower() if image.filename else '.png'
            
            if extension not in self.supported_formats:
                return False
                
            min_dimension = self.config.get('MIN_IMAGE_DIMENSION', 100)
            if image.size[0] < min_dimension or image.size[1] < min_dimension:
                return False
                
            return True
        except Exception:
            return False


class ImageProcessingError(Exception):
    """Custom exception for image processing errors."""
    pass