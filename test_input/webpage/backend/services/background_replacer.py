from typing import Tuple, Optional, Union
import io
import numpy as np
from PIL import Image
from .image_processor import process_image
from ..utils.validators import validate_color_code

class BackgroundReplacer:
    """Service for handling background replacement operations in photos."""
    
    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color code to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def validate_background_color(color: Union[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
        """Validate and convert background color to RGB tuple."""
        if isinstance(color, str):
            if not validate_color_code(color):
                raise ValueError("Invalid hex color code")
            return BackgroundReplacer.hex_to_rgb(color)
        elif isinstance(color, tuple) and len(color) == 3:
            if not all(isinstance(c, int) and 0 <= c <= 255 for c in color):
                raise ValueError("RGB values must be integers between 0 and 255")
            return color
        else:
            raise ValueError("Color must be a hex string or RGB tuple")

    def replace_background(
        self,
        image_data: bytes,
        background_color: Union[str, Tuple[int, int, int]] = "#FFFFFF",
        maintain_size: bool = True
    ) -> bytes:
        """
        Replace the background of an image with a specified color.
        
        Args:
            image_data: Input image as bytes
            background_color: Target background color (hex string or RGB tuple)
            maintain_size: Whether to maintain original image dimensions
            
        Returns:
            Processed image as bytes
        """
        try:
            # Validate and convert background color
            bg_color = self.validate_background_color(background_color)
            
            # Process image using core image processor
            processed_image_data = process_image(image_data, bg_color)
            
            if maintain_size:
                # Ensure output maintains original dimensions
                original_image = Image.open(io.BytesIO(image_data))
                processed_image = Image.open(io.BytesIO(processed_image_data))
                
                if original_image.size != processed_image.size:
                    processed_image = processed_image.resize(
                        original_image.size, 
                        Image.Resampling.LANCZOS
                    )
                    
                    # Convert back to bytes
                    output_buffer = io.BytesIO()
                    processed_image.save(output_buffer, format='PNG')
                    processed_image_data = output_buffer.getvalue()
            
            return processed_image_data
            
        except Exception as e:
            raise RuntimeError(f"Background replacement failed: {str(e)}")

    def preview_backgrounds(
        self,
        image_data: bytes,
        colors: list[Union[str, Tuple[int, int, int]]],
        thumbnail_size: Tuple[int, int] = (200, 200)
    ) -> list[bytes]:
        """
        Generate preview thumbnails with different background colors.
        
        Args:
            image_data: Input image as bytes
            colors: List of background colors to preview
            thumbnail_size: Size of preview thumbnails
            
        Returns:
            List of processed thumbnail images as bytes
        """
        try:
            previews = []
            for color in colors:
                # Process image with current color
                processed = self.replace_background(image_data, color)
                
                # Create thumbnail
                img = Image.open(io.BytesIO(processed))
                img.thumbnail(thumbnail_size)
                
                # Convert to bytes
                output_buffer = io.BytesIO()
                img.save(output_buffer, format='PNG')
                previews.append(output_buffer.getvalue())
                
            return previews
            
        except Exception as e:
            raise RuntimeError(f"Preview generation failed: {str(e)}")