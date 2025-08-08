from io import BytesIO
import os
from pathlib import Path
from typing import Optional, Tuple, Union
import uuid

import numpy as np
from PIL import Image
from rembg import remove
from fastapi import UploadFile

from core.config import Settings, get_settings

class ImageProcessor:
    def __init__(self, settings: Settings = get_settings()):
        self.settings = settings
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure required directories exist."""
        Path(self.settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.settings.PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.settings.CACHE_DIR).mkdir(parents=True, exist_ok=True)

    async def save_upload_file(self, file: UploadFile) -> Tuple[str, str]:
        """Save uploaded file and return the filename and file path."""
        ext = os.path.splitext(file.filename)[1].lower()
        filename = f"{uuid.uuid4()}{ext}"
        file_path = os.path.join(self.settings.UPLOAD_DIR, filename)
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        return filename, file_path

    def remove_background(self, 
                         input_path: Union[str, Path], 
                         output_path: Optional[Union[str, Path]] = None) -> Path:
        """Remove background from image using rembg."""
        # Read input image
        with open(input_path, 'rb') as f:
            input_data = f.read()
        
        # Process image with rembg
        output_data = remove(input_data)
        
        # If no output path specified, create one in processed directory
        if output_path is None:
            filename = f"{uuid.uuid4()}_nobg.png"
            output_path = Path(self.settings.PROCESSED_DIR) / filename
        
        # Save processed image
        with open(output_path, 'wb') as f:
            f.write(output_data)
        
        return Path(output_path)

    def replace_background(self,
                         image_path: Union[str, Path],
                         bg_color: Optional[Tuple[int, int, int]] = None,
                         bg_image_path: Optional[Union[str, Path]] = None) -> Path:
        """Replace background with solid color or another image."""
        # Load the image with removed background
        img = Image.open(image_path)
        
        # Create new image with same size
        if bg_color:
            # Create solid color background
            background = Image.new('RGB', img.size, bg_color)
        elif bg_image_path:
            # Load and resize background image
            background = Image.open(bg_image_path)
            background = background.resize(img.size, Image.Resampling.LANCZOS)
        else:
            bg_color = (255, 255, 255)  # Default white
            background = Image.new('RGB', img.size, bg_color)
        
        # Ensure image has alpha channel
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Composite images
        background.paste(img, (0, 0), img)
        
        # Save result
        output_filename = f"{uuid.uuid4()}_replaced.png"
        output_path = Path(self.settings.PROCESSED_DIR) / output_filename
        background.save(output_path, 'PNG')
        
        return output_path

    def optimize_image(self, 
                      image_path: Union[str, Path], 
                      max_size: Optional[int] = None) -> Path:
        """Optimize image size while maintaining quality."""
        img = Image.open(image_path)
        
        # Resize if max_size is specified and image is larger
        if max_size and (img.width > max_size or img.height > max_size):
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Save optimized image
        output_filename = f"{uuid.uuid4()}_optimized.png"
        output_path = Path(self.settings.PROCESSED_DIR) / output_filename
        
        # Save with optimization
        img.save(output_path, 'PNG', optimize=True)
        
        return output_path

    def get_image_info(self, image_path: Union[str, Path]) -> dict:
        """Get image information including size, format, and mode."""
        img = Image.open(image_path)
        return {
            'width': img.width,
            'height': img.height,
            'format': img.format,
            'mode': img.mode,
            'size_bytes': os.path.getsize(image_path)
        }