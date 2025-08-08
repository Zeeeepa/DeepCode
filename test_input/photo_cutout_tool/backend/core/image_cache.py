"""
Image caching functionality for the photo cutout tool.
Provides caching mechanisms to store and retrieve processed images.
"""
import os
import time
from pathlib import Path
from typing import Optional, Dict, Union
import hashlib
import json
from datetime import datetime, timedelta

class ImageCache:
    def __init__(self, cache_dir: Union[str, Path], max_age_hours: int = 24):
        """
        Initialize the image cache.
        
        Args:
            cache_dir: Directory to store cached images
            max_age_hours: Maximum age of cached files in hours before they expire
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = timedelta(hours=max_age_hours)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata: Dict = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load cache metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_metadata(self):
        """Save cache metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)

    def _generate_cache_key(self, image_path: Union[str, Path], operation: str, **params) -> str:
        """
        Generate a unique cache key based on image content and processing parameters.
        
        Args:
            image_path: Path to the original image
            operation: Type of operation (e.g., 'remove_bg', 'replace_bg')
            params: Additional parameters that affect the operation
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Create a unique hash based on image content and parameters
        hasher = hashlib.sha256()
        with open(image_path, 'rb') as f:
            hasher.update(f.read())
        
        # Add operation and parameters to the hash
        param_str = f"{operation}:{sorted(str(params.items()))}"
        hasher.update(param_str.encode())
        
        return hasher.hexdigest()

    def get_cached_path(self, image_path: Union[str, Path], operation: str, **params) -> Optional[Path]:
        """
        Get the path to a cached image if it exists and is not expired.
        
        Args:
            image_path: Path to the original image
            operation: Type of operation
            params: Additional parameters that affect the operation
        
        Returns:
            Path to cached image if available, None otherwise
        """
        cache_key = self._generate_cache_key(image_path, operation, **params)
        
        if cache_key not in self.metadata:
            return None
            
        cache_info = self.metadata[cache_key]
        cache_path = Path(cache_info['path'])
        
        # Check if cache has expired
        cache_time = datetime.fromisoformat(cache_info['timestamp'])
        if datetime.now() - cache_time > self.max_age:
            self._remove_cache_entry(cache_key)
            return None
            
        # Check if cached file exists
        if not cache_path.exists():
            self._remove_cache_entry(cache_key)
            return None
            
        return cache_path

    def cache_image(self, original_path: Union[str, Path], processed_path: Union[str, Path], 
                    operation: str, **params) -> Path:
        """
        Cache a processed image.
        
        Args:
            original_path: Path to the original image
            processed_path: Path to the processed image
            operation: Type of operation performed
            params: Parameters used in the operation
        
        Returns:
            Path to the cached image
        """
        cache_key = self._generate_cache_key(original_path, operation, **params)
        
        # Create cache file path with original extension
        ext = Path(processed_path).suffix
        cache_path = self.cache_dir / f"{cache_key}{ext}"
        
        # Copy processed image to cache
        with open(processed_path, 'rb') as src, open(cache_path, 'wb') as dst:
            dst.write(src.read())
        
        # Update metadata
        self.metadata[cache_key] = {
            'path': str(cache_path),
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'params': params
        }
        self._save_metadata()
        
        return cache_path

    def _remove_cache_entry(self, cache_key: str):
        """Remove a cache entry and its associated file."""
        if cache_key in self.metadata:
            cache_path = Path(self.metadata[cache_key]['path'])
            if cache_path.exists():
                cache_path.unlink()
            del self.metadata[cache_key]
            self._save_metadata()

    def cleanup_expired(self):
        """Remove all expired cache entries and their files."""
        current_time = datetime.now()
        expired_keys = []
        
        for key, info in self.metadata.items():
            cache_time = datetime.fromisoformat(info['timestamp'])
            if current_time - cache_time > self.max_age:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_cache_entry(key)

    def clear_cache(self):
        """Clear all cached images and metadata."""
        for key in list(self.metadata.keys()):
            self._remove_cache_entry(key)
        
        # Remove metadata file
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        
        self.metadata = {}