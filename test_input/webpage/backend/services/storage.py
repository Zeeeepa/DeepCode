import os
import tempfile
import shutil
from typing import Optional, Tuple
from datetime import datetime, timedelta

class StorageService:
    """Service for handling temporary file storage and cleanup."""
    
    def __init__(self, temp_dir: Optional[str] = None, max_age_hours: int = 24):
        """
        Initialize the storage service.
        
        Args:
            temp_dir: Optional custom temporary directory path
            max_age_hours: Maximum age of temporary files before cleanup (default: 24 hours)
        """
        self.temp_dir = temp_dir or os.path.join(tempfile.gettempdir(), 'id_photo_tool')
        self.max_age_hours = max_age_hours
        self._ensure_temp_dir()

    def _ensure_temp_dir(self) -> None:
        """Ensure the temporary directory exists."""
        os.makedirs(self.temp_dir, exist_ok=True)

    def create_temp_file(self, prefix: str = "", suffix: str = "") -> Tuple[str, str]:
        """
        Create a temporary file and return its path.
        
        Args:
            prefix: Prefix for the temporary file name
            suffix: Suffix for the temporary file name (e.g., file extension)
            
        Returns:
            Tuple containing (file_path, file_name)
        """
        _, temp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=self.temp_dir)
        return temp_path, os.path.basename(temp_path)

    def save_uploaded_file(self, file_data: bytes, original_filename: str) -> Tuple[str, str]:
        """
        Save an uploaded file to temporary storage.
        
        Args:
            file_data: Binary file data
            original_filename: Original name of the uploaded file
            
        Returns:
            Tuple containing (file_path, file_name)
        """
        ext = os.path.splitext(original_filename)[1]
        temp_path, temp_name = self.create_temp_file(prefix="upload_", suffix=ext)
        
        with open(temp_path, 'wb') as f:
            f.write(file_data)
        
        return temp_path, temp_name

    def get_file_path(self, filename: str) -> Optional[str]:
        """
        Get the full path for a file in temporary storage.
        
        Args:
            filename: Name of the file
            
        Returns:
            Full path to the file or None if not found
        """
        full_path = os.path.join(self.temp_dir, filename)
        return full_path if os.path.exists(full_path) else None

    def cleanup_old_files(self) -> int:
        """
        Remove files older than max_age_hours.
        
        Returns:
            Number of files removed
        """
        if not os.path.exists(self.temp_dir):
            return 0

        cutoff_time = datetime.now() - timedelta(hours=self.max_age_hours)
        removed_count = 0

        for filename in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, filename)
            if os.path.isfile(file_path):
                mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                if mtime < cutoff_time:
                    try:
                        os.remove(file_path)
                        removed_count += 1
                    except OSError:
                        continue

        return removed_count

    def remove_file(self, filename: str) -> bool:
        """
        Remove a specific file from temporary storage.
        
        Args:
            filename: Name of the file to remove
            
        Returns:
            True if file was removed successfully, False otherwise
        """
        file_path = self.get_file_path(filename)
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                return True
            except OSError:
                return False
        return False

    def clear_storage(self) -> bool:
        """
        Remove all files from temporary storage.
        
        Returns:
            True if cleanup was successful, False otherwise
        """
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            self._ensure_temp_dir()
            return True
        except OSError:
            return False