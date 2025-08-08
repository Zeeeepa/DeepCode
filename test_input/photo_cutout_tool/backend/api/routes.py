from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from typing import Optional
import shutil
import os

from core.image_processor import ImageProcessor
from core.config import get_settings

router = APIRouter()

# Use configured directories from settings with get_settings() function
settings = get_settings()
UPLOAD_DIR = Path(settings.UPLOAD_DIR)
PROCESSED_DIR = Path(settings.PROCESSED_DIR)
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Create ImageProcessor instance
image_processor = ImageProcessor()

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image file"""
    try:
        filename, file_path = await image_processor.save_upload_file(file)
        return JSONResponse({
            "status": "success",
            "message": "File uploaded successfully",
            "filename": filename,
            "file_path": str(file_path)
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/remove-background/{filename}")
async def process_remove_background(filename: str):
    """Remove background from an uploaded image"""
    try:
        input_path = UPLOAD_DIR / filename
        if not input_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        output_path = PROCESSED_DIR / f"nobg_{filename}"
        result_path = image_processor.remove_background(input_path, output_path)
        
        return FileResponse(
            path=str(result_path),
            filename=f"nobg_{filename}",
            media_type="image/png"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/replace-background/{filename}")
async def process_replace_background(
    filename: str,
    bg_color: Optional[str] = Form(None),
    bg_image: Optional[UploadFile] = File(None)
):
    """Replace background with color or image"""
    try:
        input_path = UPLOAD_DIR / filename
        if not input_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        bg_image_path = None
        if bg_image:
            _, bg_image_path = await image_processor.save_upload_file(bg_image)

        # Convert hex color to RGB if provided
        bg_color_rgb = None
        if bg_color:
            bg_color = bg_color.lstrip('#')
            bg_color_rgb = tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4))

        output_path = PROCESSED_DIR / f"replaced_{filename}"
        result_path = image_processor.replace_background(
            input_path,
            bg_color=bg_color_rgb,
            bg_image_path=bg_image_path
        )

        return FileResponse(
            path=str(result_path),
            filename=f"replaced_{filename}",
            media_type="image/png"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/optimize/{filename}")
async def process_optimize_image(
    filename: str,
    max_size: Optional[int] = Form(None)
):
    """Optimize image size while maintaining quality"""
    try:
        input_path = UPLOAD_DIR / filename
        if not input_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        output_path = PROCESSED_DIR / f"optimized_{filename}"
        result_path = image_processor.optimize_image(input_path, max_size)

        return FileResponse(
            path=str(result_path),
            filename=f"optimized_{filename}"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/info/{filename}")
async def get_image_details(filename: str):
    """Get information about an image"""
    try:
        image_path = UPLOAD_DIR / filename
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        info = image_processor.get_image_info(image_path)
        return JSONResponse(info)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/cleanup/{filename}")
async def cleanup_files(filename: str):
    """Delete processed files for cleanup"""
    try:
        patterns = [
            UPLOAD_DIR / filename,
            PROCESSED_DIR / f"nobg_{filename}",
            PROCESSED_DIR / f"replaced_{filename}",
            PROCESSED_DIR / f"optimized_{filename}"
        ]
        
        deleted = []
        for path in patterns:
            if path.exists():
                os.remove(path)
                deleted.append(str(path))

        return JSONResponse({
            "status": "success",
            "message": "Files cleaned up successfully",
            "deleted_files": deleted
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))