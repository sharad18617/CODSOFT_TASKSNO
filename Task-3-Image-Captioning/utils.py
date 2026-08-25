"""
Utility functions for AI Image Captioning System.
Provides device detection (Apple Silicon MPS / CUDA / CPU) and image validation helpers.
"""

from typing import Tuple, Optional
import io
from PIL import Image, ImageOps


def get_device() -> torch.device:
    """
    Detect and return the optimal compute device.
    Prioritizes:
      1. Apple Silicon GPU (MPS - Metal Performance Shaders) on macOS
      2. NVIDIA CUDA GPU (if available on Linux/Windows)
      3. Standard CPU fallback
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_device_name(device: torch.device) -> str:
    """
    Returns a human-readable display string for the detected device.
    """
    device_type = device.type
    if device_type == "mps":
        return "Apple Silicon GPU (MPS Accelerated)"
    elif device_type == "cuda":
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA"
        return f"NVIDIA GPU ({gpu_name})"
    else:
        return "CPU (Standard Processor)"


def validate_and_load_image(image_input) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Validates and loads an image from an uploaded file or file path.
    - Handles EXIF rotation metadata from mobile camera photos.
    - Converts image to RGB mode.
    - Caps excessive phone camera resolutions (e.g. 48MP) to prevent cloud memory overflow.

    Args:
        image_input: Streamlit UploadedFile, file-like object, or string file path.

    Returns:
        Tuple of (PIL.Image.Image or None, error_message or None)
    """
    if image_input is None:
        return None, "No image provided."

    try:
        if isinstance(image_input, (str, io.BytesIO)) or hasattr(image_input, "read"):
            image = Image.open(image_input)

            # Auto-rotate phone camera photos according to EXIF orientation tag
            try:
                image = ImageOps.exif_transpose(image)
            except Exception:
                pass
            
            # Supported standard formats
            supported_formats = {"JPEG", "JPG", "PNG", "WEBP", "BMP", "TIFF", "MPO"}
            if image.format and image.format.upper() not in supported_formats:
                return None, f"Unsupported image format: '{image.format}'. Please use JPG, JPEG, PNG, or WEBP."
            
            # Convert to RGB (handles RGBA, grayscale, CMYK, etc.)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Downsample ultra-high-resolution mobile photos (max dimension 1920px) to prevent cloud RAM crash
            max_dimension = 1920
            if max(image.width, image.height) > max_dimension:
                image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                
            return image, None
        else:
            return None, "Invalid image input type."
            
    except Exception as e:
        return None, f"Could not process the image: {str(e)}"
