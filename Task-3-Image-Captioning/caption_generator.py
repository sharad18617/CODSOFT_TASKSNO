"""
Core Image Captioning Module using Salesforce BLIP (Vision Transformer + Language Decoder).
Supports unconditional and conditional captioning with configurable beam search.
"""

import time
import argparse
from typing import Dict, Any, Optional
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

try:
    from utils import get_device, get_device_name, validate_and_load_image
except ImportError:
    from .utils import get_device, get_device_name, validate_and_load_image


class ImageCaptioner:
    """
    Wrapper around Hugging Face BLIP (Bootstrapping Language-Image Pre-training)
    for extracting visual features and generating natural language descriptions.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: Optional[torch.device] = None,
    ):
        self.model_name = model_name
        self.device = device if device is not None else get_device()
        self.device_name = get_device_name(self.device)
        self.processor: Optional[BlipProcessor] = None
        self.model: Optional[BlipForConditionalGeneration] = None
        self.is_loaded = False

    def load_model(self) -> None:
        """
        Loads the pre-trained BLIP Processor and Model weights.
        Transfers model to the detected compute device (MPS/CUDA/CPU).
        """
        if self.is_loaded:
            return

        print(f"Loading BLIP Model [{self.model_name}] on device: {self.device_name}...")
        
        # Load processor (handles image resizing, normalization, and text tokenization)
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        
        # Load model weights
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        self.is_loaded = True
        print("Model loaded successfully!")

    def generate_caption(
        self,
        image: Image.Image,
        prefix_prompt: Optional[str] = None,
        num_beams: int = 3,
        max_new_tokens: int = 50,
        min_new_tokens: int = 5,
    ) -> Dict[str, Any]:
        """
        Generates a natural-language caption for the provided image.

        Args:
            image: PIL.Image object in RGB mode.
            prefix_prompt: Optional text prefix for guided / conditional captioning.
            num_beams: Number of beams for beam search decoding (higher = more accurate).
            max_new_tokens: Maximum number of tokens/words to generate.
            min_new_tokens: Minimum number of tokens/words to generate.

        Returns:
            Dictionary containing:
                - 'caption': The generated description string.
                - 'latency_seconds': Time taken to generate the caption in seconds.
                - 'device': Name of the device used for inference.
        """
        if not self.is_loaded:
            self.load_model()

        start_time = time.perf_counter()

        # Ensure image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 1. Preprocess: Convert image (and optional text prompt) into PyTorch tensors
        if prefix_prompt and prefix_prompt.strip():
            inputs = self.processor(
                images=image,
                text=prefix_prompt.strip(),
                return_tensors="pt"
            ).to(self.device)
        else:
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)

        # 2. Inference: Generate caption token IDs using Vision-Language Transformer
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                early_stopping=True,
            )

        # 3. Post-process: Decode token IDs into human-readable text
        raw_caption = self.processor.decode(output_tokens[0], skip_special_tokens=True)
        
        # Clean up caption formatting
        caption = raw_caption.strip()
        if caption:
            caption = caption[0].upper() + caption[1:]  # Capitalize first letter

        end_time = time.perf_counter()
        latency = round(end_time - start_time, 3)

        return {
            "caption": caption,
            "latency_seconds": latency,
            "device": self.device_name,
        }


def main():
    """Command-line interface for testing caption generation directly."""
    parser = argparse.ArgumentParser(description="AI Image Captioning CLI using BLIP")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--prefix", type=str, default=None, help="Optional prefix prompt")
    parser.add_argument("--beams", type=int, default=3, help="Number of beams for beam search")
    args = parser.parse_args()

    image, err = validate_and_load_image(args.image)
    if err:
        print(f"Error: {err}")
        return

    captioner = ImageCaptioner()
    captioner.load_model()
    result = captioner.generate_caption(image, prefix_prompt=args.prefix, num_beams=args.beams)

    print("\n" + "=" * 50)
    print(f"Caption : {result['caption']}")
    print(f"Latency : {result['latency_seconds']} seconds")
    print(f"Device  : {result['device']}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
