import logging
import base64
import io
import asyncio
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from collections import OrderedDict
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import cv2
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import aiohttp
from .ocr_base import OCRResult, BaseOCRBackend, HTTPClient
from .models import MathFormula, MathType


logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration for image preprocessing."""
    target_dpi: int = 300
    min_size: int = 100
    max_size: int = 4096
    min_region_width: int = 20
    min_region_height: int = 10
    region_padding: int = 5
    contrast_factor: float = 1.5
    sharpness_factor: float = 1.2
    denoise_strength: int = 10
    enable_denoise: bool = True
    enable_binarization: bool = True
    

@dataclass
class OCRConfig:
    """Configuration for math OCR services."""
    mathpix_api_id: Optional[str] = None
    mathpix_api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    max_image_size: int = 4096
    max_image_bytes: int = 1024 * 1024  # 1MB
    min_confidence: float = 0.7
    enable_preprocessing: bool = True
    enable_fallback: bool = True
    cache_results: bool = True
    max_cache_entries: int = 1000
    max_cache_memory_mb: int = 100
    max_workers: int = 4
    preprocess_config: PreprocessConfig = field(default_factory=PreprocessConfig)
    

class RequestsHTTPClient(HTTPClient):
    """Default HTTP client using requests."""
    
    def __init__(self):
        self.session = requests.Session()
        
    def post(self, url: str, **kwargs) -> requests.Response:
        return self.session.post(url, **kwargs)
        
    async def post_async(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        # Not implemented for requests client
        raise NotImplementedError("Use AsyncHTTPClient for async operations")
        
    def close(self):
        self.session.close()


class AsyncHTTPClient(HTTPClient):
    """Async HTTP client using aiohttp."""
    
    def __init__(self):
        self._session = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
        
    def post(self, url: str, **kwargs) -> requests.Response:
        # Not implemented for async client
        raise NotImplementedError("Use RequestsHTTPClient for sync operations")
        
    async def post_async(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        session = await self._get_session()
        return await session.post(url, **kwargs)
        
    async def close(self):
        if self._session:
            await self._session.close()


class LRUCacheWithMemoryLimit:
    """LRU cache with memory limit."""
    
    def __init__(self, max_entries: int, max_memory_mb: int):
        self.max_entries = max_entries
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.memory_usage = 0
        
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
        
    def put(self, key: str, value: Any, size_bytes: int):
        if key in self.cache:
            self.memory_usage -= self._get_size(self.cache[key])
            
        self.cache[key] = value
        self.memory_usage += size_bytes
        
        # Evict entries if needed
        while len(self.cache) > self.max_entries or self.memory_usage > self.max_memory_bytes:
            if not self.cache:
                break
            oldest_key, oldest_value = self.cache.popitem(last=False)
            self.memory_usage -= self._get_size(oldest_value)
            
    def _get_size(self, value: Any) -> int:
        # Estimate size (simplified)
        return len(str(value).encode('utf-8'))
        
    def clear(self):
        self.cache.clear()
        self.memory_usage = 0


class ImagePreprocessor:
    """Preprocess images for better OCR accuracy."""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        
    def preprocess(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Apply preprocessing pipeline to image."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Quick check if preprocessing needed
        if self._is_high_quality(image):
            logger.debug("Image already high quality, skipping preprocessing")
            return image
            
        # Convert to RGB if necessary
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
            
        # Resize if too small or too large
        image = self._resize_image(image)
        
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
            
        # Apply enhancement pipeline
        image = self._enhance_image(image)
        
        # Denoise
        if self.config.enable_denoise:
            image = self._denoise_image(image)
            
        # Binarize
        if self.config.enable_binarization:
            image = self._binarize_image(image)
            
        return image
        
    def _is_high_quality(self, image: Image.Image) -> bool:
        """Check if image is already high quality."""
        width, height = image.size
        
        # Check resolution
        if width < self.config.min_size or height < self.config.min_size:
            return False
            
        # Check contrast (simplified)
        if image.mode == 'L':
            hist = image.histogram()
            # Check if histogram is well distributed
            low_pixels = sum(hist[:50])
            high_pixels = sum(hist[200:])
            total_pixels = sum(hist)
            
            if (low_pixels + high_pixels) / total_pixels > 0.8:
                return True
                
        return False
        
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to optimal size."""
        width, height = image.size
        
        if width < self.config.min_size or height < self.config.min_size:
            scale = self.config.min_size / min(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
        elif width > self.config.max_size or height > self.config.max_size:
            scale = self.config.max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
        return image
        
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Enhance image contrast and sharpness."""
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(self.config.contrast_factor)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(self.config.sharpness_factor)
        
        # Remove noise while preserving edges
        image = image.filter(ImageFilter.MedianFilter(size=3))
        
        return image
        
    def _denoise_image(self, image: Image.Image) -> Image.Image:
        """Remove noise from image using OpenCV."""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply Non-local Means Denoising
        denoised = cv2.fastNlMeansDenoising(
            img_array, 
            None, 
            h=self.config.denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # Convert back to PIL Image
        return Image.fromarray(denoised)
        
    def _binarize_image(self, image: Image.Image) -> Image.Image:
        """Binarize image using adaptive thresholding."""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Invert if necessary (ensure dark text on light background)
        if np.mean(binary) < 127:
            binary = 255 - binary
            
        return Image.fromarray(binary)
        
    def extract_math_regions(self, image: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """Extract potential math formula regions from image."""
        # Convert to numpy array
        img_array = np.array(image.convert('L'))
        
        # Find contours
        _, binary = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        height, width = img_array.shape
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out very small regions
            if w < self.config.min_region_width or h < self.config.min_region_height:
                continue
                
            # Filter out very large regions (likely not formulas)
            if w > width * 0.9 or h > height * 0.9:
                continue
                
            # Add padding
            pad = self.config.region_padding
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(width - x, w + 2 * pad)
            h = min(height - y, h + 2 * pad)
            
            # Extract region
            region = image.crop((x, y, x + w, y + h))
            regions.append((region, (x, y, w, h)))
            
        return regions


class MathpixOCR(BaseOCRBackend):
    """Mathpix OCR service integration."""
    
    def __init__(self, config: OCRConfig, http_client: Optional[HTTPClient] = None):
        self.config = config
        self.base_url = "https://api.mathpix.com/v3/text"
        self.headers = {
            "app_id": config.mathpix_api_id or "",
            "app_key": config.mathpix_api_key or "",
            "Content-Type": "application/json"
        }
        self.http_client = http_client or RequestsHTTPClient()
        self._async_client = None
        
    def is_available(self) -> bool:
        """Check if Mathpix service is configured."""
        return bool(self.config.mathpix_api_id and self.config.mathpix_api_key)
        
    def _prepare_image(self, image: Image.Image) -> str:
        """Prepare image for API request."""
        # Resize if too large
        buffered = io.BytesIO()
        image.save(buffered, format="PNG", optimize=True)
        
        # Check size and resize if needed
        if buffered.tell() > self.config.max_image_bytes:
            # Resize to reduce file size
            max_dimension = 2048
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG", optimize=True, quality=85)
            
        buffered.seek(0)
        return base64.b64encode(buffered.getvalue()).decode()
        
    def recognize(self, image: Image.Image) -> OCRResult:
        """Recognize math formula using Mathpix API."""
        if not self.is_available():
            return OCRResult(
                latex="",
                confidence=0.0,
                error="Mathpix API credentials not configured"
            )
            
        try:
            # Prepare request
            image_base64 = self._prepare_image(image)
            data = {
                "src": f"data:image/png;base64,{image_base64}",
                "formats": ["latex_styled", "mathml", "asciimath"],
                "ocr": ["math", "text"],
                "skip_recrop": False,
                "include_detected_alphabets": True,
                "include_line_data": True
            }
            
            # Make request with retries
            response = None
            last_error = None
            
            for attempt in range(self.config.max_retries):
                try:
                    if isinstance(self.http_client, RequestsHTTPClient):
                        self.http_client.session.headers.update(self.headers)
                        
                    response = self.http_client.post(
                        self.base_url,
                        json=data,
                        timeout=self.config.timeout
                    )
                    
                    if response.status_code == 429:
                        # Rate limited
                        retry_after = int(response.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limited, waiting {retry_after}s")
                        time.sleep(retry_after)
                        continue
                        
                    response.raise_for_status()
                    break
                    
                except requests.exceptions.RequestException as e:
                    last_error = e
                    if attempt < self.config.max_retries - 1:
                        delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                        time.sleep(delay)
                        continue
                    else:
                        raise
                        
            if response is None:
                raise Exception(f"Failed after {self.config.max_retries} attempts: {last_error}")
                
            result_data = response.json()
            
            # Extract results
            latex = result_data.get("latex_styled", "")
            confidence = result_data.get("latex_confidence", 0.0)
            
            # Extract alternative formats
            alternatives = {}
            if "mathml" in result_data:
                alternatives["mathml"] = result_data["mathml"]
            if "asciimath" in result_data:
                alternatives["asciimath"] = result_data["asciimath"]
                
            # Extract metadata
            metadata = {
                "detected_alphabets": result_data.get("detected_alphabets", []),
                "is_printed": result_data.get("is_printed", True),
                "is_inverted": result_data.get("is_inverted", False),
                "line_data": result_data.get("line_data", [])
            }
            
            return OCRResult(
                latex=latex,
                confidence=confidence,
                alternative_formats=alternatives,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Mathpix OCR error: {str(e)[:200]}")  # Limit error message length
            return OCRResult(
                latex="",
                confidence=0.0,
                error=str(e)
            )
            
    async def recognize_async(self, image: Image.Image) -> OCRResult:
        """Recognize math formula asynchronously."""
        if not self.is_available():
            return OCRResult(
                latex="",
                confidence=0.0,
                error="Mathpix API credentials not configured"
            )
            
        try:
            # Prepare request
            image_base64 = self._prepare_image(image)
            data = {
                "src": f"data:image/png;base64,{image_base64}",
                "formats": ["latex_styled", "mathml", "asciimath"],
                "ocr": ["math", "text"]
            }
            
            # Get async client
            if self._async_client is None:
                self._async_client = AsyncHTTPClient()
                
            # Make async request with retries
            result_data = None
            last_error = None
            
            for attempt in range(self.config.max_retries):
                try:
                    async with await self._async_client.post_async(
                        self.base_url,
                        headers=self.headers,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                    ) as response:
                        if response.status == 429:
                            retry_after = int(response.headers.get("Retry-After", 60))
                            logger.warning(f"Rate limited, waiting {retry_after}s")
                            await asyncio.sleep(retry_after)
                            continue
                            
                        response.raise_for_status()
                        result_data = await response.json()
                        break
                        
                except Exception as e:
                    last_error = e
                    if attempt < self.config.max_retries - 1:
                        delay = self.config.retry_delay * (2 ** attempt)
                        logger.warning(f"Async request failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise
                        
            if result_data is None:
                raise Exception(f"Failed after {self.config.max_retries} attempts: {last_error}")
                
            # Extract results (same as sync version)
            latex = result_data.get("latex_styled", "")
            confidence = result_data.get("latex_confidence", 0.0)
            
            alternatives = {}
            if "mathml" in result_data:
                alternatives["mathml"] = result_data["mathml"]
            if "asciimath" in result_data:
                alternatives["asciimath"] = result_data["asciimath"]
                
            metadata = {
                "detected_alphabets": result_data.get("detected_alphabets", []),
                "is_printed": result_data.get("is_printed", True),
                "is_inverted": result_data.get("is_inverted", False)
            }
            
            return OCRResult(
                latex=latex,
                confidence=confidence,
                alternative_formats=alternatives,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Mathpix async OCR error: {str(e)[:200]}")
            return OCRResult(
                latex="",
                confidence=0.0,
                error=str(e)
            )
            
    async def close(self):
        """Clean up async resources."""
        if self._async_client:
            await self._async_client.close()


class TesseractOCR(BaseOCRBackend):
    """Tesseract OCR backend."""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self._available = self._check_tesseract()
        
    def _check_tesseract(self) -> bool:
        """Check if Tesseract is available."""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except:
            return False
            
    def is_available(self) -> bool:
        return self._available
        
    def recognize(self, image: Image.Image) -> OCRResult:
        """Recognize text using Tesseract."""
        if not self.is_available():
            return OCRResult(
                latex="",
                confidence=0.0,
                error="Tesseract not available"
            )
            
        try:
            import pytesseract
            
            # Configure Tesseract for math
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/=()[]{}.,<>^_\\'
            
            # Get text
            text = pytesseract.image_to_string(image, config=custom_config)
            
            # Get confidence scores
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Basic LaTeX conversion
            latex = self._text_to_basic_latex(text)
            
            return OCRResult(
                latex=latex,
                confidence=avg_confidence / 100.0,
                metadata={"source": "tesseract", "raw_text": text}
            )
            
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            return OCRResult(
                latex="",
                confidence=0.0,
                error=str(e)
            )
            
    async def recognize_async(self, image: Image.Image) -> OCRResult:
        """Recognize asynchronously (just wraps sync version)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.recognize, image)
        
    def _text_to_basic_latex(self, text: str) -> str:
        """Convert basic text to LaTeX."""
        latex = text.strip()
        
        # Basic replacements
        replacements = {
            'sqrt': '\\sqrt',
            'sum': '\\sum',
            'int': '\\int',
            'alpha': '\\alpha',
            'beta': '\\beta',
            'gamma': '\\gamma',
            'pi': '\\pi',
            '+-': '\\pm',
            '>=': '\\geq',
            '<=': '\\leq',
            '!=': '\\neq',
            'infinity': '\\infty'
        }
        
        for old, new in replacements.items():
            latex = latex.replace(old, new)
            
        # Handle fractions
        if '/' in latex and not '\\' in latex:
            parts = latex.split('/')
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                latex = f"\\frac{{{parts[0].strip()}}}{{{parts[1].strip()}}}"
                
        return latex


class MathOCR:
    """Main math OCR interface with multiple backends."""
    
    def __init__(self, config: Optional[OCRConfig] = None, 
                 http_client: Optional[HTTPClient] = None):
        self.config = config or OCRConfig()
        self.preprocessor = ImagePreprocessor(self.config.preprocess_config)
        self.mathpix = MathpixOCR(self.config, http_client)
        self.tesseract = TesseractOCR(self.config)
        self._cache = LRUCacheWithMemoryLimit(
            self.config.max_cache_entries,
            self.config.max_cache_memory_mb
        )
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
    def recognize_image(self, image: Union[Image.Image, np.ndarray, str, Path]) -> OCRResult:
        """Recognize math formula from image."""
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Check cache
        cache_key = self._get_image_hash(image)
        cached_result = self._cache.get(cache_key)
        if self.config.cache_results and cached_result:
            logger.debug("Returning cached OCR result")
            return cached_result
            
        # Preprocess image
        if self.config.enable_preprocessing:
            processed_image = self.preprocessor.preprocess(image)
        else:
            processed_image = image
            
        # Try primary OCR (Mathpix)
        result = self.mathpix.recognize(processed_image)
        
        # Try fallback if needed
        if not result.is_valid and self.config.enable_fallback:
            logger.info("Primary OCR failed, trying fallback")
            result = self.tesseract.recognize(processed_image)
            
        # Cache result
        if self.config.cache_results and result.is_valid:
            result_size = len(str(result).encode('utf-8'))
            self._cache.put(cache_key, result, result_size)
            
        return result
        
    async def recognize_image_async(self, image: Union[Image.Image, np.ndarray, str, Path]) -> OCRResult:
        """Recognize math formula from image asynchronously."""
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            loop = asyncio.get_event_loop()
            image = await loop.run_in_executor(self._executor, Image.open, str(image))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Check cache
        cache_key = self._get_image_hash(image)
        cached_result = self._cache.get(cache_key)
        if self.config.cache_results and cached_result:
            logger.debug("Returning cached OCR result")
            return cached_result
            
        # Preprocess image
        if self.config.enable_preprocessing:
            loop = asyncio.get_event_loop()
            processed_image = await loop.run_in_executor(
                self._executor,
                self.preprocessor.preprocess,
                image
            )
        else:
            processed_image = image
            
        # Try primary OCR (Mathpix)
        result = await self.mathpix.recognize_async(processed_image)
        
        # Try fallback if needed
        if not result.is_valid and self.config.enable_fallback:
            logger.info("Primary OCR failed, trying fallback")
            result = await self.tesseract.recognize_async(processed_image)
            
        # Cache result
        if self.config.cache_results and result.is_valid:
            result_size = len(str(result).encode('utf-8'))
            self._cache.put(cache_key, result, result_size)
            
        return result
        
    def recognize_regions(self, image: Union[Image.Image, np.ndarray]) -> List[Tuple[OCRResult, Tuple[int, int, int, int]]]:
        """Detect and recognize multiple math regions in image."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Extract math regions
        regions = self.preprocessor.extract_math_regions(image)
        
        results = []
        for region_image, bbox in regions:
            result = self.recognize_image(region_image)
            results.append((result, bbox))
            
        return results
        
    async def recognize_regions_async(self, image: Union[Image.Image, np.ndarray]) -> List[Tuple[OCRResult, Tuple[int, int, int, int]]]:
        """Detect and recognize multiple math regions asynchronously."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Extract math regions
        loop = asyncio.get_event_loop()
        regions = await loop.run_in_executor(
            self._executor,
            self.preprocessor.extract_math_regions,
            image
        )
        
        # Recognize all regions concurrently
        tasks = []
        for region_image, bbox in regions:
            task = self.recognize_image_async(region_image)
            tasks.append((task, bbox))
            
        results = []
        for task, bbox in tasks:
            result = await task
            results.append((result, bbox))
            
        return results
        
    def _get_image_hash(self, image: Image.Image) -> str:
        """Get hash of image for caching (fast version)."""
        # Use a faster hash on a sample of the image
        width, height = image.size
        
        # Sample image (first 64KB equivalent)
        sample_size = min(width * height, 64 * 1024)
        
        # Convert to bytes (sample only)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Get sample pixels
        pixels = list(image.getdata())[:sample_size]
        pixel_bytes = b''.join(bytes(p) for p in pixels)
        
        # Use blake2b for speed
        return hashlib.blake2b(pixel_bytes, digest_size=16).hexdigest()
        
    def clear_cache(self):
        """Clear the results cache."""
        self._cache.clear()
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'entries': len(self._cache.cache),
            'memory_usage_mb': self._cache.memory_usage / (1024 * 1024),
            'max_entries': self._cache.max_entries,
            'max_memory_mb': self._cache.max_memory_bytes / (1024 * 1024)
        }
        
    def close(self):
        """Clean up resources."""
        self._executor.shutdown(wait=True)
        if hasattr(self.mathpix, 'http_client'):
            if hasattr(self.mathpix.http_client, 'close'):
                self.mathpix.http_client.close()
                
    async def aclose(self):
        """Clean up async resources."""
        self._executor.shutdown(wait=True)
        await self.mathpix.close()


def recognize_math(image: Union[Image.Image, np.ndarray, str, Path],
                  config: Optional[OCRConfig] = None) -> OCRResult:
    """Convenience function to recognize math from image."""
    ocr = MathOCR(config)
    try:
        return ocr.recognize_image(image)
    finally:
        ocr.close()


async def recognize_math_async(image: Union[Image.Image, np.ndarray, str, Path],
                              config: Optional[OCRConfig] = None) -> OCRResult:
    """Convenience async function to recognize math from image."""
    ocr = MathOCR(config)
    try:
        return await ocr.recognize_image_async(image)
    finally:
        await ocr.aclose()


__all__ = [
    'MathOCR', 
    'OCRConfig', 
    'PreprocessConfig',
    'OCRResult', 
    'recognize_math',
    'recognize_math_async',
    'BaseOCRBackend',
    'MathpixOCR',
    'TesseractOCR'
]
