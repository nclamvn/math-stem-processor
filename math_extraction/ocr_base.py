"""
Base classes for OCR functionality
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import requests
from PIL import Image


@dataclass
class OCRResult:
    """Result from OCR processing."""
    latex: str = ""
    confidence: float = 0.0
    text: Optional[str] = None
    mathml: Optional[str] = None
    asciimath: Optional[str] = None
    position: Optional[tuple] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    alternative_formats: Dict[str, str] = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        """Check if result is valid."""
        return bool(self.latex) and self.confidence > 0 and not self.error


class BaseOCRBackend(ABC):
    """Base class for OCR backends."""
    
    def __init__(self, config: Any):
        self.config = config
        
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        pass
        
    @abstractmethod
    def recognize(self, image: Image.Image, **kwargs) -> OCRResult:
        """Recognize math from image."""
        pass
        
    def preprocess(self, image: Image.Image) -> Image.Image:
        """Preprocess image before OCR."""
        return image


class HTTPClient:
    """Simple HTTP client wrapper."""
    
    def __init__(self, timeout: int = 30):
        self.session = requests.Session()
        self.timeout = timeout
        
    def post(self, url: str, **kwargs) -> requests.Response:
        """Make POST request."""
        kwargs.setdefault('timeout', self.timeout)
        return self.session.post(url, **kwargs)
        
    def get(self, url: str, **kwargs) -> requests.Response:
        """Make GET request."""
        kwargs.setdefault('timeout', self.timeout)
        return self.session.get(url, **kwargs)
        
    def close(self):
        """Close session."""
        self.session.close()
