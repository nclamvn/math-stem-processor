"""
Base classes for rendering system
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image


@dataclass
class RenderConfig:
    """Configuration for rendering."""
    # Output settings
    dpi: int = 300
    font_size: int = 12
    output_format: str = "png"  # png, svg, pdf
    
    # Appearance
    background_color: str = "white"
    text_color: str = "black"
    font_family: str = "Computer Modern"
    
    # LaTeX settings
    latex_preamble: str = ""
    use_unicode: bool = True
    packages: list = field(default_factory=list)
    
    # Performance
    cache_renders: bool = True
    timeout: int = 30
    
    # Paths
    temp_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    

@dataclass
class RenderResult:
    """Result of rendering operation."""
    image: Optional[Image.Image] = None
    svg: Optional[str] = None
    pdf_bytes: Optional[bytes] = None
    size: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def is_valid(self) -> bool:
        """Check if render was successful."""
        return bool(self.image or self.svg or self.pdf_bytes) and not self.error
        
    def save(self, path: Union[str, Path]):
        """Save render result to file."""
        path = Path(path)
        
        if self.image:
            self.image.save(path)
        elif self.svg:
            path.write_text(self.svg)
        elif self.pdf_bytes:
            path.write_bytes(self.pdf_bytes)
        else:
            raise ValueError("No render result to save")


class BaseRenderer(ABC):
    """Base class for all renderers."""
    
    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config or RenderConfig()
        self._cache = {} if self.config.cache_renders else None
        
    @abstractmethod
    def is_available(self) -> bool:
        """Check if renderer is available."""
        pass
        
    @abstractmethod
    def render_latex(self, latex: str, config: Optional[RenderConfig] = None) -> RenderResult:
        """Render LaTeX to image/svg/pdf."""
        pass
        
    def get_cache_key(self, latex: str, config: RenderConfig) -> str:
        """Generate cache key."""
        import hashlib
        key_str = f"{latex}_{config.dpi}_{config.output_format}_{config.font_size}"
        return hashlib.md5(key_str.encode()).hexdigest()
        
    def clear_cache(self):
        """Clear render cache."""
        if self._cache:
            self._cache.clear()
