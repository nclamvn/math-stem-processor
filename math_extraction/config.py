"""
Configuration classes for Math & STEM Content Processor
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class ProcessConfig:
    """Main processing configuration."""
    # General settings
    verbose: bool = False
    debug: bool = False
    
    # Processing options
    extract_formulas: bool = True
    extract_code: bool = True
    extract_chemicals: bool = True
    extract_diagrams: bool = True
    
    # Formula detection
    formula_confidence_threshold: float = 0.8
    merge_adjacent_formulas: bool = True
    
    # Output options
    render_formulas: bool = False
    convert_to_mathml: bool = True
    
    # Performance
    max_workers: int = 4
    batch_size: int = 50
    
    # Paths
    temp_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Initialize paths."""
        if self.temp_dir:
            self.temp_dir = Path(self.temp_dir)
        if self.cache_dir:
            self.cache_dir = Path(self.cache_dir)


@dataclass
class OCRConfig:
    """OCR configuration."""
    # API credentials
    mathpix_api_id: Optional[str] = None
    mathpix_api_key: Optional[str] = None
    
    # OCR settings
    enable_preprocessing: bool = True
    enable_fallback: bool = True
    confidence_threshold: float = 0.7
    
    # Performance
    max_retries: int = 3
    timeout: int = 30
    cache_results: bool = True
    
    # Image settings
    max_image_size: int = 4096
    max_image_bytes: int = 1024 * 1024 * 4  # 4MB


@dataclass
class RenderConfig:
    """Rendering configuration."""
    # Output settings
    dpi: int = 300
    font_size: int = 12
    output_format: str = "png"  # png, svg, pdf
    
    # Appearance
    background_color: str = "white"
    text_color: str = "black"
    
    # LaTeX settings
    latex_preamble: str = ""
    use_unicode: bool = True
    
    # Performance
    cache_renders: bool = True
    parallel_rendering: bool = True
    
    # Paths
    latex_path: Optional[str] = None
    dvipng_path: Optional[str] = None


@dataclass
class PreprocessConfig:
    """Image preprocessing configuration."""
    # Enhancement
    enhance_contrast: bool = True
    denoise: bool = True
    deskew: bool = True
    
    # Thresholds
    binary_threshold: int = 128
    noise_size: int = 3
    
    # Size constraints
    min_size: int = 50
    max_size: int = 4096
    
    # Region detection
    detect_regions: bool = True
    min_region_size: int = 100
    merge_threshold: float = 0.1
