"""
Math & STEM Content Processor

A comprehensive library for extracting, parsing, and processing mathematical
and STEM content from various document formats.
"""

__version__ = "0.1.0"
__author__ = "Math STEM Processor Team"

# Import các models
from .models import (
    MathFormula,
    MathType,
    ChemicalFormula,
    CodeBlock,
    Diagram,
    DiagramType,
    STEMContent,
    ProcessingResult
)

# Import các configuration classes
from .config import (
    ProcessConfig,
    OCRConfig,
    RenderConfig,
    PreprocessConfig
)

# Import core processors
from .formula_detector import FormulaDetector, DocumentLayout
from .latex_parser import LaTeXParser, LaTeXStructure
from .mathml_converter import MathMLConverter
from .math_ocr import MathOCR, OCRResult
from .rendering import MathRenderer, RenderResult

# Import main processor từ file processor.py
from .processor import STEMContentProcessor

# Export tất cả public APIs
__all__ = [
    # Version
    "__version__",
    
    # Models
    "MathFormula",
    "MathType", 
    "ChemicalFormula",
    "CodeBlock",
    "Diagram",
    "DiagramType",
    "STEMContent",
    "ProcessingResult",
    
    # Configurations
    "ProcessConfig",
    "OCRConfig",
    "RenderConfig",
    "PreprocessConfig",
    
    # Core components
    "FormulaDetector",
    "DocumentLayout",
    "LaTeXParser",
    "LaTeXStructure",
    "MathMLConverter",
    "MathOCR",
    "OCRResult",
    "MathRenderer",
    "RenderResult",
    
    # Main processor
    "STEMContentProcessor"
]

# Convenience function
def create_processor(**kwargs):
    """Create a configured STEM content processor instance."""
    config = ProcessConfig(**kwargs)
    return STEMContentProcessor(config)
