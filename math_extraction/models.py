"""
Data models for Math & STEM Content Processor
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from datetime import datetime


class MathType(Enum):
    """Types of mathematical expressions."""
    INLINE = "inline"
    DISPLAY = "display"
    EQUATION = "equation"
    ALIGN = "align"
    MATRIX = "matrix"
    CASES = "cases"
    GATHER = "gather"
    MULTILINE = "multiline"
    UNKNOWN = "unknown"


class DiagramType(Enum):
    """Types of diagrams."""
    FUNCTION_PLOT = "function_plot"
    PARAMETRIC_PLOT = "parametric_plot"
    CHEMICAL_STRUCTURE = "chemical_structure"
    GEOMETRY = "geometry"
    FLOWCHART = "flowchart"
    GRAPH = "graph"
    OTHER = "other"


@dataclass
class MathFormula:
    """Represents a mathematical formula."""
    latex: str
    math_type: MathType
    position: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)
    page_number: Optional[int] = None
    confidence: float = 1.0
    mathml: Optional[str] = None
    rendered_image: Optional[str] = None  # Base64 encoded image
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'latex': self.latex,
            'type': self.math_type.value,
            'position': self.position,
            'page_number': self.page_number,
            'confidence': self.confidence,
            'mathml': self.mathml,
            'rendered_image': self.rendered_image,
            'metadata': self.metadata
        }


@dataclass
class ChemicalFormula:
    """Represents a chemical formula or structure."""
    text: str
    smiles: Optional[str] = None
    inchi: Optional[str] = None
    position: Optional[Tuple[int, int, int, int]] = None
    page_number: Optional[int] = None
    rendered_image: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'smiles': self.smiles,
            'inchi': self.inchi,
            'position': self.position,
            'page_number': self.page_number,
            'rendered_image': self.rendered_image,
            'metadata': self.metadata
        }


@dataclass
class CodeBlock:
    """Represents a code block."""
    code: str
    language: Optional[str] = None
    position: Optional[Tuple[int, int, int, int]] = None
    page_number: Optional[int] = None
    highlighted_html: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'code': self.code,
            'language': self.language,
            'position': self.position,
            'page_number': self.page_number,
            'highlighted_html': self.highlighted_html,
            'metadata': self.metadata
        }


@dataclass
class Diagram:
    """Represents a diagram or plot."""
    diagram_type: DiagramType
    content: str  # Can be plot data, SVG, or other format
    position: Optional[Tuple[int, int, int, int]] = None
    page_number: Optional[int] = None
    rendered_image: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'type': self.diagram_type.value,
            'content': self.content,
            'position': self.position,
            'page_number': self.page_number,
            'rendered_image': self.rendered_image,
            'metadata': self.metadata
        }


@dataclass
class STEMContent:
    """Container for all STEM content extracted from a document."""
    formulas: List[MathFormula] = field(default_factory=list)
    chemical_formulas: List[ChemicalFormula] = field(default_factory=list)
    code_blocks: List[CodeBlock] = field(default_factory=list)
    diagrams: List[Diagram] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'formulas': [f.to_dict() for f in self.formulas],
            'chemical_formulas': [c.to_dict() for c in self.chemical_formulas],
            'code_blocks': [cb.to_dict() for cb in self.code_blocks],
            'diagrams': [d.to_dict() for d in self.diagrams],
            'metadata': self.metadata,
            'statistics': self.get_statistics()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the content."""
        return {
            'total_formulas': len(self.formulas),
            'total_chemical_formulas': len(self.chemical_formulas),
            'total_code_blocks': len(self.code_blocks),
            'total_diagrams': len(self.diagrams),
            'formula_types': dict(self._count_formula_types()),
            'diagram_types': dict(self._count_diagram_types()),
            'languages': list(self._get_code_languages())
        }
    
    def _count_formula_types(self) -> Dict[str, int]:
        """Count formulas by type."""
        from collections import Counter
        return Counter(f.math_type.value for f in self.formulas)
    
    def _count_diagram_types(self) -> Dict[str, int]:
        """Count diagrams by type."""
        from collections import Counter
        return Counter(d.diagram_type.value for d in self.diagrams)
    
    def _get_code_languages(self) -> set:
        """Get unique code languages."""
        return {cb.language for cb in self.code_blocks if cb.language}


@dataclass
class ProcessingResult:
    """Result of document processing."""
    content: STEMContent
    source_file: Optional[str] = None
    processing_time: Optional[float] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'content': self.content.to_dict(),
            'source_file': self.source_file,
            'processing_time': self.processing_time,
            'errors': self.errors,
            'warnings': self.warnings,
            'timestamp': self.timestamp.isoformat(),
            'success': len(self.errors) == 0
        }
    
    @property
    def is_successful(self) -> bool:
        """Check if processing was successful."""
        return len(self.errors) == 0
