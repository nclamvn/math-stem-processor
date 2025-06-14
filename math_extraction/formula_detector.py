"""
Formula detector for mathematical expressions
"""

import re
import logging
from typing import List, Tuple, Optional, Dict, Set, Any, Union
from dataclasses import dataclass, field
from functools import lru_cache
from collections import defaultdict
import hashlib
from .models import MathFormula, MathType


logger = logging.getLogger(__name__)


@dataclass
class DocumentLayout:
    """Document layout information."""
    columns: int = 1
    column_width: Optional[float] = None
    gutter_width: Optional[float] = None
    margin_left: float = 0
    margin_right: float = 0
    page_width: float = 595  # A4 width in points
    page_height: float = 842  # A4 height in points


class FormulaDetector:
    """Detects mathematical formulas in text."""
    
    def __init__(self, layout: Optional[DocumentLayout] = None):
        """Initialize formula detector."""
        self.layout = layout or DocumentLayout()
        self.confidence_threshold = 0.8
        self.line_height = 20.0
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Compile regex patterns for formula detection."""
        # Display patterns PHẢI được check TRƯỚC inline patterns
        self.patterns = {
            "display": [
                (re.compile(r"\$\$(.+?)\$\$", re.DOTALL), 1),  # $$...$$ 
                (re.compile(r"\\\[(.+?)\\\]", re.DOTALL), 1),  # \[...\]
            ],
            "equation": [
                (re.compile(r"\\begin\{equation\}(.+?)\\end\{equation\}", re.DOTALL), 1),
                (re.compile(r"\\begin\{equation\*\}(.+?)\\end\{equation\*\}", re.DOTALL), 1),
            ],
            "align": [
                (re.compile(r"\\begin\{align\}(.+?)\\end\{align\}", re.DOTALL), 1),
                (re.compile(r"\\begin\{align\*\}(.+?)\\end\{align\*\}", re.DOTALL), 1),
            ],
            "matrix": [
                (re.compile(r"\\begin\{[pvbBV]?matrix\}(.+?)\\end\{[pvbBV]?matrix\}", re.DOTALL), 1),
            ],
            "inline": [
                (re.compile(r"\$([^\$\n]+)\$"), 1),  # $...$ - single line only
                (re.compile(r"\\\((.+?)\\\)", re.DOTALL), 1),   # \(...\)
            ],
        }
        self.compiled_patterns = []
        # Thứ tự quan trọng: display trước inline
        for category in ["display", "equation", "align", "matrix", "inline"]:
            if category in self.patterns:
                for pattern, group in self.patterns[category]:
                    self.compiled_patterns.append((pattern, category, group))
    def detect_formulas(self, text: str, page_number: int = 1,
                       page_bounds: Optional[Tuple[float, float, float, float]] = None,
                       line_info: Optional[List[Dict[str, float]]] = None) -> List[MathFormula]:
        """Detect mathematical formulas in text."""
        formulas = []
        
        # Track positions to avoid duplicates
        used_positions = set()
        
        # Process each pattern
        for pattern, category, group_idx in self.compiled_patterns:
            for match in pattern.finditer(text):
                pos = (match.start(), match.end())
                
                if pos not in used_positions:
                    used_positions.add(pos)
                    
                    # Extract LaTeX
                    latex = match.group(group_idx)
                    
                    # Determine math type
                    if category == 'inline':
                        math_type = MathType.INLINE
                    elif category == 'display':
                        math_type = MathType.DISPLAY
                    elif category == 'equation':
                        math_type = MathType.EQUATION
                    elif category == 'align':
                        math_type = MathType.ALIGN
                    elif category == 'matrix':
                        math_type = MathType.MATRIX
                    else:
                        math_type = MathType.UNKNOWN
                    
                    formula = MathFormula(
                        latex=latex.strip(),
                        math_type=math_type,
                        page_number=page_number,
                        confidence=self._calculate_confidence(latex),
                        position=self._calculate_position(match, line_info, page_bounds)
                    )
                    formulas.append(formula)
        
        # Sort by position
        formulas.sort(key=lambda f: f.position[1] if f.position else 0)
        
        return formulas
    
    def _calculate_confidence(self, latex: str) -> float:
        """Calculate confidence score for detected formula."""
        confidence = 0.5
        
        # Check for common math symbols
        math_symbols = ['+', '-', '=', '<', '>', '\\', '{', '}', '^', '_', 
                       'frac', 'sqrt', 'sum', 'int', 'prod', 'alpha', 'beta']
        
        symbol_count = sum(1 for symbol in math_symbols if symbol in latex)
        confidence += min(0.4, symbol_count * 0.05)
        
        # Check length
        if 2 <= len(latex) <= 1000:
            confidence += 0.1
        
        # Clamp to [0, 1]
        return min(1.0, max(0.0, confidence))
    
    def _calculate_position(self, match: re.Match, 
                           line_info: Optional[List[Dict[str, float]]] = None,
                           page_bounds: Optional[Tuple[float, float, float, float]] = None) -> Tuple[int, int, int, int]:
        """Calculate position of formula on page."""
        # Simple default position
        return (0, match.start() * 10, 100, 20)
    
    def validate_extraction(self, formulas: List[MathFormula]) -> Dict[str, Any]:
        """Validate extraction results."""
        stats = {
            'total': len(formulas),
            'by_type': defaultdict(int),
            'confidence_levels': {'high': 0, 'medium': 0, 'low': 0},
            'potential_errors': [],
            'quality_metrics': {
                'has_math_symbols': 0,
                'proper_length': 0,
                'balanced_delimiters': 0
            }
        }
        
        for formula in formulas:
            # Count by type
            stats['by_type'][formula.math_type.value] += 1
            
            # Confidence levels
            if formula.confidence >= 0.8:
                stats['confidence_levels']['high'] += 1
            elif formula.confidence >= 0.5:
                stats['confidence_levels']['medium'] += 1
            else:
                stats['confidence_levels']['low'] += 1
                
            # Quality checks
            if any(sym in formula.latex for sym in ['+', '-', '=', '\\', '^', '_']):
                stats['quality_metrics']['has_math_symbols'] += 1
                
            if 2 <= len(formula.latex) <= 1000:
                stats['quality_metrics']['proper_length'] += 1
        
        return dict(stats)
    
    def export_statistics(self, formulas: List[MathFormula]) -> str:
        """Export statistics as formatted string."""
        stats = self.validate_extraction(formulas)
        
        lines = [
            f"Total formulas detected: {stats['total']}",
            "\nBy type:"
        ]
        
        for math_type, count in stats['by_type'].items():
            lines.append(f"  - {math_type}: {count}")
            
        lines.extend([
            "\nConfidence levels:",
            f"  - High (≥0.8): {stats['confidence_levels']['high']}",
            f"  - Medium (0.5-0.8): {stats['confidence_levels']['medium']}",
            f"  - Low (<0.5): {stats['confidence_levels']['low']}"
        ])
        
        return '\n'.join(lines)

    def detect_complex_environments(self, text: str) -> Dict[str, int]:
        """Detect complex LaTeX environments."""
        envs = defaultdict(int)
        # Simple pattern for environments
        env_pattern = re.compile(r'\\begin\{(\w+\*?)\}')
        for match in env_pattern.finditer(text):
            envs[match.group(1)] += 1
        return dict(envs)
