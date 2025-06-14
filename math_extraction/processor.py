"""
Main processor for STEM content extraction
"""

import logging
import time
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass, field

from .models import STEMContent, ProcessingResult, MathFormula
from .config import ProcessConfig
from .formula_detector import FormulaDetector
from .latex_parser import LaTeXParser
from .mathml_converter import MathMLConverter


logger = logging.getLogger(__name__)


class STEMContentProcessor:
    """Main processor for extracting STEM content from documents."""
    
    def __init__(self, config: Optional[ProcessConfig] = None):
        """Initialize processor with configuration."""
        self.config = config or ProcessConfig()
        
        # Initialize components
        self.formula_detector = FormulaDetector()
        self.latex_parser = LaTeXParser()
        self.mathml_converter = MathMLConverter()
        
        # Setup logging
        if self.config.verbose:
            logging.basicConfig(level=logging.INFO)
        if self.config.debug:
            logging.basicConfig(level=logging.DEBUG)
    
    def process_text(self, text: str, source_file: Optional[str] = None) -> ProcessingResult:
        """Process text content to extract STEM elements."""
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # Create content container
            content = STEMContent()
            
            # Extract formulas
            if self.config.extract_formulas:
                try:
                    formulas = self.formula_detector.detect_formulas(text, page_number=1)
                    
                    # Process each formula
                    for formula in formulas:
                        # Parse LaTeX
                        structure = self.latex_parser.parse(formula.latex)
                        if not structure.is_valid:
                            warnings.append({
                                'type': 'parse_warning',
                                'formula': formula.latex,
                                'errors': structure.errors
                            })
                        
                        # Convert to MathML if requested
                        if self.config.convert_to_mathml:
                            try:
                                formula.mathml = self.mathml_converter.convert(formula.latex)
                            except Exception as e:
                                warnings.append({
                                    'type': 'mathml_conversion',
                                    'formula': formula.latex,
                                    'error': str(e)
                                })
                        
                        content.formulas.append(formula)
                        
                except Exception as e:
                    errors.append({
                        'type': 'formula_extraction',
                        'error': str(e)
                    })
                    logger.error(f"Formula extraction failed: {e}")
            
            # TODO: Extract other content types (code, chemicals, diagrams)
            # if self.config.extract_code:
            #     content.code_blocks = self.extract_code_blocks(text)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                content=content,
                source_file=source_file,
                processing_time=processing_time,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return ProcessingResult(
                content=STEMContent(),
                source_file=source_file,
                processing_time=time.time() - start_time,
                errors=[{'type': 'general', 'error': str(e)}]
            )
    
    def process_file(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Process a file to extract STEM content."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return ProcessingResult(
                content=STEMContent(),
                source_file=str(file_path),
                errors=[{'type': 'file_not_found', 'error': f"File not found: {file_path}"}]
            )
        
        # Determine file type and process accordingly
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt':
            text = file_path.read_text(encoding='utf-8')
            return self.process_text(text, str(file_path))
        elif suffix == '.tex':
            text = file_path.read_text(encoding='utf-8')
            return self.process_text(text, str(file_path))
        elif suffix == '.pdf':
            # TODO: Implement PDF processing
            return ProcessingResult(
                content=STEMContent(),
                source_file=str(file_path),
                errors=[{'type': 'unsupported_format', 'error': 'PDF processing not yet implemented'}]
            )
        else:
            return ProcessingResult(
                content=STEMContent(),
                source_file=str(file_path),
                errors=[{'type': 'unsupported_format', 'error': f"Unsupported file format: {suffix}"}]
            )
    
    def process_batch(self, files: List[Union[str, Path]], 
                     max_workers: Optional[int] = None) -> List[ProcessingResult]:
        """Process multiple files in batch."""
        max_workers = max_workers or self.config.max_workers
        results = []
        
        # Simple sequential processing for now
        # TODO: Add parallel processing with ThreadPoolExecutor
        for file_path in files:
            result = self.process_file(file_path)
            results.append(result)
            
        return results
    
    def export_results(self, result: ProcessingResult, 
                      output_path: Union[str, Path], 
                      format: str = "json") -> bool:
        """Export processing results to file."""
        output_path = Path(output_path)
        
        try:
            if format == "json":
                import json
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            elif format == "latex":
                # Export as LaTeX document
                latex_content = self._generate_latex_output(result)
                output_path.write_text(latex_content, encoding='utf-8')
            elif format == "markdown":
                # Export as Markdown
                md_content = self._generate_markdown_output(result)
                output_path.write_text(md_content, encoding='utf-8')
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def _generate_latex_output(self, result: ProcessingResult) -> str:
        """Generate LaTeX document from results."""
        lines = [
            "\\documentclass{article}",
            "\\usepackage{amsmath}",
            "\\usepackage{amssymb}",
            "\\begin{document}",
            "",
            "\\section{Extracted Mathematical Content}",
            ""
        ]
        
        for i, formula in enumerate(result.content.formulas, 1):
            lines.append(f"\\subsection{{Formula {i}}}")
            lines.append(f"Type: {formula.math_type.value}")
            lines.append("")
            
            if formula.math_type.value == "inline":
                lines.append(f"${formula.latex}$")
            else:
                lines.append(f"$${formula.latex}$$")
            lines.append("")
        
        lines.append("\\end{document}")
        return "\n".join(lines)
    
    def _generate_markdown_output(self, result: ProcessingResult) -> str:
        """Generate Markdown document from results."""
        lines = [
            "# Extracted STEM Content",
            "",
            f"Source: {result.source_file or 'Unknown'}",
            f"Processing time: {result.processing_time:.2f}s",
            "",
            "## Mathematical Formulas",
            ""
        ]
        
        for i, formula in enumerate(result.content.formulas, 1):
            lines.append(f"### Formula {i}")
            lines.append(f"- Type: {formula.math_type.value}")
            lines.append(f"- Confidence: {formula.confidence:.2f}")
            lines.append("")
            
            if formula.math_type.value == "inline":
                lines.append(f"${formula.latex}$")
            else:
                lines.append(f"$${formula.latex}$$")
            lines.append("")
        
        # Add statistics
        stats = result.content.get_statistics()
        lines.extend([
            "## Statistics",
            "",
            f"- Total formulas: {stats['total_formulas']}",
            f"- Formula types: {stats['formula_types']}",
            ""
        ])
        
        return "\n".join(lines)
