import pytest
from math_extraction.formula_detector import FormulaDetector, DocumentLayout
from math_extraction.models import MathType


class TestFormulaDetector:
    
    def test_initialization(self):
        """Test FormulaDetector initialization."""
        detector = FormulaDetector()
        assert detector.line_height == 20.0
        assert detector.confidence_threshold == 0.8
        assert len(detector.compiled_patterns) > 0
        
    def test_detect_inline_formulas(self, sample_latex_formulas):
        """Test detection of inline formulas."""
        detector = FormulaDetector()
        
        text = f"This is {sample_latex_formulas['simple_inline']} and more text."
        formulas = detector.detect_formulas(text, page_number=1)
        
        assert len(formulas) == 1
        assert formulas[0].latex == "x + y = z"
        assert formulas[0].math_type == MathType.INLINE
        assert formulas[0].confidence >= 0.8
        
    def test_detect_display_formulas(self, sample_latex_formulas):
        """Test detection of display formulas."""
        detector = FormulaDetector()
        
        text = f"Text before {sample_latex_formulas['simple_display']} text after."
        formulas = detector.detect_formulas(text, page_number=1)
        
        assert len(formulas) == 1
        assert formulas[0].latex == "E = mc^2"
        assert formulas[0].math_type == MathType.DISPLAY
        
    def test_detect_environments(self):
        """Test detection of LaTeX environments."""
        detector = FormulaDetector()
        
        text = r"""
        \begin{equation}
        x^2 + y^2 = z^2
        \end{equation}
        
        \begin{align}
        a &= b + c \\
        d &= e + f
        \end{align}
        """
        
        formulas = detector.detect_formulas(text, page_number=1)
        assert len(formulas) == 2
        
        # Check equation
        eq_formula = next(f for f in formulas if f.math_type == MathType.EQUATION)
        assert "x^2 + y^2 = z^2" in eq_formula.latex
        
        # Check align
        align_formula = next(f for f in formulas if f.math_type == MathType.ALIGN)
        assert "a &= b + c" in align_formula.latex
        
    def test_escaped_dollars(self):
        """Test handling of escaped dollar signs."""
        detector = FormulaDetector()
        
        text = r"Price is \$100 and formula is $x + y$"
        formulas = detector.detect_formulas(text, page_number=1)
        
        assert len(formulas) == 1
        assert formulas[0].latex == "x + y"
        
    def test_nested_environments(self):
        """Test detection of nested environments."""
        detector = FormulaDetector()
        
        text = r"""
        \begin{align}
        \begin{pmatrix}
        a & b \\
        c & d
        \end{pmatrix}
        \begin{pmatrix}
        x \\
        y
        \end{pmatrix}
        =
        \begin{pmatrix}
        e \\
        f
        \end{pmatrix}
        \end{align}
        """
        
        formulas = detector.detect_formulas(text, page_number=1)
        assert len(formulas) >= 1
        
        # The align environment should be detected
        align_found = any(f.math_type == MathType.ALIGN for f in formulas)
        assert align_found
        
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        detector = FormulaDetector()
        
        # High confidence - proper math formula
        text1 = r"$\int_0^1 x^2 \, dx = \frac{1}{3}$"
        formulas1 = detector.detect_formulas(text1, page_number=1)
        assert formulas1[0].confidence > 0.9
        
        # Low confidence - just text
        text2 = "$hello world$"
        formulas2 = detector.detect_formulas(text2, page_number=1)
        assert formulas2[0].confidence < 0.5
        
    def test_position_calculation(self):
        """Test position calculation with line info."""
        detector = FormulaDetector()
        
        line_info = [
            {'x0': 50, 'y0': 100, 'width': 500, 'height': 20},
            {'x0': 50, 'y0': 130, 'width': 500, 'height': 20},
        ]
        
        text = "First line\n$x + y$ on second line"
        formulas = detector.detect_formulas(
            text, 
            page_number=1,
            line_info=line_info
        )
        
        assert len(formulas) == 1
        # Formula is on second line
        assert formulas[0].position[1] == 130
        
    def test_two_column_layout(self):
        """Test formula detection in two-column layout."""
        layout = DocumentLayout(columns=2, column_width=250, gutter_width=20)
        detector = FormulaDetector(layout=layout)
        
        text = "Left column $a + b$ and right column $c + d$"
        formulas = detector.detect_formulas(
            text,
            page_number=1,
            page_bounds=(0, 0, 520, 800)
        )
        
        assert len(formulas) == 2
        # First formula should be in left column
        assert formulas[0].position[0] < 250
        # Second formula should be in right column  
        assert formulas[1].position[0] > 270
        
    def test_merge_overlapping(self):
        """Test merging of overlapping formulas."""
        detector = FormulaDetector()
        
        # Create text with potential duplicate detection
        text = "$x + y$ $x + y$"  # Same formula twice
        formulas = detector.detect_formulas(text, page_number=1)
        
        # Should detect both if they're at different positions
        assert len(formulas) == 2
        
    def test_complex_environments_detection(self):
        """Test detection of complex environments."""
        detector = FormulaDetector()
        
        text = r"""
        \begin{equation}
        test
        \end{equation}
        
        \begin{custom}
        content
        \end{custom}
        """
        
        envs = detector.detect_complex_environments(text)
        assert 'equation' in envs
        assert 'custom' in envs
        assert envs['equation'] == 1
        assert envs['custom'] == 1
        
    def test_validation_statistics(self):
        """Test extraction validation statistics."""
        detector = FormulaDetector()
        
        text = r"""
        $x + y$
        $$E = mc^2$$
        \begin{align}
        a &= b
        \end{align}
        $invalid{$
        """
        
        formulas = detector.detect_formulas(text, page_number=1)
        stats = detector.validate_extraction(formulas)
        
        assert stats['total'] >= 3
        assert 'inline' in stats['by_type']
        assert 'display' in stats['by_type']
        assert stats['confidence_levels']['high'] > 0
        assert len(stats['potential_errors']) >= 0
        
    def test_export_statistics(self):
        """Test statistics export."""
        detector = FormulaDetector()
        
        text = "$x + y$ and $$z = w$$"
        formulas = detector.detect_formulas(text, page_number=1)
        
        report = detector.export_statistics(formulas)
        assert isinstance(report, str)
        assert "Total formulas detected: 2" in report
        assert "inline" in report
        assert "display" in report


class TestEdgeCases:
    
    def test_empty_text(self):
        """Test with empty text."""
        detector = FormulaDetector()
        formulas = detector.detect_formulas("", page_number=1)
        assert len(formulas) == 0
        
    def test_no_formulas(self):
        """Test with text containing no formulas."""
        detector = FormulaDetector()
        text = "This is just plain text without any math."
        formulas = detector.detect_formulas(text, page_number=1)
        assert len(formulas) == 0
        
    def test_malformed_latex(self):
        """Test with malformed LaTeX."""
        detector = FormulaDetector()
        
        # Unmatched braces
        text = r"$x + \frac{a}{b$"
        formulas = detector.detect_formulas(text, page_number=1)
        
        if formulas:
            # Should have low confidence
            assert formulas[0].confidence < 0.8
            
    def test_very_long_formula(self):
        """Test with very long formula."""
        detector = FormulaDetector()
        
        # Create a long formula
        long_formula = "$" + " + ".join([f"x_{i}" for i in range(100)]) + "$"
        formulas = detector.detect_formulas(long_formula, page_number=1)
        
        assert len(formulas) == 1
        assert len(formulas[0].latex) > 200
        
    def test_unicode_in_text(self):
        """Test with Unicode characters."""
        detector = FormulaDetector()
        
        text = "Greek α and formula $\\alpha + \\beta$"
        formulas = detector.detect_formulas(text, page_number=1)
        
        assert len(formulas) == 1
        assert formulas[0].latex == "\\alpha + \\beta"
