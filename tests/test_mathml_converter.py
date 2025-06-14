import pytest
from math_extraction.mathml_converter import MathMLConverter, MathMLElement


class TestMathMLConverter:
    
    def test_initialization(self):
        """Test MathMLConverter initialization."""
        converter = MathMLConverter()
        assert converter.parser is not None
        assert converter.symbols is not None
        
    def test_convert_simple_expression(self, mathml_converter):
        """Test conversion of simple expressions."""
        latex = "x + y = z"
        mathml = mathml_converter.convert(latex)
        
        assert '<math' in mathml
        assert 'xmlns="http://www.w3.org/1998/Math/MathML"' in mathml
        assert '<mi>x</mi>' in mathml
        assert '<mo>+</mo>' in mathml
        assert '<mi>y</mi>' in mathml
        assert '<mo>=</mo>' in mathml
        assert '<mi>z</mi>' in mathml
        
    def test_convert_fraction(self, mathml_converter):
        """Test fraction conversion."""
        latex = r"\frac{a}{b}"
        mathml = mathml_converter.convert(latex)
        
        assert '<mfrac>' in mathml
        assert '</mfrac>' in mathml
        assert '<mi>a</mi>' in mathml
        assert '<mi>b</mi>' in mathml
        
    def test_convert_sqrt(self, mathml_converter):
        """Test square root conversion."""
        latex = r"\sqrt{x}"
        mathml = mathml_converter.convert(latex)
        
        assert '<msqrt>' in mathml
        assert '</msqrt>' in mathml
        assert '<mi>x</mi>' in mathml
        
        # Test with optional argument (nth root)
        latex2 = r"\sqrt[3]{x}"
        mathml2 = mathml_converter.convert(latex2)
        assert '<mroot>' in mathml2 or '<msqrt>' in mathml2
        
    def test_convert_greek_letters(self, mathml_converter):
        """Test Greek letter conversion."""
        latex = r"\alpha + \beta = \gamma"
        mathml = mathml_converter.convert(latex)
        
        assert 'α' in mathml or '&#x03B1;' in mathml  # alpha
        assert 'β' in mathml or '&#x03B2;' in mathml  # beta
        assert 'γ' in mathml or '&#x03B3;' in mathml  # gamma
        
    def test_convert_subscripts_superscripts(self, mathml_converter):
        """Test subscript/superscript conversion."""
        # Subscript
        latex1 = "x_1"
        mathml1 = mathml_converter.convert(latex1)
        assert '<msub>' in mathml1
        
        # Superscript
        latex2 = "x^2"
        mathml2 = mathml_converter.convert(latex2)
        assert '<msup>' in mathml2
        
        # Both
        latex3 = "x_1^2"
        mathml3 = mathml_converter.convert(latex3)
        assert '<msubsup>' in mathml3 or ('<msub>' in mathml3 and '<msup>' in mathml3)
        
    def test_convert_matrix(self, mathml_converter):
        """Test matrix conversion."""
        latex = r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}"
        mathml = mathml_converter.convert(latex)
        
        assert '<mtable>' in mathml
        assert '<mtr>' in mathml
        assert '<mtd>' in mathml
        assert mathml.count('<mtr>') == 2  # Two rows
        assert mathml.count('<mtd>') == 4  # Four cells
        
    def test_convert_operators(self, mathml_converter):
        """Test operator conversion."""
        latex = r"a \times b \div c \pm d"
        mathml = mathml_converter.convert(latex)
        
        assert '×' in mathml or '&times;' in mathml
        assert '÷' in mathml or '&divide;' in mathml
        assert '±' in mathml or '&plusmn;' in mathml
        
    def test_convert_integrals_sums(self, mathml_converter):
        """Test integral and sum conversion."""
        # Integral
        latex1 = r"\int_0^1 x \, dx"
        mathml1 = mathml_converter.convert(latex1)
        assert '∫' in mathml1 or '&int;' in mathml1
        
        # Sum
        latex2 = r"\sum_{i=1}^n i"
        mathml2 = mathml_converter.convert(latex2)
        assert '∑' in mathml2 or '&sum;' in mathml2
        
    def test_convert_text(self, mathml_converter):
        """Test text conversion."""
        latex = r"\text{Hello} + x"
        mathml = mathml_converter.convert(latex)
        
        assert '<mtext>Hello</mtext>' in mathml
        
    def test_convert_accents(self, mathml_converter):
        """Test accent conversion."""
        latex = r"\hat{x} + \tilde{y}"
        mathml = mathml_converter.convert(latex)
        
        assert '<mover>' in mathml
        assert 'accent="true"' in mathml
        
    def test_display_mode(self, mathml_converter):
        """Test display mode setting."""
        latex = "x + y"
        
        # Inline mode
        mathml_inline = mathml_converter.convert(latex, display_mode=False)
        assert 'display="inline"' in mathml_inline
        
        # Display mode
        mathml_display = mathml_converter.convert(latex, display_mode=True)
        assert 'display="block"' in mathml_display
        
    def test_validation(self, mathml_converter):
        """Test MathML validation."""
        # Valid MathML
        valid_mathml = '''
        <math xmlns="http://www.w3.org/1998/Math/MathML">
            <mi>x</mi>
        </math>
        '''
        is_valid, errors = mathml_converter.validate_mathml(valid_mathml)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid MathML - wrong root element
        invalid_mathml = '<notmath><mi>x</mi></notmath>'
        is_valid, errors = mathml_converter.validate_mathml(invalid_mathml)
        assert not is_valid
        assert len(errors) > 0
        
    def test_optimization(self, mathml_converter):
        """Test MathML optimization."""
        # Create MathML with redundant mrow
        mathml = '''
        <math xmlns="http://www.w3.org/1998/Math/MathML">
            <mrow>
                <mrow>
                    <mi>x</mi>
                </mrow>
            </mrow>
        </math>
        '''
        
        optimized = mathml_converter.optimize_mathml(mathml)
        
        # Should remove redundant mrow elements
        assert optimized.count('<mrow>') < mathml.count('<mrow>')
        
    def test_cache_functionality(self, mathml_converter):
        """Test caching functionality."""
        latex = r"\frac{a}{b}"
        
        # Clear cache first
        mathml_converter.clear_cache()
        
        # First conversion
        mathml1 = mathml_converter.convert(latex)
        
        # Second conversion (should use cache)
        mathml2 = mathml_converter.convert(latex)
        
        assert mathml1 == mathml2
        
        # Check cache stats
        stats = mathml_converter.get_cache_stats()
        assert stats['cache_size'] > 0
        
    def test_complex_expressions(self, mathml_converter, sample_latex_formulas):
        """Test conversion of complex expressions."""
        # Test complex integral
        latex = sample_latex_formulas['complex']
        mathml = mathml_converter.convert(latex)
        
        assert '<math' in mathml
        assert '∫' in mathml or '&int;' in mathml
        assert '<mfrac>' in mathml
        assert '<msqrt>' in mathml
        
    def test_fallback_mathml(self, mathml_converter):
        """Test fallback MathML generation."""
        # Force an error by passing invalid input
        result = mathml_converter._fallback_mathml("test", False)
        
        assert '<math' in result
        assert '<merror>' in result
        assert '<mtext>' in result


class TestEdgeCases:
    
    def test_empty_input(self, mathml_converter):
        """Test with empty input."""
        mathml = mathml_converter.convert("")
        assert '<math' in mathml
        
    def test_very_long_input(self, mathml_converter):
        """Test with very long input."""
        # Create long expression
        terms = [f"x_{i}" for i in range(100)]
        latex = " + ".join(terms)
        
        # Should not raise exception
        mathml = mathml_converter.convert(latex)
        assert '<math' in mathml
        
    def test_invalid_latex(self, mathml_converter):
        """Test with invalid LaTeX."""
        # Unmatched braces
        latex = r"\frac{a}{b"
        
        # Should still produce some output
        mathml = mathml_converter.convert(latex)
        assert '<math' in mathml
        
    def test_nested_environments(self, mathml_converter):
        """Test nested environments."""
        latex = r"""
        \begin{align}
        \begin{pmatrix}
        a & b \\
        c & d
        \end{pmatrix}
        x = 
        \begin{pmatrix}
        e \\
        f
        \end{pmatrix}
        \end{align}
        """
        
        mathml = mathml_converter.convert(latex)
        assert '<mtable>' in mathml
        
    def test_special_characters(self, mathml_converter):
        """Test special characters in LaTeX."""
        latex = r"x & y"  # Ampersand outside of environment
        
        # Should handle gracefully
        mathml = mathml_converter.convert(latex)
        assert '<math' in mathml
