import pytest
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
from math_extraction.models import MathFormula, MathType, STEMContent
from math_extraction.latex_parser import LaTeXParser
from math_extraction.mathml_converter import MathMLConverter
from math_extraction.math_ocr import OCRConfig, MathOCR


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_latex_formulas():
    """Sample LaTeX formulas for testing."""
    return {
        'simple_inline': '$x + y = z$',
        'simple_display': '$$E = mc^2$$',
        'fraction': r'\frac{a}{b}',
        'sqrt': r'\sqrt{x^2 + y^2}',
        'matrix': r'\begin{pmatrix} a & b \\ c & d \end{pmatrix}',
        'align': r'\begin{align} x &= 1 \\ y &= 2 \end{align}',
        'cases': r'\begin{cases} x & \text{if } x > 0 \\ -x & \text{otherwise} \end{cases}',
        'complex': r'\int_0^{\infty} e^{-x^2} \, dx = \frac{\sqrt{\pi}}{2}',
        'nested': r'x^{y^z}',
        'greek': r'\alpha + \beta = \gamma',
        'accents': r'\hat{x} + \tilde{y} + \vec{z}',
        'text': r'\text{Hello } x + y',
        'binom': r'\binom{n}{k} = \frac{n!}{k!(n-k)!}',
        'sum_product': r'\sum_{i=1}^n i = \prod_{j=1}^m j',
        'escaped_dollar': r'Price: \$100',
        'multiline': r'\begin{multline} a + b + c + d + e + f \\ + g + h + i = j \end{multline}'
    }


@pytest.fixture
def sample_mathml():
    """Sample MathML for testing."""
    return {
        'simple': '''
            <math xmlns="http://www.w3.org/1998/Math/MathML">
                <mi>x</mi><mo>+</mo><mi>y</mi>
            </math>
        ''',
        'fraction': '''
            <math xmlns="http://www.w3.org/1998/Math/MathML">
                <mfrac>
                    <mi>a</mi>
                    <mi>b</mi>
                </mfrac>
            </math>
        ''',
        'matrix': '''
            <math xmlns="http://www.w3.org/1998/Math/MathML">
                <mrow>
                    <mo>(</mo>
                    <mtable>
                        <mtr>
                            <mtd><mi>a</mi></mtd>
                            <mtd><mi>b</mi></mtd>
                        </mtr>
                        <mtr>
                            <mtd><mi>c</mi></mtd>
                            <mtd><mi>d</mi></mtd>
                        </mtr>
                    </mtable>
                    <mo>)</mo>
                </mrow>
            </math>
        '''
    }


@pytest.fixture
def sample_image():
    """Create sample image for OCR testing."""
    # Create a simple white image with black text
    img = Image.new('RGB', (200, 100), color='white')
    return img


@pytest.fixture
def latex_parser():
    """LaTeX parser instance."""
    return LaTeXParser()


@pytest.fixture
def mathml_converter():
    """MathML converter instance."""
    return MathMLConverter()


@pytest.fixture
def mock_ocr_config():
    """Mock OCR configuration."""
    return OCRConfig(
        mathpix_api_id="mock_id",
        mathpix_api_key="mock_key",
        enable_preprocessing=False,
        enable_fallback=False,
        cache_results=False
    )


@pytest.fixture
def sample_pdf_content():
    """Sample PDF-like content for testing."""
    return {
        'text': """
        This is a sample document with math formulas.
        
        The quadratic formula is $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$.
        
        Here's a matrix equation:
        $$
        \\begin{bmatrix}
        1 & 2 \\\\
        3 & 4
        \\end{bmatrix}
        \\begin{bmatrix}
        x \\\\
        y
        \\end{bmatrix}
        =
        \\begin{bmatrix}
        5 \\\\
        6
        \\end{bmatrix}
        $$
        
        And some inline math: $\\alpha + \\beta = \\gamma$.
        """,
        'page_bounds': (0, 0, 595, 842),  # A4 size in points
        'line_info': [
            {'x0': 50, 'y0': 100, 'width': 495, 'height': 20},
            {'x0': 50, 'y0': 140, 'width': 495, 'height': 20},
            {'x0': 50, 'y0': 180, 'width': 495, 'height': 20},
        ]
    }


@pytest.fixture
def sample_stem_content():
    """Sample STEM content."""
    return STEMContent(
        formulas=[
            MathFormula(
                latex="E = mc^2",
                math_type=MathType.DISPLAY,
                position=(100, 200, 200, 50),
                page_number=1,
                confidence=0.95
            ),
            MathFormula(
                latex="\\int_0^1 x^2 dx",
                math_type=MathType.INLINE,
                position=(150, 300, 150, 30),
                page_number=1,
                confidence=0.90
            )
        ],
        code_blocks=[],
        chemical_formulas=[],
        diagrams=[]
    )
