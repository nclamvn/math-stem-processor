import pytest
from pathlib import Path
from PIL import Image
import numpy as np
from math_extraction import (
    FormulaDetector, LaTeXParser, MathMLConverter,
    MathOCR, OCRConfig, MathRenderer, RenderConfig
)


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_latex_to_mathml_pipeline(self, sample_latex_formulas):
        """Test LaTeX detection → parsing → MathML conversion."""
        # Step 1: Detect formulas
        detector = FormulaDetector()
        text = f"The formula is {sample_latex_formulas['simple_inline']} here."
        formulas = detector.detect_formulas(text, page_number=1)
        
        assert len(formulas) > 0
        formula = formulas[0]
        
        # Step 2: Parse LaTeX
        parser = LaTeXParser()
        structure = parser.parse(formula.latex)
        assert structure.is_valid
        
        # Step 3: Convert to MathML
        converter = MathMLConverter()
        mathml = converter.convert(formula.latex)
        assert '<math' in mathml
        assert '<mi>x</mi>' in mathml
        
    def test_ocr_to_latex_pipeline(self):
        """Test image → OCR → LaTeX extraction."""
        # Create simple test image
        img = Image.new('RGB', (200, 100), color='white')
        img_array = np.array(img)
        
        # Add some black pixels to simulate text
        img_array[40:60, 50:150] = 0
        img = Image.fromarray(img_array)
        
        # OCR (will likely fail without real math content, but shouldn't crash)
        config = OCRConfig(enable_preprocessing=True, enable_fallback=True)
        ocr = MathOCR(config)
        
        result = ocr.recognize_image(img)
        # Should return a result (even if empty/failed)
        assert isinstance(result.latex, str)
        
    def test_latex_to_image_pipeline(self, sample_latex_formulas):
        """Test LaTeX → rendering → image."""
        renderer = MathRenderer()
        
        # Skip if no renderer available
        available = renderer.get_available_renderers()
        if not available['latex']:
            pytest.skip("No LaTeX renderer available")
            
        # Render simple formula
        result = renderer.render_latex(sample_latex_formulas['fraction'])
        
        if result.is_valid:
            assert result.image is not None
            assert result.size[0] > 0
            assert result.size[1] > 0
            
    def test_full_document_processing(self, sample_pdf_content):
        """Test processing a full document."""
        # Step 1: Formula detection
        detector = FormulaDetector()
        formulas = detector.detect_formulas(
            sample_pdf_content['text'],
            page_number=1,
            page_bounds=sample_pdf_content['page_bounds'],
            line_info=sample_pdf_content['line_info']
        )
        
        assert len(formulas) >= 3  # Should find inline and display formulas
        
        # Step 2: Parse all formulas
        parser = LaTeXParser()
        parsed_formulas = []
        
        for formula in formulas:
            structure = parser.parse(formula.latex)
            if structure.is_valid:
                parsed_formulas.append((formula, structure))
                
        assert len(parsed_formulas) > 0
        
        # Step 3: Convert to MathML
        converter = MathMLConverter()
        mathml_results = []
        
        for formula, structure in parsed_formulas:
            mathml = converter.convert(formula.latex)
            is_valid, errors = converter.validate_mathml(mathml)
            mathml_results.append((formula, mathml, is_valid))
            
        # Most should be valid
        valid_count = sum(1 for _, _, valid in mathml_results if valid)
        assert valid_count >= len(mathml_results) * 0.8
        
    def test_matrix_processing_pipeline(self):
        """Test matrix detection and conversion."""
        # Matrix LaTeX
        matrix_latex = r"""
        The system of equations can be written as:
        \begin{pmatrix}
        1 & 2 & 3 \\
        4 & 5 & 6 \\
        7 & 8 & 9
        \end{pmatrix}
        \begin{pmatrix}
        x \\
        y \\
        z
        \end{pmatrix}
        =
        \begin{pmatrix}
        10 \\
        11 \\
        12
        \end{pmatrix}
        """
        
        # Detect
        detector = FormulaDetector()
        formulas = detector.detect_formulas(matrix_latex, page_number=1)
        
        # Should find matrix environment
        matrix_formulas = [f for f in formulas if 'matrix' in f.latex]
        assert len(matrix_formulas) > 0
        
        # Convert to MathML
        converter = MathMLConverter()
        for formula in matrix_formulas:
            mathml = converter.convert(formula.latex)
            assert '<mtable>' in mathml
            assert '<mtr>' in mathml
            assert '<mtd>' in mathml
            
    def test_complex_math_document(self):
        """Test processing document with various math elements."""
        document = r"""
        \section{Introduction}
        
        The quadratic formula is given by:
        $$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$
        
        For the integral $\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$,
        we can use the substitution method.
        
        Consider the matrix equation:
        \begin{align}
        A &= \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \\
        \det(A) &= 1 \cdot 4 - 2 \cdot 3 = -2
        \end{align}
        
        The series $\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6}$ converges.
        """
        
        # Process document
        detector = FormulaDetector()
        formulas = detector.detect_formulas(document, page_number=1)
        
        # Validate statistics
        stats = detector.validate_extraction(formulas)
        
        assert stats['total'] >= 4
        assert 'display' in stats['by_type']
        assert 'inline' in stats['by_type']
        assert stats['quality_metrics']['has_math_symbols'] > 0
        
    def test_error_handling_pipeline(self):
        """Test error handling throughout pipeline."""
        # Malformed LaTeX
        bad_latex = r"\frac{a}{b"  # Missing closing brace
        
        # Parser should handle
        parser = LaTeXParser()
        structure = parser.parse(bad_latex)
        assert not structure.is_valid
        assert len(structure.errors) > 0
        
        # Converter should still produce something
        converter = MathMLConverter()
        mathml = converter.convert(bad_latex)
        assert '<math' in mathml  # Should have fallback
        
    def test_caching_performance(self, sample_latex_formulas):
        """Test caching improves performance."""
        import time
        
        converter = MathMLConverter()
        converter.clear_cache()
        
        # First conversion
        start = time.time()
        for _ in range(10):
            converter.convert(sample_latex_formulas['complex'])
        first_time = time.time() - start
        
        # Second conversion (should use cache)
        start = time.time()
        for _ in range(10):
            converter.convert(sample_latex_formulas['complex'])
        second_time = time.time() - start
        
        # Cache should make it faster (or at least not slower)
        assert second_time <= first_time * 1.1
        
        # Check cache stats
        stats = converter.get_cache_stats()
        assert stats['hits'] > 0


class TestErrorRecovery:
    """Test error recovery and edge cases."""
    
    def test_empty_document(self):
        """Test with empty document."""
        detector = FormulaDetector()
        formulas = detector.detect_formulas("", page_number=1)
        assert len(formulas) == 0
        
        stats = detector.validate_extraction(formulas)
        assert stats['total'] == 0
        
    def test_invalid_unicode(self):
        """Test with invalid Unicode."""
        text = "Formula: $x + y$ \ud800"  # Invalid surrogate
        
        detector = FormulaDetector()
        # Should not crash
        formulas = detector.detect_formulas(text, page_number=1)
        assert len(formulas) >= 1
        
    def test_very_large_document(self):
        """Test with very large document."""
        # Create large document
        large_text = "Some text $x + y$ more text " * 1000
        
        detector = FormulaDetector()
        formulas = detector.detect_formulas(large_text, page_number=1)
        
        # Should handle efficiently
        assert len(formulas) == 1000
        
    def test_concurrent_processing(self):
        """Test concurrent processing safety."""
        import threading
        
        def process_formula(formula, results, index):
            parser = LaTeXParser()
            converter = MathMLConverter()
            
            structure = parser.parse(formula)
            mathml = converter.convert(formula)
            results[index] = (structure.is_valid, mathml)
            
        formulas = ["x + y", r"\frac{a}{b}", r"\int_0^1 x dx"]
        results = [None] * len(formulas)
        threads = []
        
        for i, formula in enumerate(formulas):
            t = threading.Thread(
                target=process_formula,
                args=(formula, results, i)
            )
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        # All should complete successfully
        assert all(r is not None for r in results)
        assert all(r[0] for r in results)  # All valid


class TestPerformance:
    """Performance benchmarks."""
    
    @pytest.mark.slow
    def test_large_batch_processing(self):
        """Test processing large batch of formulas."""
        # Generate many formulas
        formulas = []
        for i in range(100):
            formulas.append(f"x_{i} + y_{i} = z_{i}")
            formulas.append(rf"\frac{{a_{i}}}{{b_{i}}}")
            formulas.append(rf"\int_0^{i} x^{i} dx")
            
        detector = FormulaDetector()
        parser = LaTeXParser()
        converter = MathMLConverter()
        
        # Process all
        for formula in formulas:
            text = f"Formula: ${formula}$"
            detected = detector.detect_formulas(text, page_number=1)
            
            if detected:
                structure = parser.parse(detected[0].latex)
                mathml = converter.convert(detected[0].latex)
                
        # Should complete without errors
        assert True
        
    @pytest.mark.slow
    def test_memory_usage(self):
        """Test memory usage stays reasonable."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process many formulas
        converter = MathMLConverter()
        for i in range(1000):
            formula = f"x^{i} + y^{i} = z^{i}"
            converter.convert(formula)
            
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB)
        assert memory_increase < 100
        
        # Cache should have limit
        stats = converter.get_cache_stats()
        assert stats['cache_size'] <= converter.MAX_CACHE_SIZE
