# Tests

This directory contains unit tests for the Math & STEM Content Processor.

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_formula_detector.py

# Run with coverage
pytest --cov=math_extraction tests/
Test Structure

conftest.py - Test fixtures and configuration
test_formula_detector.py - Tests for formula detection
test_latex_parser.py - Tests for LaTeX parsing
test_mathml_converter.py - Tests for MathML conversion
test_math_ocr.py - Tests for OCR functionality
test_rendering.py - Tests for rendering module
test_integration.py - Integration tests
test_utils.py - Test utilities
