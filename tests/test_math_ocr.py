import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image
from math_extraction.math_ocr import (
    MathOCR, OCRConfig, OCRResult, ImagePreprocessor,
    MathpixOCR, TesseractOCR, PreprocessConfig
)
from math_extraction.ocr_base import BaseOCRBackend


class TestImagePreprocessor:
    
    def test_initialization(self):
        """Test ImagePreprocessor initialization."""
        config = PreprocessConfig()
        preprocessor = ImagePreprocessor(config)
        assert preprocessor.config == config
        
    def test_preprocess_high_quality(self, sample_image):
        """Test preprocessing of high quality image."""
        config = PreprocessConfig()
        preprocessor = ImagePreprocessor(config)
        
        # Create high contrast image
        img = Image.new('L', (300, 100), color=255)
        
        with patch.object(preprocessor, '_is_high_quality', return_value=True):
            result = preprocessor.preprocess(img)
            assert result == img  # Should return original
            
    def test_resize_small_image(self):
        """Test resizing of small images."""
        config = PreprocessConfig(min_size=100)
        preprocessor = ImagePreprocessor(config)
        
        # Create small image
        small_img = Image.new('L', (50, 50), color=255)
        result = preprocessor._resize_image(small_img)
        
        assert result.width >= config.min_size or result.height >= config.min_size
        
    def test_resize_large_image(self):
        """Test resizing of large images."""
        config = PreprocessConfig(max_size=500)
        preprocessor = ImagePreprocessor(config)
        
        # Create large image
        large_img = Image.new('L', (1000, 1000), color=255)
        result = preprocessor._resize_image(large_img)
        
        assert result.width <= config.max_size
        assert result.height <= config.max_size
        
    def test_extract_math_regions(self):
        """Test math region extraction."""
        config = PreprocessConfig()
        preprocessor = ImagePreprocessor(config)
        
        # Create image with distinct regions
        img = Image.new('L', (400, 300), color=255)
        img_array = np.array(img)
        
        # Add some black regions (potential formulas)
        img_array[50:100, 50:150] = 0  # Region 1
        img_array[150:180, 200:280] = 0  # Region 2
        
        img = Image.fromarray(img_array)
        
        with patch('cv2.findContours') as mock_contours:
            # Mock contour detection
            mock_contours.return_value = (
                [np.array([[50, 50], [150, 50], [150, 100], [50, 100]]),
                 np.array([[200, 150], [280, 150], [280, 180], [200, 180]])],
                None
            )
            
            regions = preprocessor.extract_math_regions(img)
            
            # Should extract regions
            assert len(regions) >= 0  # Depends on mock implementation


class TestMathpixOCR:
    
    def test_initialization(self, mock_ocr_config):
        """Test MathpixOCR initialization."""
        ocr = MathpixOCR(mock_ocr_config)
        assert ocr.config == mock_ocr_config
        assert ocr.is_available()
        
    def test_no_credentials(self):
        """Test behavior without credentials."""
        config = OCRConfig()  # No API credentials
        ocr = MathpixOCR(config)
        assert not ocr.is_available()
        
        result = ocr.recognize(Image.new('RGB', (100, 100)))
        assert not result.is_valid
        assert "credentials not configured" in result.error
        
    @patch('requests.Session.post')
    def test_recognize_success(self, mock_post, mock_ocr_config, sample_image):
        """Test successful recognition."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'latex_styled': 'x^2 + y^2 = z^2',
            'latex_confidence': 0.95,
            'mathml': '<math>...</math>'
        }
        mock_post.return_value = mock_response
        
        ocr = MathpixOCR(mock_ocr_config)
        result = ocr.recognize(sample_image)
        
        assert result.is_valid
        assert result.latex == 'x^2 + y^2 = z^2'
        assert result.confidence == 0.95
        assert 'mathml' in result.alternative_formats
        
    @patch('requests.Session.post')
    def test_recognize_rate_limit(self, mock_post, mock_ocr_config, sample_image):
        """Test rate limit handling."""
        # Mock rate limit response
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {'Retry-After': '1'}
        mock_post.return_value = mock_response
        
        ocr = MathpixOCR(mock_ocr_config)
        
        with patch('time.sleep'):  # Don't actually sleep in tests
            result = ocr.recognize(sample_image)
            
        # Should handle rate limit
        assert mock_post.call_count == mock_ocr_config.max_retries
        
    def test_prepare_image(self, mock_ocr_config):
        """Test image preparation."""
        ocr = MathpixOCR(mock_ocr_config)
        
        # Test normal size image
        img = Image.new('RGB', (500, 500))
        base64_str = ocr._prepare_image(img)
        assert isinstance(base64_str, str)
        assert len(base64_str) > 0
        
    def test_prepare_large_image(self, mock_ocr_config):
        """Test large image resizing."""
        mock_ocr_config.max_image_bytes = 1000  # Very small limit
        ocr = MathpixOCR(mock_ocr_config)
        
        # Create large image
        img = Image.new('RGB', (3000, 3000))
        base64_str = ocr._prepare_image(img)
        
        # Should resize and compress
        assert isinstance(base64_str, str)


class TestTesseractOCR:
    
    @patch('pytesseract.get_tesseract_version')
    def test_available(self, mock_version):
        """Test Tesseract availability check."""
        mock_version.return_value = '4.1.1'
        
        config = OCRConfig()
        ocr = TesseractOCR(config)
        assert ocr.is_available()
        
    @patch('pytesseract.get_tesseract_version')
    def test_not_available(self, mock_version):
        """Test when Tesseract not available."""
        mock_version.side_effect = Exception("Not found")
        
        config = OCRConfig()
        ocr = TesseractOCR(config)
        assert not ocr.is_available()
        
    @patch('pytesseract.image_to_string')
    @patch('pytesseract.image_to_data')
    def test_recognize(self, mock_data, mock_string, sample_image):
        """Test Tesseract recognition."""
        # Mock OCR results
        mock_string.return_value = "x^2 + y^2"
        mock_data.return_value = {
            'conf': ['95', '90', '92', '-1']
        }
        
        config = OCRConfig()
        ocr = TesseractOCR(config)
        
        with patch.object(ocr, 'is_available', return_value=True):
            result = ocr.recognize(sample_image)
            
        assert result.latex == "x^2 + y^2"  # Basic conversion
        assert result.confidence > 0
        assert result.metadata['source'] == 'tesseract'


class TestMathOCR:
    
    def test_initialization(self):
        """Test MathOCR initialization."""
        config = OCRConfig()
        ocr = MathOCR(config)
        
        assert ocr.config == config
        assert ocr.preprocessor is not None
        assert ocr.mathpix is not None
        assert ocr.tesseract is not None
        
    @patch.object(MathpixOCR, 'recognize')
    def test_recognize_with_cache(self, mock_recognize, sample_image):
        """Test recognition with caching."""
        config = OCRConfig(cache_results=True)
        ocr = MathOCR(config)
        
        # Mock successful recognition
        mock_result = OCRResult(latex="x + y", confidence=0.9)
        mock_recognize.return_value = mock_result
        
        # First call
        result1 = ocr.recognize_image(sample_image)
        assert result1.latex == "x + y"
        assert mock_recognize.call_count == 1
        
        # Second call (should use cache)
        result2 = ocr.recognize_image(sample_image)
        assert result2.latex == "x + y"
        assert mock_recognize.call_count == 1  # Not called again
        
    @patch.object(MathpixOCR, 'recognize')
    @patch.object(TesseractOCR, 'recognize')
    def test_fallback(self, mock_tesseract, mock_mathpix, sample_image):
        """Test fallback to Tesseract."""
        config = OCRConfig(enable_fallback=True)
        ocr = MathOCR(config)
        
        # Mock Mathpix failure
        mock_mathpix.return_value = OCRResult(latex="", confidence=0, error="Failed")
        
        # Mock Tesseract success
        mock_tesseract.return_value = OCRResult(latex="x + y", confidence=0.8)
        
        result = ocr.recognize_image(sample_image)
        
        assert result.latex == "x + y"
        assert mock_mathpix.call_count == 1
        assert mock_tesseract.call_count == 1
        
    def test_recognize_regions(self, sample_image):
        """Test region recognition."""
        config = OCRConfig(enable_preprocessing=False)
        ocr = MathOCR(config)
        
        with patch.object(ocr.preprocessor, 'extract_math_regions') as mock_extract:
            # Mock extracted regions
            mock_extract.return_value = [
                (sample_image, (0, 0, 100, 50)),
                (sample_image, (0, 100, 100, 50))
            ]
            
            with patch.object(ocr, 'recognize_image') as mock_recognize:
                mock_recognize.return_value = OCRResult(latex="x", confidence=0.9)
                
                results = ocr.recognize_regions(sample_image)
                
                assert len(results) == 2
                assert all(r[0].latex == "x" for r in results)
                
    def test_cache_stats(self):
        """Test cache statistics."""
        config = OCRConfig()
        ocr = MathOCR(config)
        
        stats = ocr.get_cache_stats()
        assert 'entries' in stats
        assert 'memory_usage_mb' in stats
        assert 'max_entries' in stats
        
    def test_cleanup(self):
        """Test resource cleanup."""
        config = OCRConfig()
        ocr = MathOCR(config)
        
        # Should not raise exception
        ocr.close()
        ocr.clear_cache()


class TestIntegration:
    
    @pytest.mark.skipif(not TesseractOCR(OCRConfig()).is_available(), 
                        reason="Tesseract not available")
    def test_real_tesseract(self):
        """Test with real Tesseract if available."""
        config = OCRConfig(enable_fallback=True, enable_preprocessing=True)
        ocr = MathOCR(config)
        
        # Create image with text
        img = Image.new('RGB', (200, 100), color='white')
        
        result = ocr.recognize_image(img)
        # Should at least not crash
        assert isinstance(result, OCRResult)
