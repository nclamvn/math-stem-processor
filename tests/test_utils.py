import pytest
import numpy as np
from PIL import Image
from pathlib import Path


def create_test_image_with_text(text: str, size=(400, 200)) -> Image.Image:
    """Create test image with text."""
    from PIL import ImageDraw, ImageFont
    
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to use a font
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 20)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
        
    draw.text((50, 50), text, fill='black', font=font)
    
    return img


def create_math_image(formula: str) -> Image.Image:
    """Create image with math formula (simulated)."""
    # This is a placeholder - in real tests, you might use matplotlib
    return create_test_image_with_text(formula)


def assert_mathml_valid(mathml: str):
    """Assert that MathML is valid."""
    assert '<math' in mathml
    assert 'xmlns="http://www.w3.org/1998/Math/MathML"' in mathml
    
    # Check balanced tags
    import re
    open_tags = re.findall(r'<(\w+)', mathml)
    close_tags = re.findall(r'</(\w+)>', mathml)
    
    # Simple balance check
    for tag in set(open_tags):
        if tag not in ['math']:  # Self-closing tags
            assert open_tags.count(tag) == close_tags.count(tag), f"Unbalanced tag: {tag}"


def compare_images(img1: Image.Image, img2: Image.Image, threshold=0.95) -> bool:
    """Compare two images for similarity."""
    if img1.size != img2.size:
        return False
        
    # Convert to numpy arrays
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    # Calculate similarity
    diff = np.abs(arr1.astype(float) - arr2.astype(float))
    similarity = 1 - (diff.sum() / (arr1.size * 255))
    
    return similarity >= threshold


def generate_test_document(num_formulas: int = 10) -> str:
    """Generate test document with formulas."""
    doc_parts = ["# Test Document\n\n"]
    
    for i in range(num_formulas):
        doc_parts.append(f"Section {i+1}: ")
        
        if i % 3 == 0:
            doc_parts.append(f"Inline formula $x_{{{i}}} + y_{{{i}}} = z_{{{i}}}$.\n\n")
        elif i % 3 == 1:
            doc_parts.append(f"Display formula:\n$$\\frac{{a_{{{i}}}}}{{b_{{{i}}}}} = c_{{{i}}}$$\n\n")
        else:
            doc_parts.append(f"Matrix:\n\\begin{{pmatrix}} {i} & {i+1} \\\\ {i+2} & {i+3} \\end{{pmatrix}}\n\n")
            
    return ''.join(doc_parts)


class MockHTTPClient:
    """Mock HTTP client for testing."""
    
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.call_count = 0
        self.last_request = None
        
    def post(self, url, **kwargs):
        self.call_count += 1
        self.last_request = {'url': url, 'kwargs': kwargs}
        
        if url in self.responses:
            return self.responses[url]
            
        # Default response
        from unittest.mock import Mock
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            'latex_styled': 'x + y',
            'latex_confidence': 0.9
        }
        return response


def save_test_results(results: dict, output_dir: Path):
    """Save test results for inspection."""
    import json
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    summary = {
        'total': len(results),
        'passed': sum(1 for r in results.values() if r.get('passed')),
        'failed': sum(1 for r in results.values() if not r.get('passed'))
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    # Save detailed results
    with open(output_dir / 'details.json', 'w') as f:
        json.dump(results, f, indent=2)
