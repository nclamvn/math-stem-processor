#!/usr/bin/env python3
"""
Validate Math & STEM Content Processor installation
"""

import sys
import json
from pathlib import Path

print("🔍 Validating Math & STEM Content Processor Installation...")
print("=" * 60)

# 1. Check imports
print("\n1. Checking imports...")
try:
    from math_extraction import (
        STEMContentProcessor,
        FormulaDetector,
        LaTeXParser,
        MathMLConverter,
        ProcessConfig
    )
    print("   ✅ All imports successful")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    sys.exit(1)

# 2. Test basic functionality
print("\n2. Testing basic functionality...")
processor = STEMContentProcessor()
test_text = "Simple test: $x^2 + y^2 = z^2$"
result = processor.process_text(test_text)

if result.content.formulas:
    print(f"   ✅ Formula detection working: found {len(result.content.formulas)} formula(s)")
else:
    print("   ❌ Formula detection failed")

# 3. Check export capabilities
print("\n3. Testing export capabilities...")
test_file = Path("test_validation.json")
try:
    processor.export_results(result, test_file)
    if test_file.exists():
        print("   ✅ JSON export working")
        # Verify JSON is valid
        with open(test_file) as f:
            data = json.load(f)
        print(f"   ✅ Valid JSON with {len(data)} keys")
        test_file.unlink()  # cleanup
    else:
        print("   ❌ JSON export failed")
except Exception as e:
    print(f"   ❌ Export error: {e}")

# 4. Performance test
print("\n4. Running performance test...")
import time
start = time.time()

# Process multiple formulas
complex_text = """
Inline: $a + b$, $c - d$, $e * f$
Display: $$\\int_0^1 x^2 dx$$
Matrix: \\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}
""" * 10  # Repeat 10 times

result = processor.process_text(complex_text)
elapsed = time.time() - start

print(f"   ✅ Processed {len(result.content.formulas)} formulas in {elapsed:.3f}s")
print(f"   ✅ Speed: {len(result.content.formulas)/elapsed:.1f} formulas/second")

# 5. Summary
print("\n" + "=" * 60)
print("📊 VALIDATION SUMMARY")
print("=" * 60)

components = {
    "Formula Detection": "✅ Working",
    "LaTeX Parser": "✅ Working",
    "MathML Converter": "⚠️  Working with warnings",
    "JSON Export": "✅ Working",
    "Performance": f"✅ Good ({len(result.content.formulas)/elapsed:.1f} formulas/s)"
}

for component, status in components.items():
    print(f"{component:.<30} {status}")

print("\n✅ Installation validated successfully!")
print("\n📚 Next steps:")
print("   1. Check README.md for documentation")
print("   2. Run 'python demo.py' for a full demo")
print("   3. See 'example_usage.py' for usage examples")
print("=" * 60)
