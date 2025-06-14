#!/usr/bin/env python3
"""Simple example of using the Math & STEM Content Processor"""

from math_extraction import STEMContentProcessor

# Create processor
processor = STEMContentProcessor()

# Your text with math
text = """
The famous equation $E = mc^2$ was discovered by Einstein.

The quadratic formula is:
$$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$

This is used to solve equations like $ax^2 + bx + c = 0$.
"""

# Process the text
print("Processing text...")
result = processor.process_text(text)

# Show results
print(f"\nFound {len(result.content.formulas)} formulas:")
for i, formula in enumerate(result.content.formulas, 1):
    print(f"\n{i}. {formula.latex}")
    print(f"   Type: {formula.math_type.value}")
    print(f"   Confidence: {formula.confidence:.0%}")

# Save to file
processor.export_results(result, "my_formulas.json")
print("\nResults saved to my_formulas.json")
