#!/usr/bin/env python3
"""
Demo script for Math & STEM Content Processor
"""

from math_extraction import STEMContentProcessor
from pathlib import Path

# Khởi tạo processor
processor = STEMContentProcessor()

# Document mẫu
sample_document = """
# Introduction to Calculus

The derivative of a function $f(x)$ at point $a$ is defined as:

$$f'(a) = \lim_{h \to 0} \frac{f(a+h) - f(a)}{h}$$

For example, if $f(x) = x^2$, then:

$$f'(x) = 2x$$

## Integration

The fundamental theorem of calculus states:

$$\int_a^b f'(x) \, dx = f(b) - f(a)$$

Some common integrals:
- $\int x^n \, dx = \frac{x^{n+1}}{n+1} + C$ (for $n \neq -1$)
- $\int e^x \, dx = e^x + C$
- $\int \sin(x) \, dx = -\cos(x) + C$

## Matrix Operations

Matrix multiplication:
\begin{equation}
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
ax + by \\
cx + dy
\end{pmatrix}
\end{equation}
"""

print("Processing sample document...")
print("=" * 60)

# Process document
result = processor.process_text(sample_document)

# Hiển thị kết quả
print(f"\nProcessing completed in {result.processing_time:.3f} seconds")
print(f"Found {len(result.content.formulas)} mathematical formulas:")
print()

# Hiển thị chi tiết từng formula
for i, formula in enumerate(result.content.formulas, 1):
    print(f"{i}. Type: {formula.math_type.value}")
    print(f"   LaTeX: {formula.latex}")
    print(f"   Confidence: {formula.confidence:.2%}")
    if formula.mathml:
        print(f"   MathML: {formula.mathml[:50]}...")
    print()

# Export kết quả
output_file = Path("demo_output.json")
if processor.export_results(result, output_file):
    print(f"\nResults exported to: {output_file}")
    
# Hiển thị thống kê
stats = result.content.get_statistics()
print(f"\nStatistics:")
print(f"- Total formulas: {stats['total_formulas']}")
print(f"- Formula types: {stats['formula_types']}")

print("\nDemo completed!")
