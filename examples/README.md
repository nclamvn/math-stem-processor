Examples
This directory contains example scripts demonstrating how to use the Math & STEM Content Processor.
Examples

simple_example.py - Basic usage example
bashpython examples/simple_example.py

demo.py - Full feature demonstration
bashpython examples/demo.py

validate_installation.py - Validate your installation
bashpython examples/validate_installation.py


Quick Start
pythonfrom math_extraction import STEMContentProcessor

processor = STEMContentProcessor()
result = processor.process_text("$E = mc^2$")
