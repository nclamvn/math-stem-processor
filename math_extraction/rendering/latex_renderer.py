import os
import tempfile
import subprocess
import hashlib
import shutil
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging
from PIL import Image
import numpy as np
from .base_renderer import BaseRenderer, RenderConfig, RenderResult


logger = logging.getLogger(__name__)


class LaTeXRenderer(BaseRenderer):
    """Render LaTeX to images using system LaTeX installation."""
    
    LATEX_TEMPLATE = r"""
\documentclass[12pt,border={{{padding}pt}}]{{standalone}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{amsfonts}}
\usepackage{{mathtools}}
\usepackage{{xcolor}}
\usepackage{{graphicx}}
\usepackage{{physics}}
\usepackage{{siunitx}}
\usepackage{{cancel}}
\usepackage{{mathrsfs}}
\usepackage{{bbm}}
\usepackage{{dsfont}}
\usepackage{{esint}}
\pagestyle{{empty}}
\begin{{document}}
\color{{{color}}}
{content}
\end{{document}}
"""
    
    def __init__(self, latex_cmd: str = "pdflatex", cache_dir: Optional[Path] = None):
        self.latex_cmd = latex_cmd
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "latex_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._available = None
        
    def is_available(self) -> bool:
        """Check if LaTeX is available."""
        if self._available is None:
            self._available = self._check_latex()
        return self._available
        
    def _check_latex(self) -> bool:
        """Check if LaTeX installation is available."""
        try:
            result = subprocess.run(
                [self.latex_cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
            
    def render_latex(self, latex: str, config: RenderConfig) -> RenderResult:
        """Render LaTeX to image."""
        if not self.is_available():
            return RenderResult(error="LaTeX not available")
            
        # Check cache
        cache_key = self._get_cache_key(latex, config)
        cached_path = self.cache_dir / f"{cache_key}.png"
        
        if config.cache_renders and cached_path.exists():
            try:
                image = Image.open(cached_path)
                return RenderResult(
                    image=image,
                    size=image.size,
                    metadata={"cached": True, "renderer": "latex"}
                )
            except:
                logger.warning(f"Failed to load cached image: {cached_path}")
                
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Prepare LaTeX content
            if not latex.startswith('$') and not latex.startswith('\\['):
                # Assume display mode if not specified
                content = f"\\[{latex}\\]"
            else:
                content = latex
                
            # Create LaTeX document
            tex_content = self.LATEX_TEMPLATE.format(
                padding=config.padding,
                color=config.text_color,
                content=content
            )
            
            tex_file = temp_path / "formula.tex"
            tex_file.write_text(tex_content, encoding='utf-8')
            
            # Compile LaTeX to PDF
            try:
                result = subprocess.run(
                    [self.latex_cmd, "-interaction=nonstopmode", "-halt-on-error", "formula.tex"],
                    cwd=temp_path,
                    capture_output=True,
                    text=True,
                    timeout=config.timeout
                )
                
                if result.returncode != 0:
                    error_msg = self._extract_error_message(result.stdout + result.stderr)
                    return RenderResult(error=f"LaTeX compilation failed: {error_msg}")
                    
            except subprocess.TimeoutExpired:
                return RenderResult(error="LaTeX compilation timeout")
            except Exception as e:
                return RenderResult(error=f"LaTeX compilation error: {str(e)}")
                
            # Convert PDF to image
            pdf_file = temp_path / "formula.pdf"
            if not pdf_file.exists():
                return RenderResult(error="PDF output not found")
                
            # Convert using different methods
            image = self._pdf_to_image(pdf_file, config)
            
            if image:
                # Apply background
                if not config.transparent_background:
                    background = Image.new('RGBA', image.size, config.background_color)
                    if image.mode == 'RGBA':
                        background.paste(image, mask=image.split()[3])
                    else:
                        background.paste(image)
                    image = background.convert('RGB')
                    
                # Cache result
                if config.cache_renders:
                    try:
                        image.save(cached_path, "PNG")
                    except:
                        logger.warning(f"Failed to cache image: {cached_path}")
                        
                return RenderResult(
                    image=image,
                    size=image.size,
                    metadata={"renderer": "latex", "dpi": config.dpi}
                )
            else:
                return RenderResult(error="Failed to convert PDF to image")
                
    def render_mathml(self, mathml: str, config: RenderConfig) -> RenderResult:
        """Render MathML by converting to LaTeX first."""
        # This would require MathML to LaTeX conversion
        # For now, return error
        return RenderResult(error="MathML rendering not implemented for LaTeX renderer")
        
    def _get_cache_key(self, latex: str, config: RenderConfig) -> str:
        """Generate cache key for render."""
        key_parts = [
            latex,
            str(config.dpi),
            str(config.font_size),
            config.font_family,
            config.text_color,
            config.background_color,
            str(config.padding)
        ]
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
        
    def _extract_error_message(self, output: str) -> str:
        """Extract meaningful error message from LaTeX output."""
        lines = output.split('\n')
        
        # Look for error indicators
        error_lines = []
        for i, line in enumerate(lines):
            if line.startswith('!'):
                error_lines.append(line)
                # Get next few lines for context
                for j in range(i + 1, min(i + 3, len(lines))):
                    if lines[j].strip():
                        error_lines.append(lines[j])
                        
        if error_lines:
            return " ".join(error_lines[:3])
        else:
            # Return last non-empty lines
            non_empty = [l for l in lines if l.strip()]
            return " ".join(non_empty[-3:]) if non_empty else "Unknown error"
            
    def _pdf_to_image(self, pdf_path: Path, config: RenderConfig) -> Optional[Image.Image]:
        """Convert PDF to image using available tools."""
        # Try pdf2image (using Poppler)
        try:
            from pdf2image import convert_from_path
            
            images = convert_from_path(
                pdf_path,
                dpi=config.dpi,
                transparent=config.transparent_background,
                fmt='png',
                single_file=True
            )
            
            if images:
                return images[0]
                
        except ImportError:
            logger.warning("pdf2image not available")
        except Exception as e:
            logger.warning(f"pdf2image failed: {e}")
            
        # Try ImageMagick/Wand
        try:
            from wand.image import Image as WandImage
            
            with WandImage(filename=str(pdf_path), resolution=config.dpi) as img:
                img.format = 'png'
                img.background_color = 'transparent' if config.transparent_background else 'white'
                img.alpha_channel = 'remove'
                
                # Convert to PIL Image
                blob = img.make_blob()
                return Image.open(io.BytesIO(blob))
                
        except ImportError:
            logger.warning("Wand not available")
        except Exception as e:
            logger.warning(f"Wand failed: {e}")
            
        # Try subprocess with ImageMagick convert
        try:
            output_path = pdf_path.with_suffix('.png')
            
            cmd = [
                'convert',
                '-density', str(config.dpi),
                '-background', 'transparent' if config.transparent_background else 'white',
                '-alpha', 'remove',
                str(pdf_path),
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            
            if result.returncode == 0 and output_path.exists():
                return Image.open(output_path)
                
        except:
            logger.warning("ImageMagick convert not available")
            
        return None
        
    def clear_cache(self):
        """Clear the render cache."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)


class SymPyRenderer(BaseRenderer):
    """Render LaTeX using SymPy's preview functionality."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "sympy_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._available = None
        
    def is_available(self) -> bool:
        """Check if SymPy is available."""
        if self._available is None:
            try:
                import sympy
                from sympy import preview
                self._available = True
            except ImportError:
                self._available = False
        return self._available
        
    def render_latex(self, latex: str, config: RenderConfig) -> RenderResult:
        """Render LaTeX using SymPy."""
        if not self.is_available():
            return RenderResult(error="SymPy not available")
            
        try:
            from sympy import preview, symbols
            from sympy.parsing.latex import parse_latex
            
            # Try to parse as SymPy expression
            try:
                expr = parse_latex(latex)
            except:
                # If parsing fails, use raw LaTeX
                expr = latex
                
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                output_file = tmp.name
                
            # Set viewer to 'file' to save to file instead of displaying
            preview(
                expr,
                output='png',
                filename=output_file,
                euler=False,
                dvioptions=[f'-D {config.dpi}'],
                viewer='file'
            )
            
            # Load image
            if Path(output_file).exists():
                image = Image.open(output_file)
                
                # Clean up
                os.unlink(output_file)
                
                return RenderResult(
                    image=image,
                    size=image.size,
                    metadata={"renderer": "sympy", "dpi": config.dpi}
                )
            else:
                return RenderResult(error="SymPy failed to generate image")
                
        except Exception as e:
            logger.error(f"SymPy render error: {e}")
            return RenderResult(error=f"SymPy render error: {str(e)}")
            
    def render_mathml(self, mathml: str, config: RenderConfig) -> RenderResult:
        """Render MathML (not supported by SymPy)."""
        return RenderResult(error="MathML rendering not supported by SymPy")


class MatplotlibRenderer(BaseRenderer):
    """Render LaTeX using Matplotlib."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "matplotlib_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._available = None
        
    def is_available(self) -> bool:
        """Check if Matplotlib is available."""
        if self._available is None:
            try:
                import matplotlib
                import matplotlib.pyplot as plt
                self._available = True
            except ImportError:
                self._available = False
        return self._available
        
    def render_latex(self, latex: str, config: RenderConfig) -> RenderResult:
        """Render LaTeX using Matplotlib."""
        if not self.is_available():
            return RenderResult(error="Matplotlib not available")
            
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            from matplotlib import rcParams
            
            # Configure matplotlib for better math rendering
            rcParams['mathtext.fontset'] = 'cm'  # Computer Modern
            rcParams['font.family'] = 'serif'
            rcParams['font.size'] = config.font_size
            
            # Create figure
            fig = plt.figure(figsize=(10, 2))
            fig.patch.set_facecolor('none' if config.transparent_background else config.background_color)
            
            # Ensure LaTeX is properly formatted
            if not latex.startswith('$'):
                latex = f'${latex}$'
                
            # Add text
            text = fig.text(
                0.5, 0.5, latex,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=config.font_size,
                color=config.text_color,
                transform=fig.transFigure
            )
            
            # Get bounding box
            fig.canvas.draw()
            bbox = text.get_window_extent()
            
            # Adjust figure size to fit text
            width = bbox.width / fig.dpi + 2 * config.padding / fig.dpi
            height = bbox.height / fig.dpi + 2 * config.padding / fig.dpi
            fig.set_size_inches(width, height)
            
            # Re-center text
            text.set_position((0.5, 0.5))
            
            # Save to buffer
            from io import BytesIO
            buffer = BytesIO()
            fig.savefig(
                buffer,
                format='png',
                dpi=config.dpi,
                transparent=config.transparent_background,
                bbox_inches='tight',
                pad_inches=config.padding / config.dpi
            )
            
            plt.close(fig)
            
            # Load image
            buffer.seek(0)
            image = Image.open(buffer)
            
            return RenderResult(
                image=image,
                size=image.size,
                metadata={"renderer": "matplotlib", "dpi": config.dpi}
            )
            
        except Exception as e:
            logger.error(f"Matplotlib render error: {e}")
            return RenderResult(error=f"Matplotlib render error: {str(e)}")
            
    def render_mathml(self, mathml: str, config: RenderConfig) -> RenderResult:
        """Render MathML (not directly supported)."""
        return RenderResult(error="MathML rendering not supported by Matplotlib")
