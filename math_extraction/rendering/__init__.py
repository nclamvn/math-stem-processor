from typing import Optional, Union, List
from pathlib import Path
import logging
from .base_renderer import BaseRenderer, RenderConfig, RenderResult
from .latex_renderer import LaTeXRenderer, SymPyRenderer, MatplotlibRenderer
from .mathml_renderer import MathMLRenderer
from .diagram_renderer import DiagramRenderer


logger = logging.getLogger(__name__)


class MathRenderer:
    """Main interface for rendering mathematical content."""
    
    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config or RenderConfig()
        
        # Initialize available renderers
        self.latex_renderers = [
            LaTeXRenderer(),
            MatplotlibRenderer(),
            SymPyRenderer()
        ]
        
        self.mathml_renderer = MathMLRenderer()
        self.diagram_renderer = DiagramRenderer()
        
        # Find first available LaTeX renderer
        self.default_latex_renderer = None
        for renderer in self.latex_renderers:
            if renderer.is_available():
                self.default_latex_renderer = renderer
                logger.info(f"Using {renderer.__class__.__name__} as default LaTeX renderer")
                break
                
        if not self.default_latex_renderer:
            logger.warning("No LaTeX renderer available")
            
    def render_latex(self, latex: str, config: Optional[RenderConfig] = None,
                    renderer: Optional[str] = None) -> RenderResult:
        """Render LaTeX to image."""
        config = config or self.config
        
        # Use specified renderer if provided
        if renderer:
            for r in self.latex_renderers:
                if r.__class__.__name__.lower() == renderer.lower():
                    if r.is_available():
                        return r.render_latex(latex, config)
                    else:
                        return RenderResult(error=f"{renderer} not available")
                        
            return RenderResult(error=f"Unknown renderer: {renderer}")
            
        # Use default renderer
        if self.default_latex_renderer:
            result = self.default_latex_renderer.render_latex(latex, config)
            
            # Try fallback renderers if failed
            if not result.is_valid:
                logger.warning(f"Primary renderer failed: {result.error}")
                
                for renderer in self.latex_renderers:
                    if renderer != self.default_latex_renderer and renderer.is_available():
                        logger.info(f"Trying {renderer.__class__.__name__}")
                        fallback_result = renderer.render_latex(latex, config)
                        if fallback_result.is_valid:
                            return fallback_result
                            
            return result
        else:
            return RenderResult(error="No LaTeX renderer available")
            
    def render_mathml(self, mathml: str, config: Optional[RenderConfig] = None) -> RenderResult:
        """Render MathML to image."""
        config = config or self.config
        
        if self.mathml_renderer.is_available():
            return self.mathml_renderer.render_mathml(mathml, config)
        else:
            return RenderResult(error="MathML renderer not available")
            
    def render_latex_batch(self, latex_list: List[str], 
                          config: Optional[RenderConfig] = None) -> List[RenderResult]:
        """Render multiple LaTeX expressions."""
        config = config or self.config
        results = []
        
        for latex in latex_list:
            results.append(self.render_latex(latex, config))
            
        return results
        
    def render_diagram(self, diagram_type: str, data: dict,
                      config: Optional[RenderConfig] = None) -> RenderResult:
        """Render scientific diagram."""
        config = config or self.config
        
        if diagram_type == "chemical":
            if "smiles" in data:
                return self.diagram_renderer.render_chemical_structure(
                    data["smiles"], config
                )
        elif diagram_type in ["function", "scatter", "vector_field"]:
            return self.diagram_renderer.render_plot(diagram_type, data, config)
            
        return RenderResult(error=f"Unknown diagram type: {diagram_type}")
        
    def get_available_renderers(self) -> dict:
        """Get list of available renderers."""
        return {
            "latex": [r.__class__.__name__ for r in self.latex_renderers if r.is_available()],
            "mathml": self.mathml_renderer.is_available(),
            "diagram": self.diagram_renderer.is_available()
        }
        
    def clear_cache(self):
        """Clear all renderer caches."""
        for renderer in self.latex_renderers:
            if hasattr(renderer, 'clear_cache'):
                renderer.clear_cache()


def render_math(content: str, content_type: str = "latex",
               config: Optional[RenderConfig] = None) -> RenderResult:
    """Convenience function to render mathematical content."""
    renderer = MathRenderer(config)
    
    if content_type == "latex":
        return renderer.render_latex(content)
    elif content_type == "mathml":
        return renderer.render_mathml(content)
    else:
        return RenderResult(error=f"Unknown content type: {content_type}")


__all__ = [
    'MathRenderer',
    'RenderConfig',
    'RenderResult',
    'render_math',
    'LaTeXRenderer',
    'MathMLRenderer',
    'DiagramRenderer'
]
