import os
import tempfile
import subprocess
from typing import Optional, Dict, Any
from pathlib import Path
import logging
from PIL import Image
import io
from .base_renderer import BaseRenderer, RenderConfig, RenderResult


logger = logging.getLogger(__name__)


class MathMLRenderer(BaseRenderer):
    """Render MathML to images using MathJax or other tools."""
    
    MATHJAX_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<script>
MathJax = {{
  startup: {{
    pageReady: () => {{
      return MathJax.startup.defaultPageReady().then(() => {{
        // Render complete
        if (window.callPhantom) {{
          window.callPhantom('rendered');
        }}
      }});
    }}
  }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-svg.js"></script>
<style>
body {{
  margin: 0;
  padding: {padding}px;
  background: {background};
  color: {color};
  font-size: {font_size}px;
}}
</style>
</head>
<body>
{content}
</body>
</html>
"""
    
    def __init__(self, renderer_backend: str = "mathjax"):
        self.renderer_backend = renderer_backend
        self._available = None
        
    def is_available(self) -> bool:
        """Check if renderer is available."""
        if self._available is None:
            if self.renderer_backend == "mathjax":
                self._available = self._check_node_mathjax()
            else:
                self._available = False
        return self._available
        
    def _check_node_mathjax(self) -> bool:
        """Check if Node.js and mathjax-node are available."""
        try:
            # Check Node.js
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                return False
                
            # Check if mathjax-node is installed globally or locally
            check_script = """
            try {
                require('mathjax-node');
                console.log('available');
            } catch(e) {
                console.log('not available');
            }
            """
            
            result = subprocess.run(
                ["node", "-e", check_script],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            return "available" in result.stdout
            
        except:
            return False
            
    def render_mathml(self, mathml: str, config: RenderConfig) -> RenderResult:
        """Render MathML to image."""
        if not self.is_available():
            return RenderResult(error=f"{self.renderer_backend} not available")
            
        if self.renderer_backend == "mathjax":
            return self._render_with_mathjax_node(mathml, config)
        else:
            return RenderResult(error=f"Unknown renderer: {self.renderer_backend}")
            
    def render_latex(self, latex: str, config: RenderConfig) -> RenderResult:
        """Convert LaTeX to MathML first, then render."""
        # This would require LaTeX to MathML conversion
        # For demonstration, we'll return an error
        return RenderResult(error="LaTeX rendering requires conversion to MathML first")
        
    def _render_with_mathjax_node(self, mathml: str, config: RenderConfig) -> RenderResult:
        """Render using mathjax-node."""
        try:
            # Create Node.js script
            script = f"""
            const mjAPI = require("mathjax-node");
            
            mjAPI.config({{
              MathJax: {{
                svg: {{
                  scale: 1,
                  minScale: .5,
                  mtextInheritFont: true,
                  merrorInheritFont: true,
                  mathmlSpacing: false,
                  skipAttributes: {{}},
                  exFactor: .5,
                  displayAlign: 'center',
                  displayIndent: '0',
                  fontCache: 'local',
                  localID: null,
                  internalSpeechTitles: true,
                  titleID: 0
                }}
              }}
            }});
            
            mjAPI.start();
            
            const mathml = `{mathml}`;
            
            mjAPI.typeset({{
              math: mathml,
              format: "MathML",
              svg: true,
              width: {config.max_width},
              ex: {config.font_size},
              linebreaks: true
            }}, function (data) {{
              if (!data.errors) {{
                console.log(JSON.stringify({{
                  svg: data.svg,
                  width: data.width,
                  height: data.height
                }}));
              }} else {{
                console.error(JSON.stringify({{error: data.errors.join(', ')}}));
                process.exit(1);
              }}
            }});
            """
            
            # Run script
            result = subprocess.run(
                ["node", "-e", script],
                capture_output=True,
                text=True,
                timeout=config.timeout
            )
            
            if result.returncode != 0:
                error = result.stderr or "MathJax rendering failed"
                return RenderResult(error=error)
                
            # Parse output
            import json
            output = json.loads(result.stdout)
            
            if "error" in output:
                return RenderResult(error=output["error"])
                
            # Convert SVG to image
            svg_content = output["svg"]
            
            # Save SVG if requested
            if config.output_format == "svg":
                return RenderResult(
                    svg=svg_content,
                    size=(int(float(output.get("width", 100))), 
                          int(float(output.get("height", 100)))),
                    metadata={"renderer": "mathjax-node"}
                )
                
            # Convert SVG to PNG
            image = self._svg_to_image(svg_content, config)
            if image:
                return RenderResult(
                    image=image,
                    size=image.size,
                    metadata={"renderer": "mathjax-node"}
                )
            else:
                return RenderResult(error="Failed to convert SVG to image")
                
        except Exception as e:
            logger.error(f"MathJax render error: {e}")
            return RenderResult(error=f"MathJax render error: {str(e)}")
            
    def _svg_to_image(self, svg_content: str, config: RenderConfig) -> Optional[Image.Image]:
        """Convert SVG to image."""
        try:
            # Try cairosvg
            import cairosvg
            
            png_data = cairosvg.svg2png(
                bytestring=svg_content.encode('utf-8'),
                dpi=config.dpi,
                scale=config.dpi/96.0  # Adjust for standard SVG DPI
            )
            
            return Image.open(io.BytesIO(png_data))
            
        except ImportError:
            logger.warning("cairosvg not available")
        except Exception as e:
            logger.warning(f"cairosvg failed: {e}")
            
        # Try svglib + reportlab
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM
            
            drawing = svg2rlg(io.BytesIO(svg_content.encode('utf-8')))
            if drawing:
                png_data = renderPM.drawToString(drawing, fmt='PNG', dpi=config.dpi)
                return Image.open(io.BytesIO(png_data))
                
        except ImportError:
            logger.warning("svglib not available")
        except Exception as e:
            logger.warning(f"svglib failed: {e}")
            
        return None
