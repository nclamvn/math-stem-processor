import tempfile
import subprocess
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
from PIL import Image
import io
from .base_renderer import BaseRenderer, RenderConfig, RenderResult


logger = logging.getLogger(__name__)


class DiagramRenderer(BaseRenderer):
    """Render scientific diagrams and plots."""
    
    def __init__(self):
        self._available = None
        
    def is_available(self) -> bool:
        """Check if required libraries are available."""
        if self._available is None:
            try:
                import matplotlib
                import numpy as np
                self._available = True
            except ImportError:
                self._available = False
        return self._available
        
    def render_latex(self, latex: str, config: RenderConfig) -> RenderResult:
        """Not applicable for diagram renderer."""
        return RenderResult(error="Diagram renderer does not support LaTeX input")
        
    def render_mathml(self, mathml: str, config: RenderConfig) -> RenderResult:
        """Not applicable for diagram renderer."""
        return RenderResult(error="Diagram renderer does not support MathML input")
        
    def render_plot(self, plot_type: str, data: Dict[str, Any], 
                   config: RenderConfig) -> RenderResult:
        """Render various types of plots."""
        if not self.is_available():
            return RenderResult(error="Required libraries not available")
            
        try:
            if plot_type == "function":
                return self._render_function_plot(data, config)
            elif plot_type == "scatter":
                return self._render_scatter_plot(data, config)
            elif plot_type == "bar":
                return self._render_bar_plot(data, config)
            elif plot_type == "histogram":
                return self._render_histogram(data, config)
            elif plot_type == "contour":
                return self._render_contour_plot(data, config)
            elif plot_type == "vector_field":
                return self._render_vector_field(data, config)
            else:
                return RenderResult(error=f"Unknown plot type: {plot_type}")
                
        except Exception as e:
            logger.error(f"Plot rendering error: {e}")
            return RenderResult(error=f"Plot rendering error: {str(e)}")
            
    def _render_function_plot(self, data: Dict[str, Any], config: RenderConfig) -> RenderResult:
        """Render function plot."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract parameters
        functions = data.get('functions', [])
        x_range = data.get('x_range', [-10, 10])
        y_range = data.get('y_range', None)
        title = data.get('title', '')
        x_label = data.get('x_label', 'x')
        y_label = data.get('y_label', 'y')
        grid = data.get('grid', True)
        legend = data.get('legend', True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6), dpi=config.dpi)
        
        # Generate x values
        x = np.linspace(x_range[0], x_range[1], 1000)
        
        # Plot functions
        for func_data in functions:
            if isinstance(func_data, dict):
                func = func_data.get('function')
                label = func_data.get('label', '')
                style = func_data.get('style', '-')
                color = func_data.get('color', None)
            else:
                func = func_data
                label = str(func)
                style = '-'
                color = None
                
            # Evaluate function
            try:
                if callable(func):
                    y = func(x)
                else:
                    # Parse string expression
                    import sympy as sp
                    x_sym = sp.Symbol('x')
                    expr = sp.sympify(func)
                    func_lambdified = sp.lambdify(x_sym, expr, 'numpy')
                    y = func_lambdified(x)
                    
                ax.plot(x, y, style, color=color, label=label, linewidth=2)
                
            except Exception as e:
                logger.warning(f"Failed to plot function: {e}")
                
        # Set labels and title
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14)
        
        # Set ranges
        if y_range:
            ax.set_ylim(y_range)
            
        # Add grid
        if grid:
            ax.grid(True, alpha=0.3)
            
        # Add legend
        if legend and len(functions) > 1:
            ax.legend()
            
        # Tight layout
        fig.tight_layout()
        
        # Save to buffer
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format='png',
            dpi=config.dpi,
            transparent=config.transparent_background,
            bbox_inches='tight'
        )
        
        plt.close(fig)
        
        # Load image
        buffer.seek(0)
        image = Image.open(buffer)
        
        return RenderResult(
            image=image,
            size=image.size,
            metadata={"renderer": "matplotlib", "plot_type": "function"}
        )
        
    def _render_scatter_plot(self, data: Dict[str, Any], config: RenderConfig) -> RenderResult:
        """Render scatter plot."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract data
        x_data = np.array(data.get('x', []))
        y_data = np.array(data.get('y', []))
        
        if len(x_data) != len(y_data):
            return RenderResult(error="x and y data must have same length")
            
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6), dpi=config.dpi)
        
        # Create scatter plot
        scatter = ax.scatter(
            x_data, y_data,
            c=data.get('colors', 'blue'),
            s=data.get('sizes', 50),
            alpha=data.get('alpha', 0.7),
            marker=data.get('marker', 'o')
        )
        
        # Add labels
        ax.set_xlabel(data.get('x_label', 'X'), fontsize=12)
        ax.set_ylabel(data.get('y_label', 'Y'), fontsize=12)
        ax.set_title(data.get('title', 'Scatter Plot'), fontsize=14)
        
        # Add grid
        if data.get('grid', True):
            ax.grid(True, alpha=0.3)
            
        # Add colorbar if colors vary
        if isinstance(data.get('colors'), (list, np.ndarray)):
            plt.colorbar(scatter)
            
        # Tight layout
        fig.tight_layout()
        
        # Save to buffer
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format='png',
            dpi=config.dpi,
            transparent=config.transparent_background,
            bbox_inches='tight'
        )
        
        plt.close(fig)
        
        # Load image
        buffer.seek(0)
        image = Image.open(buffer)
        
        return RenderResult(
            image=image,
            size=image.size,
            metadata={"renderer": "matplotlib", "plot_type": "scatter"}
        )
        
    def _render_vector_field(self, data: Dict[str, Any], config: RenderConfig) -> RenderResult:
        """Render vector field."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract parameters
        x_range = data.get('x_range', [-5, 5])
        y_range = data.get('y_range', [-5, 5])
        density = data.get('density', 20)
        
        # Create grid
        x = np.linspace(x_range[0], x_range[1], density)
        y = np.linspace(y_range[0], y_range[1], density)
        X, Y = np.meshgrid(x, y)
        
        # Get vector components
        if 'u_func' in data and 'v_func' in data:
            U = data['u_func'](X, Y)
            V = data['v_func'](X, Y)
        elif 'U' in data and 'V' in data:
            U = np.array(data['U'])
            V = np.array(data['V'])
        else:
            return RenderResult(error="Vector field data not provided")
            
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6), dpi=config.dpi)
        
        # Create quiver plot
        Q = ax.quiver(X, Y, U, V, 
                     color=data.get('color', 'blue'),
                     alpha=data.get('alpha', 0.7))
        
        # Add quiver key
        if data.get('show_key', True):
            ax.quiverkey(Q, X=0.9, Y=0.9, U=1, label='1 unit', labelpos='E')
            
        # Set labels and title
        ax.set_xlabel(data.get('x_label', 'x'), fontsize=12)
        ax.set_ylabel(data.get('y_label', 'y'), fontsize=12)
        ax.set_title(data.get('title', 'Vector Field'), fontsize=14)
        
        # Add grid
        if data.get('grid', True):
            ax.grid(True, alpha=0.3)
            
        # Equal aspect ratio
        ax.set_aspect('equal')
        
        # Tight layout
        fig.tight_layout()
        
        # Save to buffer
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format='png',
            dpi=config.dpi,
            transparent=config.transparent_background,
            bbox_inches='tight'
        )
        
        plt.close(fig)
        
        # Load image
        buffer.seek(0)
        image = Image.open(buffer)
        
        return RenderResult(
            image=image,
            size=image.size,
            metadata={"renderer": "matplotlib", "plot_type": "vector_field"}
        )
        
    def render_chemical_structure(self, smiles: str, config: RenderConfig) -> RenderResult:
        """Render chemical structure from SMILES."""
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return RenderResult(error="Invalid SMILES string")
                
            # Generate image
            img = Draw.MolToImage(
                mol,
                size=(config.max_width, config.max_height),
                kekulize=True,
                wedgeBonds=True,
                imageType='png'
            )
            
            return RenderResult(
                image=img,
                size=img.size,
                metadata={"renderer": "rdkit", "smiles": smiles}
            )
            
        except ImportError:
            return RenderResult(error="RDKit not available")
        except Exception as e:
            logger.error(f"Chemical structure rendering error: {e}")
            return RenderResult(error=f"Chemical structure rendering error: {str(e)}")
