import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from PIL import Image
from math_extraction.rendering import (
    MathRenderer, RenderConfig, RenderResult,
    LaTeXRenderer, MathMLRenderer, DiagramRenderer
)


class TestRenderConfig:
    
    def test_default_config(self):
        """Test default render configuration."""
        config = RenderConfig()
        assert config.dpi == 300
        assert config.font_size == 12
        assert config.background_color == "white"
        assert config.output_format == "png"


class TestLaTeXRenderer:
    
    @patch('subprocess.run')
    def test_is_available(self, mock_run):
        """Test LaTeX availability check."""
        # Mock successful latex check
        mock_run.return_value = Mock(returncode=0)
        
        renderer = LaTeXRenderer()
        assert renderer.is_available()
        
        # Mock failed latex check
        mock_run.return_value = Mock(returncode=1)
        renderer._available = None
        assert not renderer.is_available()
        
    @patch.object(LaTeXRenderer, 'is_available', return_value=True)
    @patch('subprocess.run')
    @patch.object(LaTeXRenderer, '_pdf_to_image')
    def test_render_latex(self, mock_pdf_to_image, mock_run, mock_available, temp_dir):
        """Test LaTeX rendering."""
        # Mock successful compilation
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        
        # Mock PDF to image conversion
        mock_image = Image.new('RGB', (100, 50))
        mock_pdf_to_image.return_value = mock_image
        
        renderer = LaTeXRenderer(cache_dir=temp_dir)
        config = RenderConfig()
        
        result = renderer.render_latex("x + y", config)
        
        assert result.is_valid
        assert result.image is not None
        assert result.size == (100, 50)
        
    @patch.object(LaTeXRenderer, 'is_available', return_value=False)
    def test_latex_not_available(self, mock_available):
        """Test when LaTeX is not available."""
        renderer = LaTeXRenderer()
        result = renderer.render_latex("x + y", RenderConfig())
        
        assert not result.is_valid
        assert result.error == "LaTeX not available"
        
    def test_extract_error_message(self):
        """Test LaTeX error message extraction."""
        renderer = LaTeXRenderer()
        
        output = """
        ! Undefined control sequence.
        l.5 \\unknowncommand
        The control sequence at the end of the top line
        """
        
        error_msg = renderer._extract_error_message(output)
        assert "Undefined control sequence" in error_msg
        
    @patch.object(LaTeXRenderer, 'is_available', return_value=True)
    def test_cache_functionality(self, mock_available, temp_dir):
        """Test caching functionality."""
        renderer = LaTeXRenderer(cache_dir=temp_dir)
        config = RenderConfig(cache_renders=True)
        
        with patch.object(renderer, '_pdf_to_image') as mock_convert:
            mock_convert.return_value = Image.new('RGB', (100, 50))
            
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=0)
                
                # First render
                result1 = renderer.render_latex("x + y", config)
                assert mock_run.call_count == 1
                
                # Second render (should use cache)
                result2 = renderer.render_latex("x + y", config)
                # Compilation should not be called again
                assert mock_run.call_count == 1


class TestMatplotlibRenderer:
    
    def test_is_available(self):
        """Test matplotlib availability."""
        renderer = MatplotlibRenderer()
        
        with patch('importlib.import_module'):
            assert renderer.is_available()
            
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.close')
    def test_render_latex(self, mock_close, mock_figure):
        """Test matplotlib rendering."""
        # Mock figure
        mock_fig = Mock()
        mock_fig.dpi = 100
        mock_fig.text.return_value = Mock(get_window_extent=Mock(
            return_value=Mock(width=100, height=50)
        ))
        mock_fig.canvas = Mock()
        mock_figure.return_value = mock_fig
        
        # Mock save
        from io import BytesIO
        buffer = BytesIO()
        # Create a small test image
        test_img = Image.new('RGB', (100, 50), color='white')
        test_img.save(buffer, format='PNG')
        buffer.seek(0)
        
        with patch.object(mock_fig, 'savefig'):
            with patch('PIL.Image.open', return_value=test_img):
                renderer = MatplotlibRenderer()
                result = renderer.render_latex("x + y", RenderConfig())
                
        assert result.is_valid
        assert result.image is not None


class TestMathMLRenderer:
    
    @patch('subprocess.run')
    def test_check_mathjax(self, mock_run):
        """Test MathJax availability check."""
        renderer = MathMLRenderer()
        
        # Mock node available
        mock_run.side_effect = [
            Mock(returncode=0),  # node --version
            Mock(returncode=0, stdout="available")  # mathjax check
        ]
        
        assert renderer._check_node_mathjax()
        
    def test_svg_to_image(self):
        """Test SVG to image conversion."""
        renderer = MathMLRenderer()
        
        svg_content = '<svg><rect width="100" height="50"/></svg>'
        
        with patch('cairosvg.svg2png') as mock_svg2png:
            # Mock PNG data
            mock_png = b'PNG data'
            mock_svg2png.return_value = mock_png
            
            with patch('PIL.Image.open') as mock_open:
                mock_open.return_value = Image.new('RGB', (100, 50))
                
                image = renderer._svg_to_image(svg_content, RenderConfig())
                
        assert image is not None


class TestDiagramRenderer:
    
    def test_is_available(self):
        """Test diagram renderer availability."""
        renderer = DiagramRenderer()
        
        with patch('importlib.import_module'):
            assert renderer.is_available()
            
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    def test_render_function_plot(self, mock_close, mock_subplots):
        """Test function plot rendering."""
        # Mock matplotlib
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        renderer = DiagramRenderer()
        
        plot_data = {
            'functions': [{'function': 'x**2', 'label': 'Parabola'}],
            'x_range': [-5, 5],
            'title': 'Test Plot'
        }
        
        with patch('PIL.Image.open') as mock_open:
            mock_open.return_value = Image.new('RGB', (800, 600))
            
            result = renderer.render_plot('function', plot_data, RenderConfig())
            
        assert result.is_valid
        assert result.metadata['plot_type'] == 'function'
        
    @patch('rdkit.Chem.MolFromSmiles')
    @patch('rdkit.Chem.Draw.MolToImage')
    def test_render_chemical_structure(self, mock_draw, mock_mol):
        """Test chemical structure rendering."""
        # Mock RDKit
        mock_mol.return_value = Mock()  # Valid molecule
        mock_draw.return_value = Image.new('RGB', (300, 300))
        
        renderer = DiagramRenderer()
        result = renderer.render_chemical_structure('CCO', RenderConfig())
        
        assert result.is_valid
        assert result.metadata['smiles'] == 'CCO'


class TestMathRenderer:
    
    def test_initialization(self):
        """Test MathRenderer initialization."""
        renderer = MathRenderer()
        
        assert renderer.config is not None
        assert len(renderer.latex_renderers) > 0
        assert renderer.mathml_renderer is not None
        assert renderer.diagram_renderer is not None
        
    @patch.object(LaTeXRenderer, 'is_available', return_value=True)
    @patch.object(LaTeXRenderer, 'render_latex')
    def test_render_latex(self, mock_render, mock_available):
        """Test LaTeX rendering through main interface."""
        mock_render.return_value = RenderResult(
            image=Image.new('RGB', (100, 50)),
            size=(100, 50)
        )
        
        renderer = MathRenderer()
        result = renderer.render_latex("x + y")
        
        assert result.is_valid
        mock_render.assert_called_once()
        
    def test_render_latex_fallback(self):
        """Test fallback rendering."""
        renderer = MathRenderer()
        
        # Mock all renderers failing except last one
        for i, r in enumerate(renderer.latex_renderers[:-1]):
            r.is_available = Mock(return_value=True)
            r.render_latex = Mock(return_value=RenderResult(error="Failed"))
            
        # Last renderer succeeds
        renderer.latex_renderers[-1].is_available = Mock(return_value=True)
        renderer.latex_renderers[-1].render_latex = Mock(
            return_value=RenderResult(image=Image.new('RGB', (100, 50)))
        )
        
        result = renderer.render_latex("x + y")
        assert result.is_valid
        
    def test_render_batch(self):
        """Test batch rendering."""
        renderer = MathRenderer()
        
        with patch.object(renderer, 'render_latex') as mock_render:
            mock_render.return_value = RenderResult(
                image=Image.new('RGB', (100, 50))
            )
            
            results = renderer.render_latex_batch(["x", "y", "z"])
            
            assert len(results) == 3
            assert mock_render.call_count == 3
            
    def test_get_available_renderers(self):
        """Test getting available renderers."""
        renderer = MathRenderer()
        
        available = renderer.get_available_renderers()
        
        assert 'latex' in available
        assert 'mathml' in available
        assert 'diagram' in available
        
        assert isinstance(available['latex'], list)
        assert isinstance(available['mathml'], bool)
        
    def test_render_diagram(self):
        """Test diagram rendering."""
        renderer = MathRenderer()
        
        with patch.object(renderer.diagram_renderer, 'render_chemical_structure') as mock_render:
            mock_render.return_value = RenderResult(
                image=Image.new('RGB', (300, 300))
            )
            
            result = renderer.render_diagram('chemical', {'smiles': 'CCO'})
            
            assert result.is_valid
            mock_render.assert_called_once_with('CCO', renderer.config)


class TestRenderResult:
    
    def test_save_image(self, temp_dir):
        """Test saving render result."""
        image = Image.new('RGB', (100, 50), color='red')
        result = RenderResult(image=image, size=(100, 50))
        
        output_path = temp_dir / "test.png"
        result.save(output_assert output_path.exists()
       
       # Load and verify
       loaded = Image.open(output_path)
       assert loaded.size == (100, 50)
       
   def test_save_svg(self, temp_dir):
       """Test saving SVG result."""
       svg_content = '<svg><circle cx="50" cy="50" r="40"/></svg>'
       result = RenderResult(svg=svg_content)
       
       output_path = temp_dir / "test.svg"
       result.save(output_path)
       
       assert output_path.exists()
       assert output_path.read_text() == svg_content
       
   def test_save_pdf(self, temp_dir):
       """Test saving PDF result."""
       pdf_bytes = b'%PDF-1.4 test content'
       result = RenderResult(pdf_bytes=pdf_bytes)
       
       output_path = temp_dir / "test.pdf"
       result.save(output_path)
       
       assert output_path.exists()
       assert output_path.read_bytes() == pdf_bytes
       
   def test_save_no_content(self, temp_dir):
       """Test saving with no content."""
       result = RenderResult(error="Failed")
       
       output_path = temp_dir / "test.png"
       
       with pytest.raises(ValueError, match="No render result to save"):
           result.save(output_path)
