import pytest
from math_extraction.latex_parser import LaTeXParser, LaTeXTokenType


class TestLaTeXParser:
    
    def test_initialization(self):
        """Test LaTeXParser initialization."""
        parser = LaTeXParser()
        assert parser.operator_trie is not None
        assert len(parser.GREEK_LETTERS) > 0
        assert len(parser.MATH_OPERATORS) > 0
        
    def test_tokenize_simple(self, latex_parser):
        """Test tokenization of simple expressions."""
        latex = "x + y = z"
        structure = latex_parser.parse(latex)
        
        assert structure.is_valid
        assert len(structure.tokens) >= 5  # x, +, y, =, z
        
        # Check token types
        tokens_by_type = {t.type for t in structure.tokens}
        assert LaTeXTokenType.VARIABLE in tokens_by_type
        assert LaTeXTokenType.OPERATOR in tokens_by_type
        
    def test_tokenize_commands(self, latex_parser):
        """Test tokenization of LaTeX commands."""
        latex = r"\frac{a}{b} + \sqrt{x}"
        structure = latex_parser.parse(latex)
        
        assert structure.is_valid
        
        # Find command tokens
        commands = [t for t in structure.tokens if t.type == LaTeXTokenType.COMMAND]
        command_values = {t.value for t in commands}
        
        assert r'\frac' in command_values
        assert r'\sqrt' in command_values
        
    def test_tokenize_greek_letters(self, latex_parser):
        """Test tokenization of Greek letters."""
        latex = r"\alpha + \beta = \gamma"
        structure = latex_parser.parse(latex)
        
        assert structure.is_valid
        
        # Find Greek letter tokens
        greek = [t for t in structure.tokens 
                if t.type == LaTeXTokenType.SYMBOL and 
                t.metadata.get('subtype') == 'greek']
        
        assert len(greek) == 3
        greek_values = {t.value for t in greek}
        assert r'\alpha' in greek_values
        assert r'\beta' in greek_values
        assert r'\gamma' in greek_values
        
    def test_tokenize_subscripts_superscripts(self, latex_parser):
        """Test tokenization of subscripts and superscripts."""
        latex = "x_1 + y^2 + z_i^j"
        structure = latex_parser.parse(latex)
        
        assert structure.is_valid
        
        # Count script tokens
        subscripts = [t for t in structure.tokens if t.type == LaTeXTokenType.SUBSCRIPT]
        superscripts = [t for t in structure.tokens if t.type == LaTeXTokenType.SUPERSCRIPT]
        
        assert len(subscripts) >= 2
        assert len(superscripts) >= 2
        
    def test_tokenize_environments(self, latex_parser):
        """Test tokenization of environments."""
        latex = r"\begin{align} x &= 1 \\ y &= 2 \end{align}"
        structure = latex_parser.parse(latex)
        
        assert structure.is_valid
        
        # Find environment tokens
        envs = [t for t in structure.tokens if t.type == LaTeXTokenType.ENVIRONMENT]
        assert len(envs) == 2  # begin and end
        
        begin_env = next(t for t in envs if t.metadata.get('env_type') == 'begin')
        assert begin_env.metadata['env_name'] == 'align'
        
    def test_build_tree(self, latex_parser):
        """Test AST building."""
        latex = r"\frac{a + b}{c}"
        structure = latex_parser.parse(latex)
        
        assert structure.is_valid
        assert structure.tree is not None
        assert 'children' in structure.tree
        assert len(structure.tree['children']) > 0
        
    def test_validate_structure(self, latex_parser):
        """Test structure validation."""
        # Valid structure
        valid_latex = r"\frac{a}{b}"
        valid_structure = latex_parser.parse(valid_latex)
        assert valid_structure.is_valid
        assert len(valid_structure.errors) == 0
        
        # Invalid structure - unmatched braces
        invalid_latex = r"\frac{a}{b"
        invalid_structure = latex_parser.parse(invalid_latex)
        assert not invalid_structure.is_valid
        assert len(invalid_structure.errors) > 0
        
    def test_calculate_complexity(self, latex_parser):
        """Test complexity calculation."""
        # Simple expression
        simple = "x + y"
        simple_structure = latex_parser.parse(simple)
        assert simple_structure.complexity_score < 2.0
        
        # Complex expression
        complex_latex = r"\int_0^\infty \frac{e^{-x^2}}{\sqrt{\pi}} \, dx"
        complex_structure = latex_parser.parse(complex_latex)
        assert complex_structure.complexity_score > simple_structure.complexity_score
        
    def test_extract_components(self, latex_parser):
        """Test component extraction."""
        latex = r"\frac{\alpha + \beta}{\gamma} = \sum_{i=1}^n x_i"
        components = latex_parser.extract_components(latex)
        
        assert 'greek_letters' in components
        assert len(components['greek_letters']) >= 3
        
        assert 'fractions' in components
        assert len(components['fractions']) >= 1
        
        assert 'sums_products' in components
        assert len(components['sums_products']) >= 1
        
    def test_normalize_latex(self, latex_parser):
        """Test LaTeX normalization."""
        # Test with extra spaces
        latex = r"x  +  y  =  z"
        normalized = latex_parser.normalize_latex(latex)
        assert normalized == "x + y = z"
        
        # Test with spacing commands
        latex2 = r"x\,+\,y"
        normalized2 = latex_parser.normalize_latex(latex2, preserve_spacing=False)
        assert r'\,' not in normalized2
        
    def test_optional_arguments(self, latex_parser):
        """Test parsing of optional arguments."""
        latex = r"\sqrt[3]{x}"
        structure = latex_parser.parse(latex)
        
        assert structure.is_valid
        
        # Find optional argument token
        optional = [t for t in structure.tokens if t.type == LaTeXTokenType.OPTIONAL]
        assert len(optional) >= 1
        assert optional[0].metadata.get('content') == '3'
        
    def test_nested_scripts(self, latex_parser):
        """Test nested subscripts/superscripts."""
        latex = r"x^{y^z}"
        structure = latex_parser.parse(latex)
        
        assert structure.is_valid
        # Should handle nested scripts without error
        
    def test_macro_detection(self, latex_parser):
        """Test detection of user-defined macros."""
        latex = r"\newcommand{\vect}[1]{\mathbf{#1}} \vect{x}"
        structure = latex_parser.parse(latex)
        
        # Parser should recognize \newcommand
        commands = [t for t in structure.tokens if t.value == r'\newcommand']
        assert len(commands) == 1
        
    def test_comment_removal(self, latex_parser):
        """Test LaTeX comment removal."""
        latex = "x + y % this is a comment\n+ z"
        structure = latex_parser.parse(latex, remove_comments=True)
        
        # Comment should be removed
        text_content = ''.join(t.value for t in structure.tokens 
                              if t.type != LaTeXTokenType.COMMENT)
        assert "this is a comment" not in text_content
        
    def test_delimiter_matching(self, latex_parser):
        """Test delimiter matching validation."""
        # Matched delimiters
        latex1 = r"\left( \frac{a}{b} \right)"
        structure1 = latex_parser.parse(latex1)
        assert structure1.is_valid
        
        # Unmatched delimiters
        latex2 = r"\left( \frac{a}{b} \right]"
        structure2 = latex_parser.parse(latex2)
        # Should have warnings about mismatched delimiters
        assert len(structure2.warnings) > 0 or len(structure2.errors) > 0
        
    def test_get_hash(self, latex_parser):
        """Test LaTeX hashing."""
        latex1 = "x + y = z"
        latex2 = "x+y=z"  # Different spacing
        latex3 = "a + b = c"  # Different content
        
        hash1 = latex_parser.get_hash(latex1)
        hash2 = latex_parser.get_hash(latex2)
        hash3 = latex_parser.get_hash(latex3)
        
        # After normalization, similar formulas should have same hash
        assert hash1 == hash2
        # Different formulas should have different hashes
        assert hash1 != hash3
        
    def test_is_simple_expression(self, latex_parser):
        """Test simple expression detection."""
        # Simple expressions
        assert latex_parser.is_simple_expression("x + y")
        assert latex_parser.is_simple_expression("2x^2 + 3x + 1")
        
        # Complex expressions
        assert not latex_parser.is_simple_expression(r"\frac{a}{b}")
        assert not latex_parser.is_simple_expression(r"\begin{matrix} a & b \end{matrix}")


class TestEdgeCases:
    
    def test_empty_input(self, latex_parser):
        """Test with empty input."""
        structure = latex_parser.parse("")
        assert structure.is_valid
        assert len(structure.tokens) == 0
        
    def test_only_whitespace(self, latex_parser):
        """Test with only whitespace."""
        structure = latex_parser.parse("   \n\t  ")
        assert structure.is_valid
        # Should only have spacing tokens
        
    def test_invalid_commands(self, latex_parser):
        """Test with invalid/unknown commands."""
        latex = r"\unknowncommand{x}"
        structure = latex_parser.parse(latex)
        
        # Should have warnings about unknown command
        assert len(structure.warnings) > 0
        unknown_warnings = [w for w in structure.warnings 
                           if w['type'] == 'unknown_command']
        assert len(unknown_warnings) > 0
        
    def test_deeply_nested(self, latex_parser):
        """Test deeply nested structures."""
        # Create deeply nested fractions
        latex = r"\frac{1}{\frac{2}{\frac{3}{\frac{4}{5}}}}"
        structure = latex_parser.parse(latex)
        
        assert structure.is_valid
        assert structure.complexity_score > 3.0
        assert structure.metadata['tree_depth'] > 5
